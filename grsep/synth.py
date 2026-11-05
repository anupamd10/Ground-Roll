from dataclasses import dataclass, field, replace

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq


@dataclass
class Geometry:
    nt: int = 1000
    nx: int = 200
    dt: float = 0.002
    dx: float = 10.0
    src: float | None = None

    @property
    def t(self):
        return np.arange(self.nt) * self.dt

    @property
    def offsets(self):
        xc = (self.nx - 1) / 2 if self.src is None else self.src
        return (np.arange(self.nx) - xc) * self.dx


@dataclass
class ReflectionParams:
    n_hyperbolas: tuple = (15, 30)
    t0: tuple = (0.1, 1.9)
    vrms: tuple = (1600.0, 4500.0)
    f_peak: tuple = (25.0, 50.0)
    direct_v: tuple = (1800.0, 2600.0)
    direct_amp: float = 1.5


@dataclass
class GroundRollParams:
    v: tuple = (200.0, 500.0)
    fb: float | tuple = 5.0
    fe: float | tuple = 15.0
    n_modes: tuple = (1, 2)
    t0_len: tuple = (0.10, 0.20)
    t_slope: tuple = (0.3e-3, 0.8e-3)
    decay: tuple = (0.985, 0.998)
    snr_db: tuple = (-14.0, -8.0)


@dataclass
class GatherParams:
    geom: Geometry = field(default_factory=Geometry)
    refl: ReflectionParams = field(default_factory=ReflectionParams)
    gr: GroundRollParams = field(default_factory=GroundRollParams)
    noise_ratio: float | tuple = 0.0

    def with_(self, **kw):
        out = replace(self)
        for k, v in kw.items():
            if k in ("nt", "nx", "dt", "dx", "src"):
                out.geom = replace(out.geom, **{k: v})
            elif k in ("fb", "fe", "snr_db", "v", "decay", "t0_len", "t_slope", "n_modes"):
                out.gr = replace(out.gr, **{k: v})
            elif k in ("f_peak", "vrms", "n_hyperbolas"):
                out.refl = replace(out.refl, **{k: v})
            else:
                setattr(out, k, v)
        return out


def _u(rng, rng_or_val):
    if isinstance(rng_or_val, tuple):
        return rng.uniform(*rng_or_val)
    return rng_or_val


def ricker_spectrum(f, f_peak):
    return (2.0 / np.sqrt(np.pi)) * f**2 / f_peak**3 * np.exp(-(f**2) / f_peak**2)


def events_to_gather(geom, times, amps, f_peak):
    nfft = next_fast_len(2 * geom.nt)
    f = rfftfreq(nfft, geom.dt)
    w = ricker_spectrum(f, f_peak)
    spec = np.zeros((f.size, geom.nx), dtype=np.complex128)
    for tx, a in zip(times, amps):
        spec += a * np.exp(-2j * np.pi * f[:, None] * tx[None, :])
    out = irfft(spec * w[:, None], nfft, axis=0)[: geom.nt]
    return out / geom.dt


def reflections(geom, p, rng):
    x = geom.offsets
    f_peak = _u(rng, p.f_peak)
    n = int(rng.integers(p.n_hyperbolas[0], p.n_hyperbolas[1] + 1))
    times, amps = [], []
    for _ in range(n):
        t0 = _u(rng, p.t0)
        v = _u(rng, p.vrms)
        times.append(np.sqrt(t0**2 + (x / v) ** 2))
        amps.append(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0))
    vd = _u(rng, p.direct_v)
    times.append(np.abs(x) / vd + 0.02)
    amps.append(p.direct_amp)
    return events_to_gather(geom, times, amps, f_peak)


def tukey01(u, alpha=0.5):
    w = np.ones_like(u)
    a = alpha / 2
    left = u < a
    right = u > 1 - a
    w[left] = 0.5 * (1 - np.cos(np.pi * u[left] / a))
    w[right] = 0.5 * (1 - np.cos(np.pi * (1 - u[right]) / a))
    w[(u < 0) | (u > 1)] = 0.0
    return w


def ground_roll_mode(geom, v, fb, fe, t_len0, t_slope, decay):
    t = geom.t[:, None]
    ax = np.abs(geom.offsets)[None, :]
    t_on = ax / v
    T = t_len0 + t_slope * ax
    tau = t - t_on
    u = tau / T
    w = tukey01(u)
    phase = 2 * np.pi * (fb * tau + (fe - fb) / (2 * T) * tau**2)
    att = decay ** (ax / geom.dx)
    return w * np.sin(phase) * att


def _sweep(rng, p):
    fb = _u(rng, p.fb)
    fe = _u(rng, p.fe)
    return (fb, fe) if fe > fb else (fe, fb)


def ground_roll(geom, p, rng):
    n = int(rng.integers(p.n_modes[0], p.n_modes[1] + 1))
    g = np.zeros((geom.nt, geom.nx))
    for _ in range(n):
        g += rng.uniform(0.5, 1.0) * ground_roll_mode(
            geom,
            _u(rng, p.v),
            *_sweep(rng, p),
            _u(rng, p.t0_len),
            _u(rng, p.t_slope),
            _u(rng, p.decay),
        )
    return g


def energy(a):
    return float(np.sum(a.astype(np.float64) ** 2))


def make_gather(params, rng):
    geom = params.geom
    r = reflections(geom, params.refl, rng)
    g = ground_roll(geom, params.gr, rng)
    snr = _u(rng, params.gr.snr_db)
    g *= np.sqrt(energy(r) / (energy(g) * 10 ** (snr / 10)))
    n = np.zeros_like(r)
    p = _u(rng, params.noise_ratio)
    if p > 0:
        clean = r + g
        n = rng.standard_normal(clean.shape)
        n *= np.sqrt(p / (1 - p) * energy(clean) / energy(n))
    m = r + g + n
    return {
        "m": m.astype(np.float32),
        "r": r.astype(np.float32),
        "g": g.astype(np.float32),
        "n": n.astype(np.float32),
        "dt": geom.dt,
        "dx": geom.dx,
    }

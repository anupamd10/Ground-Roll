from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq


@dataclass(frozen=True)
class Band:
    f_pass: float = 25.0
    f_stop: float = 35.0

    def __post_init__(self):
        if not 0 < self.f_pass < self.f_stop:
            raise ValueError("need 0 < f_pass < f_stop")


def lowpass_response(f, band):
    h = np.zeros_like(f)
    h[f <= band.f_pass] = 1.0
    mid = (f > band.f_pass) & (f < band.f_stop)
    h[mid] = 0.5 * (1 + np.cos(np.pi * (f[mid] - band.f_pass) / (band.f_stop - band.f_pass)))
    return h


def lowpass(x, dt, band, axis=0):
    x = np.asarray(x)
    nt = x.shape[axis]
    if band.f_stop >= 0.5 / dt:
        raise ValueError(f"f_stop={band.f_stop} Hz is above Nyquist {0.5 / dt} Hz")
    nfft = next_fast_len(2 * nt)
    spec = rfft(x, nfft, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = -1
    h = lowpass_response(rfftfreq(nfft, dt), band).reshape(shape)
    out = irfft(spec * h, nfft, axis=axis)
    return np.take(out, np.arange(nt), axis=axis).astype(x.dtype)


def split(x, dt, band, axis=0):
    low = lowpass(x, dt, band, axis)
    return low, x - low


def leakage_ratio(g, dt, band):
    _, high = split(g, dt, band)
    return float(np.sum(high.astype(np.float64) ** 2) / max(np.sum(g.astype(np.float64) ** 2), 1e-30))


def suggest_band(g_or_m, dt, energy_frac=0.995, margin=10.0, width=10.0):
    nfft = next_fast_len(2 * g_or_m.shape[0])
    p = np.sum(np.abs(rfft(g_or_m, nfft, axis=0)) ** 2, axis=1)
    f = rfftfreq(nfft, dt)
    c = np.cumsum(p) / p.sum()
    f99 = float(f[np.searchsorted(c, energy_frac)])
    return Band(f99 + margin, f99 + margin + width), f99

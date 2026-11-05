import numpy as np
from scipy.fft import fft2, fftfreq, ifft2, next_fast_len

from .bandsplit import split
from .metrics import snr_db


def _cos_ramp(x, a, b):
    return np.clip(0.5 * (1 - np.cos(np.pi * np.clip((x - a) / (b - a), 0, 1))), 0, 1)


def fk_dip_filter(m, dt, dx, v_cut, f_max=40.0, v_taper=0.3, f_taper=10.0):
    nt, nx = m.shape
    nt2, nx2 = next_fast_len(2 * nt), next_fast_len(2 * nx)
    spec = fft2(m, (nt2, nx2))
    f = np.abs(fftfreq(nt2, dt))[:, None]
    k = np.abs(fftfreq(nx2, dx))[None, :]
    v_app = f / np.maximum(k, 1e-12)
    keep_v = _cos_ramp(v_app, v_cut, v_cut * (1 + v_taper))
    keep_f = _cos_ramp(f, f_max, f_max + f_taper)
    reject = (1 - keep_v) * (1 - keep_f)
    out = np.real(ifft2(spec * (1 - reject)))[:nt, :nx]
    return out.astype(np.float32)


def highpass_filter(m, dt, band):
    return split(m, dt, band)[1]


def tune_fk(m, r, dt, dx, v_grid=None, f_grid=None):
    v_grid = v_grid if v_grid is not None else np.arange(300, 2001, 100)
    f_grid = f_grid if f_grid is not None else (20.0, 30.0, 40.0)
    best = (-np.inf, None, None)
    for f_max in f_grid:
        for v in v_grid:
            s = snr_db(r, fk_dip_filter(m, dt, dx, v, f_max))
            if s > best[0]:
                best = (s, v, f_max)
    return best

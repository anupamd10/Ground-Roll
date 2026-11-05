import numpy as np
from scipy.fft import fft2, fftshift, fftfreq, next_fast_len


def snr_db(ref, est):
    ref = np.asarray(ref, np.float64)
    err = ref - np.asarray(est, np.float64)
    return float(10 * np.log10(np.sum(ref**2) / max(np.sum(err**2), 1e-30)))


def fk_spectrum(x, dt, dx):
    nt2, nx2 = next_fast_len(2 * x.shape[0]), next_fast_len(2 * x.shape[1])
    a = np.abs(fftshift(fft2(x, (nt2, nx2)), axes=1))[: nt2 // 2 + 1]
    f = np.abs(fftfreq(nt2, dt))[: nt2 // 2 + 1]
    k = fftshift(fftfreq(nx2, dx))
    return a, f, k

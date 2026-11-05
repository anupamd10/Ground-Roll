import numpy as np
import torch

from .bandsplit import split
from .data import band_rms, patch_scale, positions


def blend_window(size):
    w = np.sin(np.pi * (np.arange(size) + 0.5) / size)
    return np.outer(w, w).astype(np.float32)


@torch.no_grad()
def separate(model, m, dt, band, size=64, stride=16, batch=128, device="cpu"):
    m = np.asarray(m, dtype=np.float32)
    nt, nx = m.shape
    pt, px = max(0, size - nt), max(0, size - nx)
    mp = np.pad(m, ((0, pt), (0, px)), mode="reflect") if (pt or px) else m
    m_low, m_high = split(mp, dt, band)
    x = np.stack([m_low, m_high])
    floor = 1e-3 * band_rms(x) + 1e-12
    w = blend_window(size)
    acc = np.zeros((2,) + mp.shape, dtype=np.float64)
    wsum = np.zeros(mp.shape, dtype=np.float64)
    coords = [(i, j) for i in positions(mp.shape[0], size, stride) for j in positions(mp.shape[1], size, stride)]
    model.eval()
    for b in range(0, len(coords), batch):
        chunk = coords[b : b + batch]
        patches, scales = [], []
        for i, j in chunk:
            p = x[:, i : i + size, j : j + size]
            sc = patch_scale(p, floor)
            patches.append(p / sc)
            scales.append(sc)
        xb = torch.from_numpy(np.stack(patches)).to(device)
        r_hat, g_hat = model(xb)
        out = torch.cat([r_hat, g_hat], 1).cpu().numpy()
        for (i, j), o, sc in zip(chunk, out, scales):
            acc[:, i : i + size, j : j + size] += o * sc * w
            wsum[i : i + size, j : j + size] += w
    acc /= wsum
    r_low, g_hat = acc[0][:nt, :nx], acc[1][:nt, :nx]
    m_high = m_high[:nt, :nx]
    return {
        "r": (r_low + m_high).astype(np.float32),
        "g": g_hat.astype(np.float32),
        "r_low": r_low.astype(np.float32),
        "m_low": m_low[:nt, :nx].astype(np.float32),
        "m_high": m_high.astype(np.float32),
    }

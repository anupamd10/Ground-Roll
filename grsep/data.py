import numpy as np
import torch
from torch.utils.data import Dataset

from .bandsplit import split


def positions(n, size, stride):
    if n < size:
        raise ValueError(f"axis length {n} is smaller than patch size {size}")
    pos = list(range(0, n - size + 1, stride))
    if pos[-1] != n - size:
        pos.append(n - size)
    return pos


def patch_count(shape, size, stride):
    return len(positions(shape[0], size, stride)) * len(positions(shape[1], size, stride))


def prepare(sample, band):
    dt = sample["dt"]
    m_low, m_high = split(sample["m"], dt, band)
    if "g" in sample:
        g_low, _ = split(sample["g"], dt, band)
        r_low = m_low - g_low
    else:
        r_low, _ = split(sample["r"], dt, band)
        g_low = m_low - r_low
    x = np.stack([m_low, m_high]).astype(np.float32)
    y = np.stack([r_low, g_low]).astype(np.float32)
    return x, y


def band_rms(x):
    return float(np.sqrt(np.mean(x[0] ** 2 + x[1] ** 2) / 2))


def patch_scale(x, floor):
    return max(band_rms(x), floor)


class PatchDataset(Dataset):
    def __init__(self, samples, band, size=64, stride=10, augment=False):
        self.size = size
        self.augment = augment
        self.items = [prepare(s, band) for s in samples]
        self.floors = [1e-3 * band_rms(x) + 1e-12 for x, _ in self.items]
        self.index = [
            (k, i, j)
            for k, (x, _) in enumerate(self.items)
            for i in positions(x.shape[1], size, stride)
            for j in positions(x.shape[2], size, stride)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        k, i, j = self.index[n]
        x, y = self.items[k]
        s = self.size
        xp = x[:, i : i + s, j : j + s]
        yp = y[:, i : i + s, j : j + s]
        sc = patch_scale(xp, self.floors[k])
        xp = xp / sc
        yp = yp / sc
        if self.augment:
            if np.random.rand() < 0.5:
                xp, yp = xp[:, :, ::-1], yp[:, :, ::-1]
            if np.random.rand() < 0.5:
                xp, yp = -xp, -yp
        return torch.from_numpy(np.ascontiguousarray(xp)), torch.from_numpy(np.ascontiguousarray(yp))

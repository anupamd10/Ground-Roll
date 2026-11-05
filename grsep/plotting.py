import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import fk_spectrum


def _tx(ax, d, dt, title, clip):
    nt, nx = d.shape
    ax.imshow(d, cmap="gray", aspect="auto", vmin=-clip, vmax=clip, extent=[0, nx, nt * dt * 1000, 0])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("trace")
    ax.set_ylabel("time (ms)")


def _fk(ax, d, dt, dx, fmax, vmax):
    a, f, k = fk_spectrum(d, dt, dx)
    sel = f <= fmax
    ax.imshow(a[sel], aspect="auto", origin="lower", cmap="turbo", vmin=0, vmax=vmax, extent=[k[0], k[-1], 0, f[sel][-1]])
    ax.set_xlabel("k (1/m)")
    ax.set_ylabel("f (Hz)")


def panel(path, panels, dt, dx, fmax=80.0, suptitle=None):
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 7), constrained_layout=True)
    ref = panels[0][1]
    clip = np.percentile(np.abs(ref), 98)
    vmax = np.percentile(fk_spectrum(ref, dt, dx)[0], 99.5)
    for c, (title, d) in enumerate(panels):
        _tx(axes[0, c], d, dt, title, clip)
        _fk(axes[1, c], d, dt, dx, fmax, vmax)
    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def loss_curve(path, history):
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_sim"] for h in history], label="train similarity")
    ax.plot(ep, [h["val_sim"] for h in history], label="val similarity")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.legend()
    fig.savefig(path, dpi=110)
    plt.close(fig)

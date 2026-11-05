import argparse
from pathlib import Path

import numpy as np
import torch

from grsep import load_model, separate
from grsep.plotting import panel


def read_input(path, dt, dx):
    path = Path(path)
    if path.suffix in (".sgy", ".segy"):
        import segyio

        with segyio.open(str(path), ignore_geometry=True) as f:
            m = segyio.tools.collect(f.trace[:]).T.astype(np.float32)
            return m, dt or segyio.tools.dt(f) / 1e6, dx
    if path.suffix == ".npz":
        d = np.load(path)
        dt = dt or (float(d["dt"]) if "dt" in d else None)
        dx = float(d["dx"]) if "dx" in d else dx
        return d["m"].astype(np.float32), dt, dx
    return np.load(path).astype(np.float32), dt, dx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help=".npy (nt x nx), .npz (key m, dt), or .sgy shot gather")
    p.add_argument("--model", default="runs/synthetic/best.pt")
    p.add_argument("--dt", type=float, help="sample interval in seconds")
    p.add_argument("--dx", type=float, default=10.0, help="trace spacing in metres (plots only)")
    p.add_argument("--out", default="separated")
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    m, dt, dx = read_input(a.input, a.dt, a.dx)
    if not dt:
        raise SystemExit("--dt is required for this input")
    model, band, _ = load_model(a.model, a.device)
    est = separate(model, m, dt, band, stride=a.stride, device=a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input).stem
    np.savez(out / f"{stem}_separated.npz", reflections=est["r"], ground_roll=est["g"], dt=dt)
    panel(out / f"{stem}_separated.png", [("input", m), ("reflections", est["r"]), ("ground roll", est["g"])], dt, dx or 10.0)
    print(f"band {band.f_pass}/{band.f_stop} Hz -> {out / (stem + '_separated.npz')}")


if __name__ == "__main__":
    main()

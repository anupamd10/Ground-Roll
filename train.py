import argparse
import functools
import glob
import json
from pathlib import Path

import numpy as np
import torch

from grsep import Band, GatherParams, GroundRollSeparator, leakage_ratio, make_gather, suggest_band
from grsep.data import PatchDataset
from grsep.plotting import loss_curve
from grsep.train import fit


print = functools.partial(print, flush=True)

def load_npz(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        s = {"m": d["m"].astype(np.float32), "dt": float(d["dt"])}
        if "r" in d:
            s["r"] = d["r"].astype(np.float32)
        elif "g" in d:
            s["g"] = d["g"].astype(np.float32)
        else:
            raise ValueError(f"{f}: needs key 'r' (reflections label) or 'g' (ground-roll label)")
        out.append(s)
    if not out:
        raise FileNotFoundError(pattern)
    return out


def synth_set(n, seed, params):
    rng = np.random.default_rng(seed)
    return [make_gather(params, rng) for _ in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/synthetic")
    p.add_argument("--data", help="glob of training .npz files with keys m, r (or g), dt")
    p.add_argument("--val-data", help="glob of validation .npz files (separate gathers)")
    p.add_argument("--n-train", type=int, default=10)
    p.add_argument("--n-val", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nt", type=int, default=1000)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--f-pass", type=float, default=25.0)
    p.add_argument("--f-stop", type=float, default=35.0)
    p.add_argument("--auto-band", action="store_true")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--optim", choices=["rmsprop", "adam"], default="rmsprop")
    p.add_argument("--steps-per-epoch", type=int)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--synth", default="{}", help='JSON overrides for synthetic gathers, e.g. \'{"dx": 12.5, "fe": [12, 35]}\'; lists become ranges')
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.data:
        train = load_npz(a.data)
        if not a.val_data:
            raise SystemExit("--val-data is required with --data: validate on whole held-out gathers, not on a random patch split")
        val = load_npz(a.val_data)
    else:
        over = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.synth).items()}
        params = GatherParams().with_(nt=a.nt, nx=a.nx, **over)
        train = synth_set(a.n_train, a.seed, params)
        val = synth_set(a.n_val, a.seed + 10_000, params)

    if a.auto_band:
        ref = np.concatenate([s["g"] if "g" in s else s["m"] - s["r"] for s in train], axis=1)
        band, f99 = suggest_band(ref, train[0]["dt"])
        print(f"ground-roll 99.5% energy below {f99:.1f} Hz -> band pass {band.f_pass:.1f} / stop {band.f_stop:.1f} Hz")
    else:
        band = Band(a.f_pass, a.f_stop)
    leak = np.mean([leakage_ratio(s["g"] if "g" in s else s["m"] - s["r"], s["dt"], band) for s in train])
    print(f"ground-roll energy leaking into the untouched high band: {100 * leak:.4f}%")
    if leak > 0.01:
        print("WARNING: >1% leakage. Raise --f-pass/--f-stop or use --auto-band.")

    tr = PatchDataset(train, band, a.patch, a.stride, augment=True)
    va = PatchDataset(val, band, a.patch, a.stride, augment=False)
    print(f"train patches {len(tr)} from {len(train)} gathers, val patches {len(va)} from {len(val)} gathers")

    model = GroundRollSeparator(width=a.width, dropout=a.dropout)
    print(f"params {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M, receptive field {model.receptive_field}px, device {a.device}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    hist = fit(model, tr, va, band, out, a.epochs, a.batch, a.lr, a.lam, a.optim, a.steps_per_epoch, a.patience, a.device, a.workers)
    loss_curve(out / "loss.png", hist)
    print(f"saved {out / 'best.pt'}")


if __name__ == "__main__":
    main()

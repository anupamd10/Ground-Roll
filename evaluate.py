import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch

from grsep import GatherParams, leakage_ratio, load_model, make_gather, separate
from grsep.baselines import fk_dip_filter, highpass_filter, tune_fk
from grsep.metrics import snr_db
from grsep.plotting import panel

TESTS = [
    ("Test 0  baseline", {}),
    ("Test 1  GR 10-20 Hz", {"fb": 10.0, "fe": 20.0}),
    ("Test 2  strong GR -20 dB", {"snr_db": -20.0}),
    ("Test 3  dt 4 ms", {"dt": 0.004, "nt": 500}),
    ("Test 4  dx 20 m", {"dx": 20.0}),
    ("Test 6  dx 40 m", {"dx": 40.0}),
    ("Test 7  noise 3.5%", {"noise_ratio": 0.0353}),
    ("Test 10 dx 40 m + noise 19.7%", {"dx": 40.0, "noise_ratio": 0.1972}),
]


print = functools.partial(print, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/synthetic/best.pt")
    p.add_argument("--out", default="runs/synthetic/eval")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=50_000)
    p.add_argument("--nt", type=int, default=1000)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    model, band, _ = load_model(a.model, a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    hdr = f"{'test':32s} {'initial':>8s} {'f-k dip':>8s} {'hi-pass':>8s} {'CNN':>8s} {'GR leak %':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for t, (name, kw) in enumerate(TESTS):
        kw = {"nt": a.nt, "nx": a.nx, **kw}
        if "dt" in kw and "nt" not in TESTS[t][1]:
            kw["nt"] = int(round(a.nt * 0.002 / kw["dt"]))
        params = GatherParams().with_(**kw)
        rng = np.random.default_rng(a.seed + 97 * t)
        res = {"initial": [], "fk": [], "hp": [], "cnn": [], "leak": []}
        for i in range(a.n):
            s = make_gather(params, rng)
            m, dt, dx = s["m"], s["dt"], s["dx"]
            r = m - s["g"]
            est = separate(model, m, dt, band, stride=a.stride, device=a.device)
            _, v, fmax = tune_fk(m, r, dt, dx)
            fk = fk_dip_filter(m, dt, dx, v, fmax)
            hp = highpass_filter(m, dt, band)
            res["initial"].append(snr_db(r, m))
            res["fk"].append(snr_db(r, fk))
            res["hp"].append(snr_db(r, hp))
            res["cnn"].append(snr_db(r, est["r"]))
            res["leak"].append(100 * leakage_ratio(s["g"], dt, band))
            if i == 0:
                panel(
                    out / f"test{t}.png",
                    [("mixture", m), ("true reflections", r), ("CNN reflections", est["r"]), ("CNN ground roll", est["g"]), (f"f-k dip (v={v})", fk), ("high-pass", hp)],
                    dt,
                    dx,
                    suptitle=name,
                )
        row = {"test": name, **{k: float(np.mean(v)) for k, v in res.items()}}
        rows.append(row)
        print(f"{name:32s} {row['initial']:8.2f} {row['fk']:8.2f} {row['hp']:8.2f} {row['cnn']:8.2f} {row['leak']:9.4f}")
    (out / "table.json").write_text(json.dumps(rows, indent=1))
    print(f"\nSNR in dB = 10 log10(|m - g|^2 / |m - g - r_hat|^2), reference = everything except ground roll, averaged over {a.n} unseen gathers per test. f-k dip is oracle-tuned per gather.")


if __name__ == "__main__":
    main()

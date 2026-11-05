import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.fft import fft2, ifft2

from grsep import Band, load_model, separate, split
from grsep.baselines import highpass_filter
from grsep.plotting import panel

print = functools.partial(print, flush=True)


def cone_masks(shape, dt, dx, src, v_in, v_out):
    nt, nx = shape
    t = (np.arange(nt) * dt)[:, None]
    off = np.abs(np.arange(nx) - src)[None, :] * dx
    return off <= v_in * t, off >= v_out * t


def fk_mask_filter(m, mask):
    spec = np.fft.fftshift(fft2(m))
    return np.real(ifft2(np.fft.ifftshift(spec * (1 - mask)))).astype(np.float32)


def energy(a, sel=None):
    a = a.astype(np.float64)
    return float(np.sum((a if sel is None else a[sel]) ** 2))


def score(m, r_hat, dt, inside, outside, fk_zone, f_gr):
    m_lo, _ = split(m, dt, f_gr)
    r_lo, _ = split(r_hat, dt, f_gr)
    out = {
        "gr_removed_db": 10 * np.log10(energy(m_lo, inside) / max(energy(r_lo, inside), 1e-30)),
        "signal_fidelity_db": 10 * np.log10(energy(m, outside) / max(energy(m - r_hat, outside), 1e-30)),
    }
    if fk_zone is not None:
        spec_m = np.abs(np.fft.fftshift(fft2(m))) ** 2
        spec_r = np.abs(np.fft.fftshift(fft2(r_hat))) ** 2
        out["fk_zone_removed_db"] = 10 * np.log10(spec_m[fk_zone].sum() / max(spec_r[fk_zone].sum(), 1e-30))
    return {k: float(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gathers", nargs="+", help=".npz with m, dt, dx and optional fk_mask")
    p.add_argument("--model", action="append", required=True, help="name=path/to/best.pt, repeatable")
    p.add_argument("--geometry", required=True, help='JSON {"gather_stem": [source_trace, v_inside, v_outside]}')
    p.add_argument("--gr-band", type=float, default=20.0, help="ground-roll energy is measured below this frequency")
    p.add_argument("--out", default="runs/real_eval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    models = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        models[name] = load_model(path, a.device)
    geom = json.loads(a.geometry)
    f_gr = Band(a.gr_band, a.gr_band + 5)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in a.gathers:
        stem = Path(path).stem
        d = np.load(path)
        m, dt, dx = d["m"].astype(np.float32), float(d["dt"]), float(d["dx"])
        fk_zone = d["fk_mask"].astype(bool) if "fk_mask" in d else None
        src, v_in, v_out = geom[stem]
        inside, outside = cone_masks(m.shape, dt, dx, src, v_in, v_out)
        outputs = {}
        for name, (model, band, _) in models.items():
            outputs[f"CNN {name}"] = separate(model, m, dt, band, device=a.device)["r"]
            outputs[f"high-pass {band.f_pass:g} Hz"] = highpass_filter(m, dt, band)
        if fk_zone is not None:
            outputs["f-k (paper mask)"] = fk_mask_filter(m, fk_zone)
        rows = {k: score(m, v, dt, inside, outside, fk_zone, f_gr) for k, v in outputs.items()}
        results[stem] = rows
        print(f"\n{stem}  {m.shape}  dt {dt * 1000:g} ms  dx {dx:g} m")
        hdr = f"{'method':24s} {'GR removed':>11s} {'fidelity':>9s} {'fk-zone':>8s}"
        print(hdr)
        print("-" * len(hdr))
        for k, s in rows.items():
            print(f"{k:24s} {s['gr_removed_db']:9.1f}dB {s['signal_fidelity_db']:7.1f}dB {s.get('fk_zone_removed_db', float('nan')):6.1f}dB")
        panels = [("input", m)] + [(k, v) for k, v in outputs.items()]
        panel(out / f"{stem}_reflections.png", panels, dt, dx, fmax=80, suptitle=f"{stem}: estimated reflections")
        panel(out / f"{stem}_removed.png", [("input", m)] + [(k, m - v) for k, v in outputs.items()], dt, dx, fmax=80, suptitle=f"{stem}: removed (ground roll estimate)")
        np.savez(out / f"{stem}_outputs.npz", **{k.replace(" ", "_"): v for k, v in outputs.items()})
    (out / "scores.json").write_text(json.dumps(results, indent=1))
    print(
        "\nGR removed: low-frequency energy removed inside the ground-roll cone (higher = more removed)."
        "\nfidelity: 10log10(|m|^2/|m - r_hat|^2) outside the cone, where there is little ground roll (higher = less signal damage)."
        "\nfk-zone: energy removed inside the paper's own f-k reject zone (favours the f-k filter by construction)."
    )


if __name__ == "__main__":
    main()

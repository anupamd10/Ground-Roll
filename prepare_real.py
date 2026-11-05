import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main():
    p = argparse.ArgumentParser(description="Convert garch-groundroll .mat gathers to .npz (m, dt, dx, fk_mask)")
    p.add_argument("mat", nargs="+")
    p.add_argument("--out", default="real")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.mat:
        d = loadmat(f)
        rec = {"m": d["data"].astype(np.float32), "dt": float(np.squeeze(d["dt"])), "dx": float(np.squeeze(d["dx"]))}
        if "fk_mask" in d:
            rec["fk_mask"] = d["fk_mask"].astype(np.uint8)
        dst = out / (Path(f).stem + ".npz")
        np.savez(dst, **rec)
        print(f"{dst}  {rec['m'].shape}  dt {rec['dt']}  dx {rec['dx']}")


if __name__ == "__main__":
    main()

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from .loss import SeparationLoss
from .model import save_model


def _run(model, loader, crit, opt=None, device="cpu"):
    train = opt is not None
    model.train(train)
    tot = {"total": 0.0, "sim": 0.0, "cons": 0.0, "disc": 0.0}
    n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            r_hat, g_hat = model(x)
            loss, parts = crit(r_hat, g_hat, y[:, :1], y[:, 1:], x[:, :1])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            b = x.shape[0]
            n += b
            tot["total"] += loss.item() * b
            for k, v in parts.items():
                tot[k] += v.item() * b
    return {k: v / max(n, 1) for k, v in tot.items()}


def fit(
    model,
    train_ds,
    val_ds,
    band,
    out_dir,
    epochs=100,
    batch_size=32,
    lr=1e-4,
    lam=0.05,
    optimizer="rmsprop",
    steps_per_epoch=None,
    patience=15,
    device="cpu",
    workers=0,
    log=None,
):
    log = log or (lambda s: print(s, flush=True))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    crit = SeparationLoss(lam=lam)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, patience // 3))
    sampler = None
    if steps_per_epoch:
        sampler = RandomSampler(train_ds, replacement=True, num_samples=steps_per_epoch * batch_size)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=workers, drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=workers)
    best, bad, history = float("inf"), 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = _run(model, tl, crit, opt, device)
        va = _run(model, vl, crit, None, device)
        sched.step(va["sim"])
        rec = {"epoch": ep, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(rec)
        improved = va["sim"] < best
        if improved:
            best, bad = va["sim"], 0
            save_model(model, out_dir / "best.pt", band, {"epoch": ep, "val_sim": best, "lam": lam})
        else:
            bad += 1
        log(f"ep {ep:3d}  train sim {tr['sim']:.4f}  val sim {va['sim']:.4f}  disc {va['disc']:.3f}  lr {rec['lr']:.1e}  {rec['sec']:.0f}s{'  *' if improved else ''}")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        if bad >= patience:
            log(f"early stop at epoch {ep}")
            break
    save_model(model, out_dir / "last.pt", band, {"epoch": ep, "lam": lam})
    return history

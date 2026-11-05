# Ground-roll separation with a CNN (corrected pipeline)


```
m ──zero-phase low-pass──► m_low ─┐
  └──────── m − m_low ───► m_high ┼─► CNN (shared 7-layer trunk, two 3-layer heads) ─► ĝ, r̂_low
                                  │
                       r̂ = r̂_low + m_high   (the high band never passes through the network)
```

## What was fixed relative to the paper

| # | Paper | Problem | Fix (file) |
|---|-------|---------|------------|
| 1 | Onset `(x − xc)/v` | negative for x < xc, so no V-shaped cone | `|x − xc|/v` (`synth.py`) |
| 2 | Decay `s^(x − xc)` | grows on one side of the source | `s^(|x − xc|/dx)`, 0 < s < 1 |
| 3 | `sin(2π f(t) t)` with absolute t | sweep phase wrong away from the source | τ = t − onset in both phase and window |
| 4 | Loss term `−λ‖r̂ − ĝ‖₁` | unbounded below; not the Huang et al. form the text describes | `−λ(‖r̂ − g‖² + ‖ĝ − r‖²)` with λ ∈ [0, 1), which keeps the loss provably bounded (`loss.py`) |
| 5 | Unspecified low-pass; "high-pass" in one place | phase shift or a mismatched filter means m_low + m_high ≠ m | FFT raised-cosine zero-phase low-pass, `m_high = m − m_low` exactly (`bandsplit.py`) |
| 6 | Ground-roll label `g` | g has a little energy above the cutoff, so `ĝ + r̂_low = m_low` cannot hold exactly | labels are `r_low` and `g_low = m_low − r_low`; leakage is measured and reported |
| 7 | Random 10% *patch* validation split | patches at stride 10 overlap ~85%, so validation leaks into training | validation on whole held-out gathers |
| 8 | Validation merged back into training | nothing left to judge convergence | kept separate; best checkpoint by validation similarity loss, early stopping |
| 9 | Normalization and patch stitching not described | amplitude mismatch between bands; seams. Normalizing by the RMS of `m` also blows up quiet zones (gaps, mutes), where low-pass ringing gives `m_low ≈ −m_high` while `m ≈ 0`; inputs reach 600×, which made the first training run unstable | per-patch scaling by the RMS of the two input bands, undone before blending; sine-tapered overlap-add (`data.py`, `infer.py`) |
| 10 | 13440 / 13400 patches | neither matches 1000×200, 64², stride 10 | edge-flush patch grid: 95 × 15 = 1425 per gather (`data.py`) |
| 11 | SNR undefined | — | `10·log10(‖m−g‖² / ‖m−g−r̂‖²)`: the reference is everything except ground roll, so random noise is not counted as a separation error (`evaluate.py`) |
| 12 | 50 Hz Ricker only | reflections have little energy below the cutoff, so the task is too easy | Ricker peak drawn from 25–50 Hz |

## Verified result (small CPU model)

Width 32, 8 training gathers, 30 short epochs (about 20 min on 2 CPU cores). Unseen gathers, 3 per test, SNR in dB:

| test | initial | f-k dip (oracle-tuned) | high-pass | CNN |
|---|---|---|---|---|
| 0 baseline | −12.6 | 11.7 | 8.2 | **11.9** |
| 1 GR 10–20 Hz | −11.5 | 10.4 | 6.3 | **12.0** |
| 2 strong GR (−20 dB) | −20.0 | 3.6 | **7.1** | 5.1 |
| 3 dt 4 ms | −10.8 | **13.4** | 10.9 | 13.1 |
| 4 dx 20 m | −10.6 | 2.2 | 6.7 | **11.3** |
| 6 dx 40 m | −13.4 | −10.0 | 6.9 | **7.7** |
| 7 noise 3.5% | −9.5 | 12.5 | 9.2 | **13.0** |
| 10 dx 40 m + noise 19.7% | −4.6 | −1.0 | 7.9 | **10.6** |

The pattern matches the paper's Table 1: the CNN holds up where the f-k dip filter collapses (coarse trace spacing, spatial aliasing). This tiny model has not reached the paper's absolute numbers, and it loses on −20 dB ground roll, which lies outside its −14 to −8 dB training range. Train the full width-64 model on a GPU and widen `GroundRollParams.snr_db` to cover that case.

## Repository layout

```
grsep/          library: synth, bandsplit, model, loss, data, infer, train, baselines, metrics, plotting
train.py        train on synthetic or your own .npz gathers
evaluate.py     synthetic generalization tests (Table 1 style)
prepare_real.py convert the field .mat gathers to .npz
evaluate_real.py score CNN / high-pass / f-k on real gathers
separate.py     apply a trained model to .npy / .npz / .sgy
tests/          unit tests, one per fix
results/        trained models, figures, logs and score tables (see results/README.md)
```

## Setup

```bash
pip install torch numpy scipy matplotlib pytest   # segyio too, for SEG-Y input
python -m pytest -q tests                        # 11 checks, one per fix
```

## 1. Train on synthetic data (paper settings)

```bash
python train.py --out runs/synthetic             # 10 train + 2 val gathers, RMSprop 1e-4, width 64
```

This needs a GPU. On a CPU, a quick smoke run looks like this:

```bash
python train.py --out runs/cpu --n-train 8 --n-val 1 --width 32 --batch 16 \
    --steps-per-epoch 80 --epochs 30 --optim adam --lr 5e-4
```

Outputs: `best.pt` (weights plus the frequency band), `history.json`, `loss.png`.

## 2. Generalization table (paper Table 1)

```bash
python evaluate.py --model runs/synthetic/best.pt --out runs/synthetic/eval
```

Each test is run on unseen gathers and compared against an oracle-tuned f-k dip filter and a plain high-pass. One t-x / f-k figure is written per test.

## 3. Train on real data

Save each shot as `.npz` with `m` (nt × nx), `dt` in seconds, and `r` (manually processed reflections) or `g` (ground roll). Keep whole shots apart for validation:

```bash
python train.py --data "real/train/*.npz" --val-data "real/val/*.npz" --auto-band --out runs/real
```

`--auto-band` puts the cutoff 10 Hz above the frequency holding 99.5% of the label ground-roll energy. Check the printed leakage: anything above 1% goes straight into your output, because the high band bypasses the network.

## 4. Test on real field data

The public repo of Aminian & Riahi (2023), `github.com/MohammadAmin-Aminian/garch-groundroll` (MIT), ships two AGC'd land shot gathers (2 ms, 12.5 m) together with the f-k reject masks used in their paper.

```bash
git clone --depth 1 https://github.com/MohammadAmin-Aminian/garch-groundroll
python prepare_real.py garch-groundroll/data/field_shot.mat garch-groundroll/data/shot_gather_502.mat --out real

python train.py --out runs/matched --n-train 16 --n-val 2 --f-pass 45 --f-stop 55 \
  --synth '{"dx": 12.5, "fb": [2, 8], "fe": [10, 35], "v": [120, 800], "snr_db": [-25, -5], "noise_ratio": [0, 0.05], "f_peak": [15, 45], "t0_len": [0.1, 0.3], "t_slope": [0.0003, 0.0015], "decay": [0.99, 0.999], "n_modes": [1, 3]}'

python evaluate_real.py real/field_shot.npz real/shot_gather_502.npz \
  --model "matched=runs/matched/best.pt" \
  --geometry '{"field_shot": [22, 300, 450], "shot_gather_502": [250, 220, 400]}'
```

Real data has no clean answer, so three label-free measures are reported:
- **GR removed:** low-frequency energy removed inside the ground-roll cone.
- **fidelity:** how little the data outside the cone is changed.
- **fk-zone:** energy removed inside the paper's own reject zone. This one favours the f-k filter by construction.

Results with the small CPU models (width 32, synthetic training only):

| gather | method | GR removed | fidelity | fk-zone |
|---|---|---|---|---|
| field_shot | CNN, survey-matched synthetic | 8.3 dB | 5.1 dB | 13.5 dB |
| field_shot | f-k, paper mask | 4.3 dB | 12.7 dB | 23.4 dB |
| field_shot | high-pass 45 Hz | 47.1 dB | 0.5 dB | 42.9 dB |
| 502 | CNN, survey-matched synthetic | 11.9 dB | 3.2 dB | 10.0 dB |
| 502 | f-k, paper mask | 15.7 dB | 3.1 dB | 27.7 dB |

What the figures show:
- **field_shot:** the CNN removes the aliased ground-roll branch that the f-k mask leaves behind, and keeps the reflections. It also removes the first-arrival refraction, which the synthetic training never contained, and that is what lowers its fidelity score.
- **shot_gather_502:** the expert-picked f-k mask removes more of the dense, very slow central cone than the CNN does.

A plain high-pass destroys the AGC'd reflections (fidelity ≈ 0 dB).

The paper trained on *real* labels for its field examples, and that is the step that closes this gap. Label a few shots with your usual processing flow, save them as `m` + `r`, and train or fine-tune with `--data`.

## 5. Apply to a gather

```bash
python separate.py shot.sgy --model runs/real/best.pt --dx 10
python separate.py shot.npy --dt 0.002 --model runs/real/best.pt
```

This writes `separated/<name>_separated.npz` (reflections, ground_roll) and a figure.

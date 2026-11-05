# Results

The outputs here were all produced by the code in this repository, using small CPU models (width 32).

## Contents

| path | what |
|---|---|
| `paper_synthetic/` | model trained with the paper's synthetic setup (dx 10 m, ground roll 5–15 Hz, 25/35 Hz split) |
| `survey_matched/` | model trained on synthetic data matched to the field survey (dx 12.5 m, 120–800 m/s, up to 35 Hz, 45/55 Hz split) |
| `*/best.pt` | weights plus frequency band; load with `grsep.load_model` |
| `*/history.json`, `*/loss.png`, `logs/` | training curves and logs |
| `paper_synthetic/eval/` | generalization tests on unseen synthetic gathers (paper Table 1 style) |
| `real_eval/` | tests on two real land shot gathers |
| `synthetic_gather_example.png` | one synthetic gather with its band split, in t-x and f-k |

## Synthetic generalization (`paper_synthetic/eval`)

SNR in dB = 10·log10(‖m−g‖² / ‖m−g−r̂‖²), averaged over 3 unseen gathers per test. The f-k dip filter was given its best possible settings for each gather.

| figure | test | initial | f-k dip | high-pass | CNN |
|---|---|---|---|---|---|
| `test0.png` | Test 0  baseline | -12.59 | 11.72 | 8.23 | **11.91** |
| `test1.png` | Test 1  GR 10-20 Hz | -11.48 | 10.41 | 6.33 | **11.96** |
| `test2.png` | Test 2  strong GR -20 dB | -20.00 | 3.61 | 7.06 | **5.11** |
| `test3.png` | Test 3  dt 4 ms | -10.77 | 13.39 | 10.87 | **13.06** |
| `test4.png` | Test 4  dx 20 m | -10.56 | 2.17 | 6.69 | **11.27** |
| `test5.png` | Test 6  dx 40 m | -13.35 | -9.97 | 6.87 | **7.70** |
| `test6.png` | Test 7  noise 3.5% | -9.47 | 12.45 | 9.23 | **12.97** |
| `test7.png` | Test 10 dx 40 m + noise 19.7% | -4.55 | -1.04 | 7.85 | **10.55** |

## Real field data (`real_eval`)

The gathers come from the MIT-licensed repository of Aminian & Riahi (2023), https://github.com/MohammadAmin-Aminian/garch-groundroll. They are not redistributed here; `prepare_real.py` converts them.

| gather | method | GR removed (dB) | fidelity (dB) | fk-zone removed (dB) |
|---|---|---|---|---|
| field_shot | CNN paper-synthetic | 7.7 | 5.4 | 14.9 |
| field_shot | high-pass 25 Hz | 41.1 | 1.4 | 34.4 |
| field_shot | CNN survey-matched | 8.3 | 5.1 | 13.5 |
| field_shot | high-pass 45 Hz | 47.1 | 0.5 | 42.9 |
| field_shot | f-k (paper mask) | 4.3 | 12.7 | 23.4 |
| shot_gather_502 | CNN paper-synthetic | 9.1 | 4.2 | 8.9 |
| shot_gather_502 | high-pass 25 Hz | 43.3 | 0.4 | 17.1 |
| shot_gather_502 | CNN survey-matched | 11.9 | 3.2 | 10.0 |
| shot_gather_502 | high-pass 45 Hz | 48.4 | 0.3 | 20.0 |
| shot_gather_502 | f-k (paper mask) | 15.7 | 3.1 | 27.7 |

- **GR removed:** low-frequency energy removed inside the ground-roll cone.
- **fidelity:** how little the data outside the cone is changed.
- **fk-zone:** energy removed inside the paper's own reject zone. This one favours the f-k filter by construction.

To reproduce, see the "Test on real field data" section of the top-level README.

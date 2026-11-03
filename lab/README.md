# The practice lab

We can't tune on the real match (the full ground truth is private, and the public 5 min are
for sanity checks only). So we tune here instead: on **Metrica Sample Game 3**, a free public
match with clean tracking and human event data, made messy on purpose to look like our ATD.

Rules-safe: nothing here touches the hackathon data. Hand labels on *other* games are allowed.

## Setup (once)
```bash
git clone https://github.com/metrica-sports/sample-data lab/sample-data
pip install kloppy
python lab/prepare_tracking.py     # step 6: tracking -> lab/cache/g3.npz      (~40 s)
python lab/make_gt.py              # step 7: events   -> lab/cache/g3_gt.csv
python lab/run_baseline.py         # step 9: is the lab realistic?
```

## Files
| file | what |
|---|---|
| `prepare_tracking.py` | loads Game 3 tracking with kloppy |
| `make_gt.py` | human events -> frame-by-frame H/A/D answers |
| `lab.py` | noise model (`corrupt`), nearest-player helper, official scorer wrapper |
| `run_baseline.py` | baseline on clean vs noisy Game 3, next to the published real numbers |

## Calibration (noisy Game 3 vs the real match, official baseline)
| window | lab spellF1 | real spellF1 |
|---|---|---|
| raw | 0.039 | 0.049 |
| 25 | 0.163 | 0.200 |
| 50 | 0.224 | 0.243 |
| 125 | 0.235 | 0.224 |

Ball missing on 41% of graded frames in both. Close enough to trust the *direction* of an
improvement. Known gaps: macroF1 is lower in the lab (~0.52 vs ~0.67), and the event-derived
truth has more short possessions (404, ~10 s mean).

## How to use it
1. Build an idea as a function: `(ball, H, A) -> labels`
2. Score it on `lab.corrupt(...)` with several `seed`s
3. Keep it only if it beats the baseline by more than ~0.04 spellF1 across seeds

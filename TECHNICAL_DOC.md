# Technical document — Phase 1, possession labelling

**Team:** IE Sports Analytics Club · **Members:** Tina Haddad, Karim Al Junadi, Jose Miguel Reyes, Ahmed Al Baba · **Repository:** (https://github.com/tinahaddad/mtah-2027--IE-Sports-Analytics-Club-)

## 1. Summary

We label each video frame of the provided match as `H`, `A`, `D` or `X` with a three-state hidden
Markov model (HMM) decoded by Viterbi. The observation at each frame is one coarse symbol derived
from the ATD tracking: which team's nearest player is closest to the ball, and how close. The
emission and transition tables are counted on a different, public match (Metrica Sample Game 3),
not on the provided game. Smart Tagging set pieces add a prior that makes dead ball more likely
just before a restart. A heavy penalty on switching state keeps the output in long possessions,
which is what the `spellF1` metric rewards.

There is no neural network, no hand labelling and no use of the private ground truth.

## 2. Why this approach

The official baseline takes a majority vote of the nearest-player team over a window. That flips
whenever the ball is lost or passes close to an opponent, and on this match the ball is missing on
about 41% of graded frames. `spellF1` matches whole possessions by temporal overlap, so many short
spurious possessions cost a lot.

An HMM fits that structure directly. A possession is a state that persists, so changing state
should be rare unless the evidence is strong. The switch penalty expresses that in one number, and
Viterbi chooses the single most likely path instead of deciding frame by frame. It needs no labels
on the provided game, which the rules forbid, and it is cheap to run and easy to audit.

## 3. Assets used, and how

| asset | used? | how |
|---|---|---|
| ATD tracking (`atd_raw`, `atd_meta`) | yes | per-frame ball position and all tracked players; track-to-team mapping from the metadata, using the baseline's `team_of_each_track` |
| Smart Tagging XML | yes | only the `SET PIECES` tags, as restart times for the dead-ball prior. Team tags are not used: all throw-ins carry `Team = N/A` |
| Homography CSV | no | not read by the pipeline |
| Video (tactical or broadcast) | no | not opened |
| Public ground truth (5 min) | sanity checks only | confirmed the file format and rough direction; not used for tuning |
| Metrica Sample Game 3 | yes | fitting the HMM tables and tuning in the lab (section 5) |

Tried and dropped:
- **Baum–Welch re-fitting on the real match.** It was unsupervised, so it needed no labels, but in
  the lab it over-segmented: more than 1,000 possessions against 404 in the truth. We dropped it.
  The code (`baum_welch` in `pipeline/hmm2.py`) is still there and is not called by the pipeline.
- **Team-specific emissions.** Replaced by team-symmetric ones in v3 (section 4).

## 4. The model

**States:** `H`, `A`, `D`.

**Observation symbol per frame** (`pipeline/hmm.py`), from the ball and both teams' positions:

| symbol | meaning |
|---|---|
| `NONE` | ball not tracked |
| `HC` / `AC` | nearest player is home / away and within 2 m of the ball |
| `HF` / `AF` | nearest player is home / away but farther than 2 m |
| `OUT` | ball tracked outside the pitch, almost surely dead |

**Emissions and transitions.** P(symbol | state) and P(next state | state) are counted on Sample
Game 3 (`pipeline/hmm_params_v3.json`). In v3 the emission table is made team-symmetric: a lost
ball should not favour either team, which removed a bias towards the home side. In the lab that
reduced the home-share error from 0.12 to 0.01.

**Switch penalties.** The `H`↔`A` and live↔`D` transition probabilities are divided by 10,000
before decoding (`team_penalty` and `dead_penalty`), then rows are renormalised. The values were
chosen in the lab.

**Set-piece prior** (`tag_prior` in `pipeline/hmm2.py`). For every Smart Tagging `SET PIECES` tag
(throw-in, corner, goal kick, free kick, kick-off), the 12 s before the tag start to 4 s after it
get +3.0 log-score towards `D`. Frames more than 20 s from every tag get −0.75 towards `D`, because
a stoppage there would usually have been tagged. Smart Tagging is automatic and not derived from
the ground truth, and the organisers' FAQ says it may be used.

**Decoding.** Viterbi on the sum of the emission log-likelihood and the prior, run separately on
each half, using the half boundaries in `games.py`. Frames outside the two halves are written as
`X`.

## 5. Training and validation data

| dataset | source | who labelled | how much |
|---|---|---|---|
| Metrica Sample Game 3 tracking and events | public, `metrica-sports/sample-data` | Metrica (human-annotated events) | one full match |

We turned the event data into frame-by-frame `H`/`A`/`D` answers with `lab/make_gt.py`. That gives
404 possessions with a mean length of about 10 s. We did no labelling ourselves. **Nothing is
labelled on the provided game**, by hand or otherwise.

**Practice lab.** Because the real match cannot be tuned against, `lab/` degrades Game 3 to look
like the ATD: ball hidden about 41% of the time, position jitter and dropouts. Calibration against
the official baseline:

| window | lab spellF1 | real spellF1 |
|---|---|---|
| raw | 0.039 | 0.049 |
| 25 | 0.163 | 0.200 |
| 50 | 0.224 | 0.243 |
| 125 | 0.235 | 0.224 |

We only kept an idea if it beat the baseline by more than the ~0.04 noise level across three
unseen noise seeds.

## 6. Results

Lab (degraded Game 3, mean of 3 unseen seeds), spellF1:

| model | normal noise | harder noise |
|---|---|---|
| official baseline | 0.212 | n/a |
| HMM v1 | 0.368 | 0.308 |
| v2 (+ set-piece prior) | 0.368 | 0.323 |
| v3 (+ symmetric emissions) | +0.002 | +0.012 over v2 |

Public 5 minutes of the real match (sanity check only, 17 possessions, so very noisy): spellF1
0.4667 against 0.3256 for the baseline. 

## 7. The possession rules we implemented

We did not write explicit rules; they come out of the model.

- **Live possession** is the team whose nearest player is closest to the ball, with 2 m as the
  "close" threshold, but only counted as evidence. The state changes only when it keeps looking
  like the other team for long enough to beat the switch penalty.
- **Loose balls and challenges.** There is no separate state. A loose ball or a contested duel
  usually keeps the previous possession, because switching is heavily penalised. When the ball is
  not tracked, the symbol says nothing about either team, so the previous state stays.
- **Restarts and dead ball.** Ball outside the pitch pushes strongly to `D`. Smart Tagging set
  pieces make `D` likelier in the seconds before the restart.
- **Abstaining.** `X` is only written outside the two halves. Under `spellF1`, an `X` is
  forward-filled with the last label in that half, so we avoid it elsewhere.

## 8. Limitations and what we would do next

- **Tracking gaps.** With the ball missing on about 41% of graded frames, the model leans on the
  stay-put prior. Long gaps around quick turnovers are likely to be labelled as the old possession.
- **Penalty values.** The 10,000 switch penalty was chosen on a degraded Game 3, not the real
  match, so it may be too strong or too weak here. We did not tune it on the real match.
- **Dead-ball share.** The lab truth had about 27% dead ball, while our v2 output had 24%. The
  real share is unknown.
- **Crude symbol.** The observation is only nearest-player team and a 2 m cut-off. We do not use
  player speed, ball height or pass direction.
- **Single training match.** The tables come from one match, so they carry that match's style.
- **Next steps.** Interpolate the missing ball from nearby player motion, add player and ball
  speed to the symbol, learn the penalties on several sample games, and use the broadcast video
  with a local detector to recover the ball on missing frames.

## 9. Reproduction

See [README.md](README.md). In short, with the data bundle unpacked into `./data`:

```bash
pip install -r requirements.txt
python make_submission.py            # writes and validates evaluation/results/hmm_v3.csv
python make_submission.py --check-repro   # builds twice, requires identical bytes
```

The model is deterministic, so the same data and the same dependencies give the same file.

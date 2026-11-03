"""The practice lab: corrupt clean Sample Game 3 tracking so it looks like the hackathon ATD,
then run any possession system on it and score it with the OFFICIAL metric.

Noise model (all knobs in NOISE):
  * dead-ball ball positions: where the clean tracking loses the ball during a stoppage, the
    automatic tracker usually still sees it sitting at the restart -> interpolate it in, then
    keep only `dead_visible` of it.
  * live-play ball dropouts: bursts (mean `burst_s` seconds) covering `live_miss` of live frames,
    more likely when players crowd the ball (occlusions).
  * players: jitter `jitter_m` metres, randomly drop `player_drop` of player-frames in bursts.
"""
import sys
import numpy as np
import pandas as pd

import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # lab/ lives inside the challenge repo
sys.path.insert(0, os.path.join(REPO, "evaluation"))
sys.path.insert(0, os.path.join(REPO, "starter"))
import possession_metrics as pm  # noqa: E402

PITCH = (105.0, 68.0)
FPS = 25
# Calibrated so the official baseline's spellF1 curve on the lab matches the published
# full-match curve (raw/25/50/125 = 0.049/0.200/0.243/0.224 -> lab 0.039/0.163/0.224/0.235)
# with the ball missing on 41% of graded frames, as in the real ATD.
NOISE = dict(dead_visible=0.4, live_miss=0.40, burst_s=2.5, crowd_boost=2.0,
             player_drop=0.07, jitter_m=0.1, ball_jitter_m=0.1, seed=0)


def load_clean():
    d = np.load(os.path.join(HERE, "cache", "g3.npz"))
    gt = pd.read_csv(os.path.join(HERE, "cache", "g3_gt.csv")).label.to_numpy()
    return d["ball"].copy(), d["H"].copy(), d["A"].copy(), gt


def bursts(n, frac, mean_len, rng, weight=None):
    """Boolean mask covering ~frac of n frames in bursts of ~mean_len, starts biased by weight."""
    mask = np.zeros(n, bool)
    if frac <= 0:
        return mask
    n_bursts = int(n * frac / mean_len)
    p = None if weight is None else weight / weight.sum()
    starts = rng.choice(n, n_bursts, replace=True, p=p)
    lens = rng.exponential(mean_len, n_bursts).astype(int) + 1
    for s, l in zip(starts, lens):
        mask[s:s + l] = True
    return mask


def interpolate_gaps(ball):
    out = ball.copy()
    idx = np.arange(len(ball))
    ok = ~np.isnan(ball[:, 0])
    for k in range(2):
        out[:, k] = np.interp(idx, idx[ok], ball[ok, k])
    return out


def corrupt(ball, H, A, gt, **kw):
    cfg = {**NOISE, **kw}
    rng = np.random.default_rng(cfg["seed"])
    n = len(ball)
    scale = np.array(PITCH)
    live = (gt == "H") | (gt == "A")
    dead = gt == "D"

    # 1. ball during stoppages: tracker sees it lying there
    filled = interpolate_gaps(ball)
    b = ball.copy()
    hole_dead = dead & np.isnan(ball[:, 0])
    b[hole_dead] = filled[hole_dead]
    b[dead & bursts(n, 1 - cfg["dead_visible"], 3 * cfg["burst_s"] * FPS, rng)] = np.nan

    # 2. live dropouts, biased towards crowded moments
    P = np.concatenate([H, A], 1)
    dist = np.linalg.norm((P - filled[:, None, :]) * scale, axis=2)
    crowd = np.nansum(dist < 3.0, axis=1)
    w = (1 + cfg["crowd_boost"] * crowd) * live
    target = cfg["live_miss"] * live.sum() / n
    b[bursts(n, target, cfg["burst_s"] * FPS, rng, w.astype(float))] = np.nan
    b += rng.normal(0, cfg["ball_jitter_m"], b.shape) / scale

    # 3. players: jitter + dropouts
    def noisy(T):
        T = T + rng.normal(0, cfg["jitter_m"], T.shape) / scale
        for j in range(T.shape[1]):
            T[bursts(n, cfg["player_drop"], 2 * FPS, rng), j] = np.nan
        return T
    return b, noisy(H), noisy(A)


def nearest_codes(ball, H, A):
    """The official baseline's step 1: team of the nearest player, -1 where no ball."""
    scale = np.array(PITCH)
    dh = np.nanmin(np.linalg.norm((H - ball[:, None]) * scale, axis=2), axis=1)
    da = np.nanmin(np.linalg.norm((A - ball[:, None]) * scale, axis=2), axis=1)
    codes = np.where(dh <= da, 0, 1).astype(np.int8)
    codes[np.isnan(ball[:, 0]) | (np.isnan(dh) & np.isnan(da))] = -1
    return codes


def score(gt, labels):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels), bootstrap_n=0)
    return r


def summary(name, r):
    keys = {k: r[k] for k in r if not isinstance(r[k], (dict, list, np.ndarray))}
    return keys

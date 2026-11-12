"""Step 10 -- HMM segmenter: replaces the baseline's majority vote.

Hidden state per frame: H, A or D.  What we observe per frame is a coarse symbol:

    0 NONE  ball not tracked
    1 HC    nearest player is home and close to the ball (<= CLOSE m)
    2 HF    nearest player is home but far
    3 AC    away, close
    4 AF    away, far
    5 OUT   ball tracked outside the pitch -> almost surely dead

Emissions P(symbol | state) and transitions P(state' | state) are COUNTED on a labelled game
(the lab -- another match, so allowed by the rules), then Viterbi finds the single most likely
H/A/D sequence. One knob on top: `switch_penalty` multiplies the probability of changing state,
so >1 means "stay put unless the evidence really insists".
"""
import warnings

import numpy as np

STATES = np.array(["H", "A", "D"])
N_SYM = 6
CLOSE = 2.0          # metres
PITCH = np.array([105.0, 68.0])


def symbols(ball, H, A, pitch=PITCH):
    """Per-frame observation symbol from ball + both teams' positions (normalised 0-1)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dh = np.nanmin(np.linalg.norm((H - ball[:, None]) * pitch, axis=2), axis=1)
        da = np.nanmin(np.linalg.norm((A - ball[:, None]) * pitch, axis=2), axis=1)
    home = dh <= da
    near = np.where(home, dh, da)
    sym = np.where(home, np.where(near <= CLOSE, 1, 2), np.where(near <= CLOSE, 3, 4))
    missing = np.isnan(ball[:, 0]) | np.isnan(near)
    out = (~missing) & ((ball[:, 0] < -0.005) | (ball[:, 0] > 1.005) |
                        (ball[:, 1] < -0.005) | (ball[:, 1] > 1.005))
    sym[missing] = 0
    sym[out] = 5
    return sym.astype(np.int8)


def fit(sym, gt, smoothing=1.0):
    """Count emissions and transitions from a labelled sequence (frames where gt is H/A/D)."""
    code = {"H": 0, "A": 1, "D": 2}
    g = np.array([code.get(x, -1) for x in gt])
    ok = g >= 0
    E = np.full((3, N_SYM), smoothing)
    np.add.at(E, (g[ok], sym[ok]), 1)
    E /= E.sum(1, keepdims=True)
    both = ok[:-1] & ok[1:]
    T = np.full((3, 3), smoothing)
    np.add.at(T, (g[:-1][both], g[1:][both]), 1)
    T /= T.sum(1, keepdims=True)
    return E, T


def viterbi(sym, E, T, switch_penalty=1.0, dead_penalty=None):
    """Most likely state path.

    switch_penalty divides P(H<->A); dead_penalty divides P(live<->D) (defaults to the same).
    """
    dead_penalty = switch_penalty if dead_penalty is None else dead_penalty
    T = T.copy()
    T[0, 1] /= switch_penalty
    T[1, 0] /= switch_penalty
    T[[0, 1, 2, 2], [2, 2, 0, 1]] /= dead_penalty
    T /= T.sum(1, keepdims=True)
    logE, logT = np.log(E), np.log(T)
    n = len(sym)
    back = np.zeros((n, 3), np.int8)
    score = np.log(np.full(3, 1 / 3)) + logE[:, sym[0]]
    for t in range(1, n):
        cand = score[:, None] + logT          # from-state x to-state
        back[t] = cand.argmax(0)
        score = cand.max(0) + logE[:, sym[t]]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return STATES[path]

"""Step 11 -- HMM v2 = v1 + two dead-ball fixes.

Fix 1  tag_prior(): Smart Tagging set pieces mark restarts. Frames shortly before a tagged
       restart get a boost towards D; frames far from every tag get pushed away from D.
Fix 2  baum_welch(): re-learn the emission table (how often each symbol shows up in H/A/D) on
       the match itself, with NO labels -- classic unsupervised EM. Transitions stay as fitted
       in the lab, so possessions keep their realistic lengths.
"""
import numpy as np

from hmm import STATES, N_SYM  # noqa: F401  (re-exported for callers)


def loglik(sym, E):
    """(n, 3) per-frame log P(symbol | state)."""
    return np.log(E)[:, sym].T.copy()


def penalised(T, team_penalty, dead_penalty):
    T = T.copy()
    T[0, 1] /= team_penalty
    T[1, 0] /= team_penalty
    T[[0, 1, 2, 2], [2, 2, 0, 1]] /= dead_penalty
    return T / T.sum(1, keepdims=True)


def viterbi_ll(L, T):
    """Viterbi on a per-frame log-likelihood matrix L (n, 3)."""
    logT = np.log(T)
    n = len(L)
    back = np.zeros((n, 3), np.int8)
    score = L[0] - np.log(3)
    for t in range(1, n):
        cand = score[:, None] + logT
        back[t] = cand.argmax(0)
        score = cand.max(0) + L[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return STATES[path]


def _forward_backward(L, T):
    """Scaled forward-backward. Returns posteriors gamma (n, 3) and total log-likelihood."""
    n = len(L)
    m = L.max(1, keepdims=True)
    B = np.exp(L - m)                     # per-frame likelihoods, rescaled per frame
    alpha = np.empty((n, 3))
    c = np.empty(n)
    a = np.full(3, 1 / 3) * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ T) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.empty((n, 3))
    beta[-1] = 1
    for t in range(n - 2, -1, -1):
        beta[t] = (T @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(1, keepdims=True)
    return gamma, np.log(c).sum() + m.sum()


def baum_welch(segments, E, T, iters=5, prior_weight=2000.0, extra=None):
    """Re-estimate emissions on unlabelled symbol sequences (one per half).

    prior_weight keeps the lab table as a pseudo-count prior, so EM adapts the noise levels
    without drifting into a different meaning for the states.
    extra: optional list of (n, 3) log-prior matrices (e.g. tag_prior) added during E-steps.
    """
    E = E.copy()
    for _ in range(iters):
        counts = prior_weight * E.copy()
        for k, sym in enumerate(segments):
            L = loglik(sym, E) + (extra[k] if extra is not None else 0)
            gamma, _ = _forward_backward(L, T)
            for s in range(N_SYM):
                counts[:, s] += gamma[sym == s].sum(0)
        E = counts / counts.sum(1, keepdims=True)
    return E


def tag_prior(n, restart_frames, pre_s=12.0, post_s=4.0, boost=3.0, far_s=20.0, shrink=1.5,
              fps=25):
    """(n, 3) log-prior on the D column from tagged restarts.

    [restart - pre_s, restart + post_s]  -> +boost   (dead ball likely: waiting for the restart)
    further than far_s from every tag    -> -shrink  (a stoppage here would usually be tagged)
    `restart_frames` are the tag START frames (the real restart sits a few seconds inside).
    """
    P = np.zeros((n, 3))
    near = np.zeros(n, bool)
    for r in restart_frames:
        lo, hi = max(0, int(r - pre_s * fps)), min(n, int(r + post_s * fps))
        P[lo:hi, 2] += boost
        near[max(0, int(r - far_s * fps)):min(n, int(r + far_s * fps))] = True
    P[~near, 2] -= shrink
    return P

"""Step 11b: make the emission table team-symmetric (no team 'owns' a lost ball)."""
import json, os, warnings, numpy as np, lab, hmm, hmm2
warnings.filterwarnings("ignore")
exec(open(os.path.join(lab.HERE,'try_hmm2.py')).read().split('def run')[0])

def symmetric(E):
    S = E.copy(); swap = [0, 3, 4, 1, 2, 5]          # NONE, HC<->AC, HF<->AF, OUT
    S[0] = S[1] = (E[0] + E[1][swap]) / 2
    S[1] = S[0][swap]
    return S
Es = symmetric(E)
print("NONE given H / A   before:", E[0,0].round(3), E[1,0].round(3), "  after:", Es[0,0].round(3), Es[1,0].round(3))
for name, noise in {"normal": {}, "harder": {"live_miss": 0.6}}.items():
    for tag, EE in (("v2", E), ("v2 + symmetric", Es)):
        v = []
        for seed in (1, 2, 3):
            b, h, a = lab.corrupt(ball, H, A, gt, seed=seed, **noise); sym = hmm.symbols(b, h, a)
            lbl = hmm2.viterbi_ll(hmm2.loglik(sym, EE) + hmm2.tag_prior(len(sym), tags(seed), shrink=0.75), T)
            r = lab.score(gt, lbl); g = (lbl != "X") & (gt != "X")
            v.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["share_err"]))
        v = np.array(v).mean(0); print(f"{name:<7}{tag:<16} spell {v[0]:.3f}  macro {v[1]:.3f}  poss {v[2]:.0f}  share_err {v[3]:.3f}")

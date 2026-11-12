"""Fit the HMM on one noisy version of Game 3, test it on OTHER noise seeds."""
import warnings, time
import numpy as np
import lab, hmm
from baseline import smooth
warnings.filterwarnings("ignore")

ball, H, A, gt = lab.load_clean()
b, h, a = lab.corrupt(ball, H, A, gt, seed=0)
E, T = hmm.fit(hmm.symbols(b, h, a), gt)
np.set_printoptions(precision=3, suppress=True)
print("emissions  NONE   HC    HF    AC    AF    OUT\n", np.c_[hmm.STATES, E.round(3)])
print("stay prob", np.diag(T).round(5))

rows = {}
for seed in (1, 2, 3):
    b, h, a = lab.corrupt(ball, H, A, gt, seed=seed)
    sym = hmm.symbols(b, h, a)
    r = lab.score(gt, smooth(lab.nearest_codes(b, h, a), 50))
    rows.setdefault("baseline50", []).append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
    for p in (1, 3, 10, 30, 100):
        r = lab.score(gt, hmm.viterbi(sym, E, T, p))
        rows.setdefault(f"hmm pen={p}", []).append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
print(f"\n{'system':<14}{'spellF1':>9}{'macroF1':>9}{'poss':>7}   (mean of seeds 1-3, truth = 404)")
for k, v in rows.items():
    v = np.array(v); print(f"{k:<14}{v[:,0].mean():>9.3f}{v[:,1].mean():>9.3f}{v[:,2].mean():>7.0f}   spell per seed {v[:,0].round(3)}")

"""Step 9 -- the lab check: run the official baseline on clean and on noisy Game 3.

If the noisy numbers sit near the published full-match curve, the lab is realistic and any idea
that beats the baseline here is worth trying on the real match.

    python lab/run_baseline.py
"""
import warnings
import numpy as np
import lab
from baseline import smooth

warnings.filterwarnings("ignore")
REAL = {0: 0.049, 25: 0.200, 50: 0.243, 125: 0.224}   # published, EVALUATION.md

ball, H, A, gt = lab.load_clean()
graded = gt != "X"
for tag, (b, h, a) in {"clean": (ball, H, A), "noisy": lab.corrupt(ball, H, A, gt)}.items():
    codes = lab.nearest_codes(b, h, a)
    print(f"\n{tag}: ball missing on {np.isnan(b[graded, 0]).mean():.0%} of graded frames")
    print(f"  {'window':>8} {'spellF1':>8} {'real':>6} {'macroF1':>8} {'poss':>6}")
    for w, real in REAL.items():
        r = lab.score(gt, smooth(codes, w))
        print(f"  {w:>8} {r['spell_f1_mean']:>8.3f} {real:>6.3f} {r['macro_f1']:>8.3f} {r['n_pred_spells']:>6}")
print(f"\nground truth: {lab.score(gt, gt)['n_gt_spells']} possessions")

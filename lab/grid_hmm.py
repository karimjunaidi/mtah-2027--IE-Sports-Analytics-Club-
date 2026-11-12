import warnings, numpy as np, lab, hmm
warnings.filterwarnings("ignore")
ball,H,A,gt=lab.load_clean()
b,h,a=lab.corrupt(ball,H,A,gt,seed=0); E,T=hmm.fit(hmm.symbols(b,h,a),gt)
syms=[hmm.symbols(*lab.corrupt(ball,H,A,gt,seed=s)) for s in (1,2,3)]
res=[]
for sp in (100,1000,10000,1e5):
  for dp in (10,100,1000,1e4):
    v=[lab.score(gt,hmm.viterbi(x,E,T,sp,dp)) for x in syms]
    res.append((np.mean([r['spell_f1_mean'] for r in v]),sp,dp,np.mean([r['macro_f1'] for r in v]),np.mean([r['n_pred_spells'] for r in v]),[round(r['spell_f1_mean'],3) for r in v]))
for r in sorted(res,reverse=True)[:8]: print(f"spell {r[0]:.3f}  team_pen {r[1]:g}  dead_pen {r[2]:g}  macro {r[3]:.3f}  poss {r[4]:.0f}  seeds {r[5]}")

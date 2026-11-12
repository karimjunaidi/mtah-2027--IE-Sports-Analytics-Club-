import json, os, warnings, numpy as np, lab, hmm, hmm2
warnings.filterwarnings("ignore")
exec(open(os.path.join(lab.HERE,'try_hmm2.py')).read().split('def run')[0])  # reuse setup + tags()
data={}
for name,noise in {"normal":{}, "harder":{"live_miss":0.6}}.items():
    data[name]=[]
    for seed in (1,2,3):
        b,h,a=lab.corrupt(ball,H,A,gt,seed=seed,**noise); sym=hmm.symbols(b,h,a)
        data[name].append((sym,tags(seed)))
res=[]
for tp in (1e3,3e3,1e4):
  for dp in (3e3,1e4):
    for shrink in (0.75,1.5):
      Tp=hmm2.penalised(T0,tp,dp); row=[]
      for name in data:
        for sym,tg in data[name]:
          r=lab.score(gt,hmm2.viterbi_ll(hmm2.loglik(sym,E)+hmm2.tag_prior(len(sym),tg,shrink=shrink),Tp))
          row.append((r['spell_f1_mean'],r['n_pred_spells']))
      row=np.array(row); res.append((row[:3,0].mean(),row[3:,0].mean(),tp,dp,shrink,row[:3,1].mean(),row[3:,1].mean()))
res.sort(key=lambda x:-(x[0]+x[1]))
print("normal  harder   team_pen dead_pen shrink   poss(normal/harder)")
for r in res[:6]: print(f"{r[0]:.3f}   {r[1]:.3f}   {r[2]:>8g} {r[3]:>8g} {r[4]:>6}   {r[5]:.0f}/{r[6]:.0f}")

"""Lab test for step 11: v1 vs +tags vs +Baum-Welch vs both, on normal AND harder noise."""
import json, os, warnings
import numpy as np
import lab, hmm, hmm2
warnings.filterwarnings("ignore")

ball, H, A, gt = lab.load_clean()
b, h, a = lab.corrupt(ball, H, A, gt, seed=0)
E, T0 = hmm.fit(hmm.symbols(b, h, a), gt)
T = hmm2.penalised(T0, 1e4, 1e4)

# simulated Smart Tagging: real restarts from Game 3's events, ~60% of them tagged,
# tag window starting 2-4 s before the restart (like the real tags)
ev = json.load(open(os.path.join(lab.HERE, "sample-data", "data", "Sample_Game_3", "Sample_Game_3_events.json")))["data"]
restarts = np.array([e["start"]["frame"] - 1 for e in ev
                     if e["type"]["name"] == "SET PIECE" or
                     (e["type"]["name"] == "PASS" and "GOAL KICK" in str(e.get("subtypes")))])

def tags(seed):
    r = np.random.default_rng(100 + seed)
    keep = restarts[r.random(len(restarts)) < 0.6]
    return keep - r.uniform(2, 4, len(keep)) * 25

def run(noise):
    out = {}
    for seed in (1, 2, 3):
        b, h, a = lab.corrupt(ball, H, A, gt, seed=seed, **noise)
        sym = hmm.symbols(b, h, a); n = len(sym)
        P = hmm2.tag_prior(n, tags(seed))
        E_bw = hmm2.baum_welch([sym], E, T, iters=4)
        E_bwt = hmm2.baum_welch([sym], E, T, iters=4, extra=[P])
        systems = {"v1": hmm2.loglik(sym, E),
                   "v1 + tags": hmm2.loglik(sym, E) + P,
                   "v1 + BW": hmm2.loglik(sym, E_bw),
                   "v2 (tags+BW)": hmm2.loglik(sym, E_bwt) + P}
        for k, L in systems.items():
            lbl = hmm2.viterbi_ll(L, T); r = lab.score(gt, lbl)
            g = gt != "X"
            out.setdefault(k, []).append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], (lbl[g] == "D").mean()))
    return out

for name, noise in {"normal lab noise": {}, "harder: ball lost more in live play": {"live_miss": 0.6}}.items():
    print(f"\n{name}   (truth: 404 possessions, D share 0.27)")
    print(f"  {'system':<14}{'spellF1':>8}{'macroF1':>9}{'poss':>6}{'D share':>9}")
    for k, v in run(noise).items():
        v = np.array(v).mean(0); print(f"  {k:<14}{v[0]:>8.3f}{v[1]:>9.3f}{v[2]:>6.0f}{v[3]:>9.2f}")

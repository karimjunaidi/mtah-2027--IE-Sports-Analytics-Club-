"""HMM v2 on the real match: v1 + Smart Tagging set pieces as a dead-ball prior.

    python pipeline/run_hmm_v2.py              # writes evaluation/results/hmm_v2.csv

Every tag with code SET PIECES (throw-in, corner, goal kick, free kick, kick-off) marks a
restart. Frames just before it lean towards D; frames far from every tag lean away from D.
Smart Tagging is automatic (not human), so the rules allow it -- see FAQ.md.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "starter"), HERE):
    sys.path.insert(0, p)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402
import hmm  # noqa: E402
import hmm2  # noqa: E402
from run_hmm import read_atd  # noqa: E402


def restart_frames(game=GAME, data_dir=None):
    tags = load_smart_tagging.load(game, data_dir)
    return np.array([t["start_frame"] for t in tags if t["code"].strip().lower() == "set pieces"])


def run(game=GAME, data_dir=None, params_path=os.path.join(HERE, "hmm_params.json")):
    p = json.load(open(params_path))
    E, T = np.array(p["emissions"]), np.array(p["transitions"])
    T = hmm2.penalised(T, p["team_penalty"], p["dead_penalty"])
    hmm.CLOSE = p["close_m"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n = source_frames("tactical", game)
    print("reading ATD ...")
    ball, players = read_atd(raw_path, n, len(team))
    sym = hmm.symbols(ball, players[:, team == 0], players[:, team == 1],
                      pitch=np.array(GAMES[game]["pitch"]))
    restarts = restart_frames(game, data_dir)
    print("set-piece tags used:", len(restarts))
    L = hmm2.loglik(sym, E) + hmm2.tag_prior(n, restarts, **p["tag_prior"])
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        labels[first:last + 1] = hmm2.viterbi_ll(L[first:last + 1], T)
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(HERE, "hmm_params.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "hmm_v2.csv"))
    args = ap.parse_args()
    labels = run(params_path=args.params)
    write_submission(args.out, labels)
    graded = labels != "X"
    print({k: int((labels == k).sum()) for k in "HADX"}, "->", os.path.relpath(args.out, ROOT))
    print("home share of live frames: {:.0%}".format(
        (labels == "H").sum() / max(1, ((labels == "H") | (labels == "A")).sum())))
    print("D share of graded frames: {:.0%}   (v1 was ~42%; lab truth ~27%)".format(
        (labels[graded] == "D").mean()))

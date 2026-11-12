"""Real-match run: ATD -> per-frame symbols -> HMM/Viterbi -> submission CSV.

    python pipeline/run_hmm.py                    # writes evaluation/results/hmm_v1.csv

Parameters come from pipeline/hmm_params.json, fitted on the public Metrica Sample Game 3 in
the lab -- nothing here is fitted on, or tuned against, the provided match.
Each graded half is decoded separately; ungraded frames are written as X.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402
import hmm  # noqa: E402


def read_atd(raw_path, n_frames, n_slots):
    """ball (n,2) and players (n,slots,2), normalised 0-1, NaN where missing. Row = video frame."""
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1                     # frame_id counts from 1
            if not 0 <= row < n_frames:
                continue
            bx, by = parts[2].split(",")[:2]
            ball[row] = (float(bx), float(by))          # float("NaN") is nan
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                px, py = group.split(",")[:2]
                players[row, slot] = (float(px), float(py))
    return ball, players


def run(game=GAME, data_dir=None, params_path=os.path.join(HERE, "hmm_params.json")):
    p = json.load(open(params_path))
    E, T = np.array(p["emissions"]), np.array(p["transitions"])
    hmm.CLOSE = p["close_m"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n = source_frames("tactical", game)
    print("reading ATD ...")
    ball, players = read_atd(raw_path, n, len(team))
    H, A = players[:, team == 0], players[:, team == 1]
    sym = hmm.symbols(ball, H, A, pitch=np.array(GAMES[game]["pitch"]))
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        labels[first:last + 1] = hmm.viterbi(sym[first:last + 1], E, T,
                                             p["team_penalty"], p["dead_penalty"])
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "hmm_v1.csv"))
    args = ap.parse_args()
    labels = run()
    write_submission(args.out, labels)
    print({k: int((labels == k).sum()) for k in "HADX"}, "->", os.path.relpath(args.out, ROOT))
    print("now: python evaluation/validate_submission.py", os.path.relpath(args.out, ROOT))

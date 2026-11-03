"""Step 7 -- turn Metrica Sample Game 3's human event data into a frame,label ground truth.

Rules (mirroring CHALLENGE.md as closely as event data allows):
  * A team is in possession from its control event (pass, carry, recovery, set piece, shot)
    until the OTHER team makes a control event. Challenges and lost balls do not hand it over.
  * Dead ball: from the end of the event before any restart (set piece or goal kick) to the
    restart itself, and from a foul / ball out until the next restart.
  * Outside the two halves (before kick-off, halftime, after the end) -> X, never scored.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.environ.get("METRICA_SAMPLE_DIR", os.path.join(HERE, "sample-data"))
EV = os.path.join(SAMPLE, "data", "Sample_Game_3", "Sample_Game_3_events.json")
N = 143761                      # tracking frames 1..N -> GT rows 0..N-1
CONTROL = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}


def subtype(e):
    s = e.get("subtypes")
    if isinstance(s, list):
        return {x["name"] for x in s}
    return {s["name"]} if s else set()


def is_restart(e):
    t, st = e["type"]["name"], subtype(e)
    return t == "SET PIECE" or (t == "PASS" and "GOAL KICK" in st)


def build():
    ev = sorted(json.load(open(EV))["data"], key=lambda e: (e["start"]["frame"], e["index"]))
    home_id = "FIFATMA"
    lab = np.full(N, "X", dtype="<U1")

    for period in (1, 2):
        pe = [e for e in ev if e["period"] == period]
        first, last = pe[0]["start"]["frame"], max(e["end"]["frame"] for e in pe)
        team, dead_from = None, None
        # walk frame boundaries event by event
        for i, e in enumerate(pe):
            s = e["start"]["frame"]
            nxt = pe[i + 1]["start"]["frame"] if i + 1 < len(pe) else last + 1
            t = e["type"]["name"]
            if is_restart(e) and i > 0:
                # ball was dead between the previous event's end and this restart
                prev_end = max(pe[i - 1]["end"]["frame"], pe[i - 1]["start"]["frame"])
                lab[prev_end - 1:s - 1] = "D"
            if t in CONTROL and e["team"]:
                team = "H" if e["team"]["id"] == home_id else "A"
            if t in ("BALL OUT", "FAULT RECEIVED"):
                end = max(e["end"]["frame"], s)
                lab[end - 1:nxt - 1] = "D"
                continue
            if team:
                seg = lab[s - 1:nxt - 1]
                seg[seg == "X"] = team
        lab[first - 1:last][lab[first - 1:last] == "X"] = "D"   # any hole inside a half
    return lab


if __name__ == "__main__":
    lab = build()
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    with open(os.path.join(HERE, "cache", "g3_gt.csv"), "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(lab))
    g = lab[lab != "X"]
    print("graded", len(g), {k: round((g == k).mean(), 3) for k in "HAD"})

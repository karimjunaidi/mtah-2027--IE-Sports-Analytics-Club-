"""Regenerate the submission file from the provided data bundle, end to end.

    python make_submission.py                       # data in ./data or $MTAH_DATA_DIR
    python make_submission.py --data-dir /path/to/mtah2027
    python make_submission.py --check-repro         # run twice, require identical bytes

What it does (nothing else, no hidden steps):
  1. reads the ATD tracking and the Smart Tagging XML from the data bundle
  2. runs the HMM v3 pipeline (pipeline/run_hmm_v2.py with pipeline/hmm_params_v3.json)
  3. writes the possession CSV (header `frame,label`, 199,319 rows)
  4. validates it with the official evaluation/validate_submission.py
  5. prints the SHA-256 of the file, so two machines can confirm they got the same file

The model is deterministic: Viterbi decoding over fixed, pre-fitted parameters. No random
numbers are drawn anywhere. The seed set below is belt-and-braces only.
"""
import argparse
import hashlib
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(ROOT, "pipeline", "hmm_params_v3.json")
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "hmm_v3.csv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(out_path):
    # games.py reads MTAH_DATA_DIR when it is imported, so it must be set before the import below.
    sys.path.insert(0, os.path.join(ROOT, "pipeline"))
    import numpy as np
    import run_hmm_v2
    from baseline import write_submission

    random.seed(0)
    np.random.seed(0)
    labels = run_hmm_v2.run(params_path=PARAMS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_submission(out_path, labels)
    print({k: int((labels == k).sum()) for k in "HADX"}, "->", os.path.relpath(out_path, ROOT))


def validate(out_path):
    cmd = [sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out_path]
    print("validating:", " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd[1:]))
    if subprocess.call(cmd) != 0:
        sys.exit("VALIDATION FAILED -- do not submit this file")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="folder holding the unpacked data bundle (default: ./data or $MTAH_DATA_DIR)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to write the CSV")
    ap.add_argument("--check-repro", action="store_true",
                    help="build the file twice and fail unless both are byte-identical")
    args = ap.parse_args()

    if args.data_dir:
        os.environ["MTAH_DATA_DIR"] = os.path.abspath(args.data_dir)

    if args.check_repro:
        # Run each build in a fresh interpreter so nothing is shared between the two.
        digests = []
        for i in (1, 2):
            out = args.out + ".run%d" % i
            cmd = [sys.executable, os.path.abspath(__file__), "--out", out]
            if args.data_dir:
                cmd += ["--data-dir", args.data_dir]
            if subprocess.call(cmd) != 0:
                sys.exit("build %d failed" % i)
            digests.append(sha256(out))
            os.replace(out, args.out) if i == 2 else os.remove(out)
        print("run 1 sha256:", digests[0])
        print("run 2 sha256:", digests[1])
        if digests[0] != digests[1]:
            sys.exit("NOT REPRODUCIBLE: the two runs differ")
        print("REPRODUCIBLE: identical bytes")
        return

    build(args.out)
    validate(args.out)
    print("sha256:", sha256(args.out))


if __name__ == "__main__":
    main()

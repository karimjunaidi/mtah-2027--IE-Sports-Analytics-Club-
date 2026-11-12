# More Than A Hack 2027 — Phase 1 submission (possession labelling, BAR–ATH)

Team: **[TEAM NAME]** · IE Sports Analytics Club

This repository labels every frame of the provided Barça vs Athletic match as `H` (home in
possession), `A` (away in possession), `D` (dead ball) or `X` (abstain), using a 3-state hidden
Markov model over the Metrica ATD tracking and the Smart Tagging set-piece tags. No frame is
labelled by hand, no private ground truth is used, and no match data is stored in this repository.

The method is described in [TECHNICAL_DOC.md](TECHNICAL_DOC.md). The challenge's own documents
(`CHALLENGE.md`, `RULES.md`, `SUBMISSION.md`, `EVALUATION.md`, `DATA.md`, `FAQ.md`) are kept
unchanged from the official starter repository.

## Reproduce the submission file

You need the gated data bundle from the organisers. It is **not** in this repository.

```bash
# 1. Python 3.10 or newer, then the pinned dependencies
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Unpack the data bundle into ./data  (see DATA.md), or point at it:
export MTAH_DATA_DIR=/path/to/mtah2027                    # Windows PowerShell: $env:MTAH_DATA_DIR = "D:\mtah2027"

# 3. Build the file (this is the command that regenerates the submission)
python make_submission.py
```

This writes `evaluation/results/hmm_v3.csv`, validates it with the official
`evaluation/validate_submission.py`, and prints the file's SHA-256.

To confirm the build is reproducible, run it twice in fresh interpreters and compare the bytes:

```bash
python make_submission.py --check-repro
```

It prints `REPRODUCIBLE: identical bytes` when both runs match. The model draws no random numbers:
the parameters are fixed in `pipeline/hmm_params_v3.json` and decoding is Viterbi, which is
deterministic.

The submitted file has a header `frame,label` and 199,319 rows, one per video frame.

## What is in here

| path | what |
|---|---|
| `make_submission.py` | one-command entry point: build, validate, print hash |
| `pipeline/run_hmm_v2.py` | reads ATD + Smart Tagging, builds the per-frame symbols, decodes each half |
| `pipeline/hmm.py`, `pipeline/hmm2.py` | symbols and Viterbi (`hmm.py`); set-piece prior and log-likelihood helpers (`hmm2.py`) |
| `pipeline/hmm_params_v3.json` | the parameters actually used (team-symmetric emissions, transitions, penalties, set-piece prior) |
| `pipeline/hmm_params.json`, `pipeline/run_hmm.py` | earlier versions (v1/v2), kept for the record |
| `lab/` | practice lab on the free Metrica Sample Game 3, used to choose the model and its settings; see `lab/README.md` |
| `evaluation/`, `starter/`, `games.py` | official scorer, validator and loaders from the starter repository |

`lab/` is optional for reproducing the submission. It needs the public
[sample data](https://github.com/metrica-sports/sample-data) and `pip install kloppy`; its steps are
in `lab/README.md`.

## Rules this repository follows

- Labels come from an algorithm only. Nothing is hand-labelled or hand-corrected on the provided game.
- The only labelled data used for fitting comes from another match, Metrica Sample Game 3.
- The public 5 minutes are used for sanity checks only, not for tuning.
- No match data, video or ground-truth file is committed (`.gitignore` blocks the usual ones).

# Pipeline (what runs on the real match)

```bash
python pipeline/run_hmm.py                                   # ~1-2 min
python evaluation/validate_submission.py evaluation/results/hmm_v1.csv
python evaluation/run_evaluation.py                          # public 5 min: sanity only
```

| file | what |
|---|---|
| `run_hmm.py` | reads the ATD, builds per-frame symbols, decodes each half with Viterbi |
| `hmm.py` | symbols + Viterbi (same code as `lab/hmm.py`) |
| `hmm_params.json` | emissions/transitions counted on public Sample Game 3 + the two switch penalties chosen in the lab |

v1 lab result (noisy Game 3, mean of 3 unseen noise seeds): spellF1 **0.368** vs baseline 0.212.

## v2 (step 11): + Smart Tagging dead-ball prior
```bash
python pipeline/run_hmm_v2.py
python evaluation/validate_submission.py evaluation/results/hmm_v2.csv
python evaluation/run_evaluation.py --no-detail
```
Lab (mean of 3 unseen seeds): normal noise 0.368 -> 0.368, harder noise 0.308 -> 0.323;
dead-ball share 0.37 -> 0.24 (truth 0.27). Baum-Welch re-fitting was tried and dropped
(over-segments: 1000+ possessions vs 404).

# Experiment archive

All process notebooks are collected in [`notebook/experiments/`](../notebook/experiments/), with original filenames and saved outputs retained. Tests use this archive location. This index groups experiments by filename/topic; it does not infer leaderboard improvements from those names.

| Family | Notebook range / examples | Explored direction |
|---|---|---|
| Diagnosis and early repair | `01`–`14` | Validation, pruning, retain fine-tuning, NFT, trigger inversion, distillation, unlearn/relearn |
| Filtering and rescoring | `15`–`32` | Geometry, density, crop networks, embeddings, DINO, confidence shaping and candidate merging |
| Detector repair | `33a`–`39` | EWC, ensembles, multiscale recall, trigger/detection-aware repair, clean and foreground KD |
| Probes and ablations | `40`–`46` | Trigger reconstruction, poison identity, prototype holdout, exact ablation, contrastive amplification |
| Saliency and cross-fitting | `48`–`50` | Saliency scrub, real-anchor distillation, nested alpha/K sweeps |
| **Final solution** | [`v126`](../notebooks/final/neural-debris-v126.ipynb), local `51_deadline_fixed_alpha_sweep.ipynb` | Standalone fixed-alpha sweep with gate-controlled exact-K exports |
| Subsequent alternatives | `52`–`56` | Sparse exact-K, response fingerprints, continuous calibration, confidence-floor sweep, PCGrad |

The same `notebook/experiments/` directory contains historical notebooks named `229`, `243`, `245`, `247`, `250`, `263`, and `290`. These are historical filenames, not asserted Kaggle version numbers or confirmed scores.

Local NB51 includes saved execution logs. The downloaded final v126 is archived separately so the project does not assume a local notebook with a similar title is identical to the submitted source. Research design/plan notes are in `docs/superpowers/`.

Final competition placement: **13/567**, reported by the author. Per-experiment leaderboard scores are not tabulated without a submission record linking each score to its exact artifact.

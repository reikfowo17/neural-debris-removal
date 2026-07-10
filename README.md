# Neural Debris Removal in Streak Detection Models

**Kaggle competition solution by heon29 · Final placement: 13 / 567 (top 2.3%)**

This repository combines a Python reproduction of my final competition notebook with an archive of the experiments behind it. The task is to remove poisoned responses from a streak detector while retaining useful detections.

The final solution is **NB51: source-disjoint contrastive scrubbing and real-anchor distillation with fixed-alpha ranking calibration**, taken from **Kaggle version 126**. The placement above is the author's reported final competition result.

- [Competition](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models)
- [Submitted notebook version](https://www.kaggle.com/code/heon29/neural-debris?scriptVersionId=337177972)
- [Archived v126 notebook](notebooks/final/neural-debris-v126.ipynb)
- [Solution explanation](docs/solution.md)
- [Reproduction guide](docs/reproduction.md)
- [Experiment archive](docs/experiments.md)

## Approach

1. Load the provided poisoned RetinaNet detector and construct matched poison/control pairs.
2. Train a legacy response ranker and mine reliable real-image retention anchors.
3. Split poison sources and anchor images into four folds.
4. Build stable contrastive saliency masks and train scrub models using forget loss, localized knowledge distillation and parameter regularization.
5. Require suppression, retention, disjointness and ranking gates to pass.
6. Combine legacy and cross-fitted scrub responses, select a deterministic exact-K keep set, and recalibrate confidences while retaining baseline boxes.

## Quick start

Use a Linux/Kaggle environment with CUDA and a working Detectron2 installation. See the [environment instructions](docs/reproduction.md) first.

```bash
python -m pip install -e ".[runtime]"
neural-debris inspect --config configs/v126.json
neural-debris check-data --data-root /path/to/extracted/competition
neural-debris run --data-root /path/to/extracted/competition --output-dir outputs/v126
```

If your Python scripts directory is not on PATH, use `python -m neural_debris` in place of `neural-debris` (for example, `python -m neural_debris inspect`).

`configs/v126.json` is empty intentionally: every default is read from the archived v126 configuration. Omit `--config` for the same behavior. Every run requires an empty output directory and writes `reproduction.json` containing configuration, environment, source hash, stage timings and completion status.

The original notebook exports four candidates:

| Candidate | Scrub weight α | Keep count K | Role in v126 |
|---|---:|---:|---|
| `nb51_A065_K466.csv` | 0.65 | 466 | Primary |
| `nb51_A080_K466.csv` | 0.80 | 466 | Stronger scrub weighting |
| `nb51_A065_K450.csv` | 0.65 | 450 | Precision hedge |
| `nb51_A065_K480.csv` | 0.65 | 480 | Recall hedge |

K is capped by the number of eligible candidates, exactly as in v126. Outputs are gate-controlled; failed training gates stop export. The specific CSV used for the final leaderboard placement has not been separately recorded here.

## Repository

```text
src/neural_debris/       CLI, configuration and execution runner
  stages/               17 readable Python computation stages from v126
configs/                JSON configuration overrides
notebooks/final/        Downloaded v126 and Kaggle metadata
01_...56_*.ipynb         Original experimental notebook archive
notebook/               Historical executed notebooks
docs/                   Solution, reproduction and archive navigation
tests/                  CPU regression checks and original notebook tests
tools/extract_v126.py    Regenerate stages from the archived source
```

The stage runner deliberately shares a namespace to preserve notebook global dependencies. Stages are Python source files, not notebook execution through Jupyter; automatic dependency installation is excluded from training. See [reproduction details](docs/reproduction.md) for fidelity and validation boundaries.

## Verification

```bash
python -m unittest discover -s tests -p test_reproduction.py -v
python -m pytest tests -q
```

CPU checks cover lossless source extraction, override timing, shared-state execution, stable exact-K selection and submission serialization. Full CUDA training and identical leaderboard scores have not been revalidated locally.

Competition data, model weights, credentials and generated outputs are excluded from Git. Download inputs through Kaggle; existing experiment notebooks retain their historical names and paths.

# Reproduction guide

## Environment

The downloaded v126 metadata specifies an NVIDIA Tesla T4 and Kaggle image:

```text
gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c
```

The downloaded notebook declares Python 3.12. A historical local NB51 run reports Torch 2.10.0+cu128 and torchvision 0.25.0+cu128 on T4; that log is not a lockfile for the downloaded version. Detectron2 was installed from its GitHub repository without a pinned commit in the original notebook, so exact environment/bitwise reproduction cannot be guaranteed from the original metadata alone.

On Kaggle, install Detectron2 against the active Torch before running the pipeline:

```bash
python -m pip install 'setuptools<81'
python -m pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install -e ".[runtime]"
```

For local Linux, install a CUDA-compatible Torch/torchvision pair first, then compile Detectron2 against it. T4 is the reference accelerator. The archived `notebooks/final/environment_setup.py` preserves the original setup procedure including its Pascal/P100 Torch downgrade; it is not invoked automatically by the CLI.

## Data

Obtain and extract competition files into one root directory:

```bash
kaggle competitions download -c neural-debris-removal-in-streak-detection-models -p data
```

Extract downloaded archives before running. The input root must contain a `.pth` detector checkpoint, `annotations_coco.json`, PNG images and `sample_submission.csv`, recursively. The existing workspace layout is `data/poisoned_model`, `data/unlearn_set`, `data/test_set` and `data/sample_submission.csv`.

The original discovery logic is preserved: prefer checkpoint paths containing `poison`, take the first sorted annotation/sample file, and treat the directory with the most PNGs as the test directory. Point the CLI at an isolated competition data root, then inspect the printed discovered paths at the beginning of the run. `check-data` checks presence, not annotation content or checkpoint compatibility.

## Execution

```bash
neural-debris inspect
neural-debris check-data --data-root data
neural-debris run --data-root data --output-dir outputs/v126
```

The equivalent invocation `python -m neural_debris` works when the console-script directory is not on PATH. On the local Windows verification environment, this module invocation was used successfully.

The output directory must be empty or absent. Training follows all 17 stages in manifest order; there is no checkpoint-resume mode. Use a new directory for each run. The pipeline writes the original checkpoints, audit tables, gate reports and candidate CSVs. `reproduction.json` records configuration, software versions, GPU, source fingerprint, timings and errors if a stage fails.

To change configuration, create a JSON object using the exact keys displayed by `inspect`, then pass `--config path/to/config.json`. Changes to seeds, folds, thresholds or iterations describe a new experiment, not default v126 reproduction. Invalid types and unknown keys are rejected before GPU execution.

## Fidelity

The stage files preserve code cells byte-for-byte apart from trailing whitespace/newline handling. Execution applies three AST-level adaptations: update CFG immediately after its definition, replace WORK with the chosen output directory, and replace CANDIDATE_ROOTS with the supplied data root. This preserves seed timing, default-argument capture and global bindings. No training formula or ranking tie-break is changed.

Regenerate extracted stages:

```bash
python tools/extract_v126.py notebooks/final/neural-debris-v126.ipynb
```

The extractor skips the environment-installation cell and records it in the manifest. The source notebook remains the authority for v126. Directly importing individual stages is unsupported because they intentionally share notebook state through the runner.

## Verification boundaries

CPU regression tests compare all extracted cells with the archived source and exercise configuration timing, shared state, deterministic selection and confidence/CSV formatting. Original notebook tests remain available. Full training needs Detectron2 and CUDA, and must be run separately to compare resulting gate reports and submission files. Random seeds reduce variation; device kernels and unpinned original dependencies can still change numerical results.

The initial full-suite check on 2026-10-10 returned 118 passed and two failures in historical NB54 tests: a missing `_build_nb54.py` script and a test expecting empty notebook outputs even though the archived NB54 was executed. The reproduction-specific tests pass; historical outputs are retained as experiment evidence.

# Reproducible v126 Implementation Plan

**Goal:** Publishable portfolio and archive with an executable Python reproduction of NB51/v126.

**Architecture:** Extract ordered code cells into inspectable Python stages with a shared execution namespace. A lightweight CLI supplies paths/configuration, while the original notebook and extraction manifest provide source provenance.

**Tech Stack:** Python, PyTorch, Detectron2, NumPy, pandas, OpenCV, scikit-learn, unittest.

## Tasks

- [x] Add extraction regression tests: source fingerprint, code-cell preservation, removal of installation cell.
- [x] Implement `tools/extract_v126.py`; archive downloaded notebook and metadata; emit Python stages and manifest.
- [x] Add runner tests for shared namespace, overridden paths/configuration and invalid input rejection.
- [x] Implement configuration, stage runner and CLI, with no GPU import for help/inspection.
- [x] Add packaging, environment setup documentation, solution explanation and experiment index.
- [x] Run CPU tests, source compilation, CLI inspection and existing notebook tests; record CUDA verification limitations. Eight reproduction tests pass; full suite returned 118 passed and two historical NB54 failures documented in reproduction.md.

Execution is inline in this session. No commit is made until repository contents and publication destination have been reviewed.

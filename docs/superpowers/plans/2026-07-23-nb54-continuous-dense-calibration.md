# NB54 Continuous Dense Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Kaggle notebook that reproduces NB51 training and exports three dense continuous-confidence submissions without dropping detector candidates.

**Architecture:** Reuse the already executed NB51 training and hard-gate cells, then replace only the submission composer. Compute deterministic global clean percentile from A065 poison scores, map it continuously to confidence for three gamma values, apply mild overlap damping without omission, and fail closed on gate or structural violations.

**Tech Stack:** Python, Jupyter nbformat JSON, NumPy, pandas, PyTorch, Detectron2, pytest.

---

### Task 1: Define NB54 Contracts

**Files:**
- Create: `tests/test_nb54_notebook.py`

- [ ] **Step 1: Write failing structural tests**

Test that `54_continuous_dense_calibration.ipynb` is standalone, contains A065 scoring, gamma variants 1/2/3, continuous confidence mapping, duplicate damping, gate-protected exports, and no sparse/exact-K omission path.

- [ ] **Step 2: Run RED test**

Run: `python -m pytest tests/test_nb54_notebook.py -q`
Expected: FAIL because the notebook does not exist.

- [ ] **Step 3: Add executable composer tests**

Extract the pure calibration helpers from the notebook and assert candidate count preservation, unchanged boxes, finite confidence in `[0.02, 1.0]`, deterministic ties, monotonic clean-rank mapping for equal teacher confidence, and non-removal under overlap damping.

### Task 2: Build Standalone NB54

**Files:**
- Create: `_build_nb54.py`
- Create: `54_continuous_dense_calibration.ipynb`

- [ ] **Step 1: Reuse proven training cells**

Load the clean source notebook `51_deadline_fixed_alpha_sweep.ipynb`, copy its standalone setup through hard-gate/test-component collection cells, clear all execution outputs, and rename NB51 artifacts to NB54.

- [ ] **Step 2: Add deterministic calibration helpers**

Implement global A065 poison score, stable global clean percentile with image/candidate tie-breaks, confidence interpolation, and mild duplicate damping that floors at `0.02` and never filters rows.

- [ ] **Step 3: Add fail-closed exports**

Export `nb54_CONTINUOUS_G2.csv`, `nb54_CONTINUOUS_G1.csv`, and `nb54_CONTINUOUS_G3.csv` only under `ALL_GATES_PASS`. Assert exact per-image candidate and box preservation, confidence validity, and one primary manifest row.

- [ ] **Step 4: Build and run GREEN tests**

Run: `python _build_nb54.py && python -m pytest tests/test_nb54_notebook.py -q`
Expected: all NB54 tests pass.

### Task 3: Regression And Reproducibility Verification

**Files:**
- Verify: `_build_nb54.py`
- Verify: `54_continuous_dense_calibration.ipynb`
- Verify: `tests/test_nb49_notebook.py` through `tests/test_nb54_notebook.py`

- [ ] **Step 1: Parse every code cell**

Compile every NB54 code cell with Python and validate notebook schema fields, empty outputs, and runtime dependency order.

- [ ] **Step 2: Run regression suite**

Run: `python -m pytest tests/test_nb49_notebook.py tests/test_nb50_notebook.py tests/test_nb51_notebook.py tests/test_nb52_notebook.py tests/test_nb53_notebook.py tests/test_nb54_notebook.py -q`
Expected: all tests pass.

- [ ] **Step 3: Verify deterministic build**

Build NB54 twice and compare SHA256 hashes. Expected: hashes are identical.

- [ ] **Step 4: Inspect final diff and report run instructions**

Confirm no existing NB49-NB53 notebook was modified. Report the notebook path, expected runtime, primary output, hedge policy, test count, and SHA256.

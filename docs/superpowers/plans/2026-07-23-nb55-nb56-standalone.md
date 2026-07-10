# NB55 and NB56 Standalone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two independent Kaggle notebooks that retrain from the poisoned model and competition data, with a conservative NB55 floor sweep and a gated NB56 PCGrad/box-KD repair.

**Architecture:** NB55 reuses only the conceptual training protocol, not NB49-NB54 runtime artifacts, and exports A065 K466 with tail floors 0.03/0.04/0.05. NB56 trains head-only repair candidates with separate forget and real-anchor classification/box losses, applies PCGrad, evaluates source-disjoint gates, and only exports a dense submission after gates pass.

**Tech Stack:** Python, Jupyter, PyTorch, Detectron2, NumPy, pandas, pytest.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/test_nb55_nb56_notebooks.py`

- [ ] **Step 1: Write failing standalone tests**

Assert both notebooks exist only after generation, contain poisoned-model/data loading, no checkpoint/CSV dependency on NB49-NB54, correct output names, and cleared outputs. Assert NB55 has floors 0.03/0.04/0.05, K466 and A065; assert NB56 contains PCGrad, classification KD, box KD, source-disjoint gates, and fail-closed export.

- [ ] **Step 2: Run RED**

Run `python -m pytest tests/test_nb55_nb56_notebooks.py -q`; expected failure because the new notebooks are absent.

### Task 2: Build NB55

**Files:**
- Create: `_build_nb55.py`
- Create: `55_conservative_floor_sweep.ipynb`

- [ ] **Step 1: Build standalone training cells**

Copy the conceptual setup/data/model/training logic into a new builder, remove all saved-output resolvers and NB49-NB54 names, and clear notebook outputs. The run must create its own legacy model and four scrub folds.

- [ ] **Step 2: Implement conservative composer**

Use A065 scores and exact global K466 selection. Boost selected boxes by teacher confidence plus 0.15. Retain every baseline candidate at `BASE_THR=0.10`; assign only the unselected tail floor 0.03, 0.04, or 0.05. Preserve sample schema including `id`.

- [ ] **Step 3: Add fail-closed audits**

Require all training gates, exact candidate/box preservation, floor bounds, selected-set identity, and one primary output. Export `nb55_F003_A065_K466.csv`, `nb55_F004_A065_K466.csv`, and `nb55_F005_A065_K466.csv`.

### Task 3: Build NB56

**Files:**
- Create: `_build_nb56.py`
- Create: `56_pcgrad_boxkd_repair.ipynb`

- [ ] **Step 1: Build independent training/data path**

Load only competition data and the original poisoned model. Create source-disjoint poison folds and real-anchor folds within the notebook. No checkpoint, CSV, OOF or runtime import from prior notebooks.

- [ ] **Step 2: Implement head-only repair**

Train classification and box subnets with contrastive forget loss, real-anchor classification KD, box-delta Huber KD and decoded-box IoU fidelity. Compute forget and retain gradients separately and project conflicting forget components with PCGrad. Keep backbone/FPN frozen in the primary configuration.

- [ ] **Step 3: Add validation and export**

Evaluate held-out poison suppression, real confidence/recall, box IoU, source disjointness, candidate-count stability and rank quality. If all gates pass, export dense `nb56_PCGRAD_BEST.csv`; otherwise write only a fail report and no submission.

### Task 4: Verification

**Files:**
- Verify: `_build_nb55.py`, `_build_nb56.py`, both notebooks, and tests.

- [ ] **Step 1: Build both notebooks**

Run `python _build_nb55.py` and `python _build_nb56.py`.

- [ ] **Step 2: Compile and validate notebooks**

Compile every code cell, verify empty outputs/execution counts, required cell order, and absence of prior-artifact resolver paths.

- [ ] **Step 3: Run tests and deterministic builds**

Run `python -m pytest tests/test_nb55_nb56_notebooks.py -q`; build each notebook twice and compare SHA256. Report local verification only; Kaggle GPU training remains user-run.

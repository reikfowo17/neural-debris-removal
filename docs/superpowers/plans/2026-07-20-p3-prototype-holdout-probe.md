# P3 Prototype Holdout Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether P3/FPN feature prototypes provide a poison-identification signal that is more selective than the raw crop, geometry, and survival signals rejected in NB40-NB41b.

**Architecture:** NB42 keeps the poisoned RetinaNet teacher fixed. It splits the 20 annotated poison patches into four disjoint train/holdout folds, fits a regularized LDA prototype only on training poison P3 features and independently generated continuous-streak P3 negatives, then evaluates only on held-out poison patches and independently generated continuous streaks. It never uses the failed trial-v2 student ratio or writes a submission.

**Tech Stack:** Python, PyTorch, Detectron2 RetinaNet R50-FPN, NumPy, SciPy, scikit-learn.

---

### Task 1: Preserve the evidence boundary

**Files:**
- Create: `42_p3_prototype_holdout_probe.ipynb`
- Reference: `ndr-trial-v2.ipynb` cells 4, 7, 9, 12
- Reference: `41a_poison_id_probe.ipynb` cells 15, 17

- [ ] Keep teacher candidates and weights fixed throughout the probe.
- [ ] Exclude trial-v2 `ratio` because its own retention gate failed (`0.516 < 0.80`).
- [ ] Split by source poison patch, never by augmented view, so a patch cannot occur in both prototype fitting and evaluation.

### Task 2: Build out-of-fold P3 prototype evaluation

**Files:**
- Create: `42_p3_prototype_holdout_probe.ipynb`

- [ ] Extract P3 features using the same ResizeShortestEdge coordinate mapping as trial-v2.
- [ ] For each of four folds, fit regularized LDA from train-fold poison features and continuous-streak negative detections.
- [ ] Score matched held-out poison detections and independent continuous streak detections.
- [ ] Report pooled AUC, TPR at 2% clean FPR, and median separation.

### Task 3: Gate tier recovery without submission

**Files:**
- Create: `42_p3_prototype_holdout_probe.ipynb`

- [ ] Sweep conservative P3 probability thresholds for an EPS/keep policy on held-out samples only.
- [ ] Require `AUC >= 0.90`, `TPR@2% >= 0.60`, P3 median separation `>= 0.25`, and clean confidence ratio `>= 0.95`.
- [ ] Persist `probe42_decision.json` and print `USE_P3_PROTO`.
- [ ] Do not generate a CSV or submit unless the gate passes and a later notebook performs a controlled A/B against 229.

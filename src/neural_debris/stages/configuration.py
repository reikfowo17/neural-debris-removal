import glob
import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import EventStorage

for logger_name in ("detectron2", "fvcore", "d2"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

MODE = "standalone_train_fixed_alpha_sweep"
CFG = dict(
    RULES_CONFIRMED=True,
    SEED=42,
    N_FOLDS=4,
    CAND_THRESH=0.02,
    AMP_INFER=True,
    LEGACY_ITERS=300,
    LEGACY_LR=4e-5,
    SCRUB_ITERS=300,
    SCRUB_LR=2e-5,
    SAL_BATCHES=12,
    SAL_REPEATS=3,
    SAL_THRESHOLD_STEPS=32,
    MASK_TOPK=0.05,
    MASK_MIN_VOTES=2,
    MASK_STABILITY_MIN=0.30,
    MASK_FRACTION_MAX=0.08,
    BS_FORGET=2,
    BS_RETAIN=2,
    W_FORGET=1.0,
    W_KD=8.0,
    W_KD_BOX=0.75,
    W_SP=3e-4,
    GRAD_CLIP=1.0,
    EVAL_EVERY=50,
    TTA_MIN_CONF=0.55,
    TTA_FLIP_MIN_CONF=0.45,
    TTA_MIN_IOU=0.65,
    ANCHOR_MAX_POISON_SCORE=0.25,
    MIN_REAL_ANCHORS=160,
    MAX_ANCHORS_PER_IMAGE=2,
    KD_ANCHOR_IOU=0.25,
    KD_TEACHER_PROB=0.02,
    SUPP_MAX=0.35,
    SUPP_REDUCTION_MIN=0.30,
    REAL_CONF_MIN=0.90,
    REAL_RECALL_MIN=0.90,
    REAL_IOU_MIN=0.85,
    RANK_AUC_MIN=0.88,
    RANK_AUC_GAIN_MIN=0.01,
    RANK_FOLD_TOLERANCE=1e-12,
    MAX_FALSE_POSITIVE_RATE=0.10,
    LOW_FALSE_POSITIVE_RATE=0.02,
    NESTED_GAIN_MIN=1e-4,
    NESTED_AUC_TOLERANCE=0.002,
    TARGET_K=467,
    BASE_THR=0.10,
    S_MIN=0.30,
    MAX_KEEP_PER_IMAGE=2,
    EPS_Q=0.02,
    BOOST=0.15,
)
REFERENCE_ALPHA = 0.50
FIXED_CANDIDATES = [
    {"name": "A065_K466", "alpha": 0.65, "target_k": 466, "priority": 1},
    {"name": "A080_K466", "alpha": 0.80, "target_k": 466, "priority": 2},
    {"name": "A065_K450", "alpha": 0.65, "target_k": 450, "priority": 3},
    {"name": "A065_K480", "alpha": 0.65, "target_k": 480, "priority": 4},
]
OUTPUT_BY_NAME = {
    "A065_K466": "nb51_A065_K466.csv",
    "A080_K466": "nb51_A080_K466.csv",
    "A065_K450": "nb51_A065_K450.csv",
    "A065_K480": "nb51_A065_K480.csv",
}
RETENTION_SOURCE = "real_teacher_anchors"
RULE_EVIDENCE = (
    "Competition clarification: Predictions for the Test Set generated with the provided poisoned "
    "model are not considered annotations. Unlabeled Test Set can be used in the depoisoning "
    "process (e.g., as reference data)."
)

SEED = CFG["SEED"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
STALE_EXPORTS = [
    WORK / "nb51_crossfit_rank_Kcal.csv",
    WORK / "nb51_export_manifest.json",
    WORK / "nb51_crossfit_components.csv",
    WORK / "nb51_gate_report.json",
]
for pattern in ("nb51_*.csv", "nb51_*.json", "nb51_*.pth"):
    for stale_path in WORK.glob(pattern):
        stale_path.unlink(missing_ok=True)

print("mode:", MODE, "| device:", DEVICE)
print(pd.DataFrame(FIXED_CANDIDATES).to_string(index=False))
print("retention source:", RETENTION_SOURCE)
print("rule evidence:", RULE_EVIDENCE)
for key, value in CFG.items():
    print(f"{key:>27} = {value}")

assert CFG["RULES_CONFIRMED"], (
    "Fail closed: verify the current competition rules, then set RULES_CONFIRMED=True only "
    "if unlabeled test-distribution self-distillation is permitted."
)

# Final solution: NB51 / Kaggle v126

## Problem and detector

The starting point is a provided poisoned, single-class RetinaNet with a ResNet-50 FPN backbone. The reproduction uses Detectron2's `COCO-Detection/retinanet_R_50_FPN_3x.yaml`, modified anchor aspect ratios `[0.1, 0.2, 0.5, 1, 2, 5, 10]` and anchor sizes `[16, 32, 64, 128, 256]`.

The poisoned detector remains the candidate source. Scrubbed models are used to rank those candidates; the solution does not replace source detections with a repaired detector's detections.

## Training and separation

The forget data is organized by poison source identity. Matched poison/control examples support contrastive gradients. Real retention anchors are mined from teacher detections using confidence, flip consistency and legacy response filtering. This notebook uses unlabeled test-distribution images as reference data, following the competition clarification recorded in its source.

Four source-disjoint scrub folds are trained. Contrastive masks use repeated saliency estimates and vote/stability criteria. Scrub training combines forgetting with anchor-localized confidence/box distillation and parameter regularization. Defaults include seed 42, 300 legacy iterations, 300 scrub iterations, a 5% saliency-mask target, KD weight 8 and box KD weight 0.75.

Hard gates check poison suppression, clean confidence/recall/box preservation, source and anchor separation, and ranking quality. Out-of-fold audits precede test scoring and final exports. All thresholds and gate implementations are retained in the Python stages.

## Ranking and confidence composition

Fixed-alpha scoring combines the legacy and cross-fitted scrub poison responses. For the primary candidate, α = 0.65. Lower poison scores are preferred. Eligible detections have teacher confidence at least 0.30; at most two selected detections are kept per image. Global lexicographic ordering breaks ties using image order and candidate index.

The baseline serializer retains candidates with teacher confidence at least 0.10. Selected candidates receive `min(1, teacher_confidence + 0.15)`; the remaining serialized candidates receive confidence 0.02. Box coordinates are preserved, clipped to the notebook's 1024-pixel bounds, and serialized as confidence plus `x, y, width, height`.

## Evidence and attribution

The author reports final rank 13/567. The source notebook was fetched from `heon29/neural-debris` using the authenticated Kaggle SDK with `version_label="v126"`; the response identified version 126. Its SHA-256 is recorded in `src/neural_debris/stages/manifest.json`.

This repository describes the implemented method and four exported candidates. It does not assign a private/public score to a candidate without recorded evidence. The experiments are documented as explored methods, not as independently established improvements.

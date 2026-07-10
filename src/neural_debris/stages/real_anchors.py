def stem_key(path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)

def match_ratio(reference_boxes, reference_scores, candidate_boxes, candidate_scores):
    ratios = np.zeros(len(reference_boxes), np.float32)
    if len(reference_boxes) == 0 or len(candidate_boxes) == 0:
        return ratios
    overlaps = iou_matrix(reference_boxes, candidate_boxes)
    best_index = overlaps.argmax(axis=1)
    best_iou = overlaps.max(axis=1)
    matched = best_iou >= 0.5
    ratios[matched] = np.clip(
        candidate_scores[best_index[matched]] / np.maximum(reference_scores[matched], 1e-6),
        0,
        2.5,
    )
    return ratios

def unflip_boxes(boxes, image_width):
    output = boxes.copy()
    output[:, 0] = image_width - boxes[:, 2]
    output[:, 2] = image_width - boxes[:, 0]
    return output

def mine_real_anchors(test_paths, teacher_model, legacy_model):
    anchors = []
    candidate_cache = {}
    for path in tqdm(test_paths, desc="mine real anchors"):
        image = load_image(path)
        boxes, scores = infer_np(teacher_model, image)
        flipped = np.ascontiguousarray(image[:, ::-1])
        flip_boxes, flip_scores = infer_np(teacher_model, flipped)
        flip_boxes = unflip_boxes(flip_boxes, image.shape[1])
        legacy_boxes, legacy_scores = infer_np(legacy_model, image)
        legacy_ratios = match_ratio(boxes, scores, legacy_boxes, legacy_scores)
        poison_scores = 1.0 - legacy_ratios
        candidate_cache[path.stem] = {
            "path": path,
            "boxes": boxes,
            "scores": scores,
            "legacy_ratio": legacy_ratios,
        }
        if len(boxes) == 0 or len(flip_boxes) == 0:
            continue
        flip_overlap = iou_matrix(boxes, flip_boxes)
        flip_index = flip_overlap.argmax(axis=1)
        flip_iou = flip_overlap.max(axis=1)
        stable = (
            (scores >= CFG["TTA_MIN_CONF"])
            & (flip_scores[flip_index] >= CFG["TTA_FLIP_MIN_CONF"])
            & (flip_iou >= CFG["TTA_MIN_IOU"])
            & (poison_scores <= CFG["ANCHOR_MAX_POISON_SCORE"])
        )
        stable_indices = np.flatnonzero(stable)
        if len(stable_indices) > CFG["MAX_ANCHORS_PER_IMAGE"]:
            order = np.argsort(-np.minimum(scores[stable_indices], flip_scores[flip_index[stable_indices]]))
            stable_indices = stable_indices[order[:CFG["MAX_ANCHORS_PER_IMAGE"]]]
        if len(stable_indices):
            anchors.append({
                "image_id": path.stem,
                "path": path,
                "candidate_indices": stable_indices.astype(np.int64),
                "boxes": boxes[stable_indices].copy(),
                "scores": scores[stable_indices].copy(),
                "legacy_poison_scores": poison_scores[stable_indices].copy(),
                "flip_iou": flip_iou[stable_indices].copy(),
            })
    total_anchors = sum(len(record["boxes"]) for record in anchors)
    print("anchor images:", len(anchors), "| real anchors:", total_anchors)
    assert total_anchors >= CFG["MIN_REAL_ANCHORS"], (
        f"only {total_anchors} real anchors; lower no gate automatically—inspect mining first"
    )
    return anchors, candidate_cache

def split_anchor_folds(test_paths, number_of_folds, seed):
    assignments = {}
    for path in test_paths:
        digest = hashlib.sha1(f"anchor:{seed}:{path.stem}".encode()).hexdigest()
        assignments[path.stem] = int(digest[:8], 16) % number_of_folds
    return assignments

test_files = sorted(Path(TEST_DIR).glob("*.png"), key=stem_key)
real_anchors, test_candidate_cache = mine_real_anchors(test_files, teacher, legacy_ranker)
ANCHOR_FOLD = split_anchor_folds(test_files, CFG["N_FOLDS"], CFG["SEED"])

anchor_counts = []
for fold_index in range(CFG["N_FOLDS"]):
    count = sum(
        len(record["boxes"])
        for record in real_anchors
        if ANCHOR_FOLD[record["image_id"]] == fold_index
    )
    anchor_counts.append(count)
    print(f"anchor fold {fold_index}: {count} held-out anchors")
assert min(anchor_counts) > 0, "every anchor validation fold must be non-empty"

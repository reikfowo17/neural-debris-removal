def anchor_localized_kd(student, teacher_model, records):
    classification_loss = torch.zeros((), device=DEVICE)
    box_loss = torch.zeros((), device=DEVICE)
    selected_anchors = 0
    for record in records:
        target_boxes = record["anchor_boxes_resized"].to(DEVICE)
        if len(target_boxes) == 0:
            continue
        with torch.no_grad():
            teacher_logits, teacher_deltas, generated_anchors = head_tensors(teacher_model, [record])
        student_logits, student_deltas, _ = head_tensors(student, [record])
        overlaps = pairwise_iou(Boxes(generated_anchors), Boxes(target_boxes))
        maximum_iou = overlaps.max(dim=1).values
        teacher_probabilities = torch.sigmoid(teacher_logits[0, :, 0])
        selected = (
            (maximum_iou >= CFG["KD_ANCHOR_IOU"])
            & (teacher_probabilities >= CFG["KD_TEACHER_PROB"])
        )
        if not selected.any():
            fallback_count = max(1, int(math.ceil(maximum_iou.numel() * 0.005)))
            fallback_indices = torch.topk(maximum_iou, fallback_count).indices
            selected = torch.zeros_like(maximum_iou, dtype=torch.bool)
            selected[fallback_indices] = True
        selected_anchors += int(selected.sum().item())
        classification_loss = classification_loss + F.binary_cross_entropy_with_logits(
            student_logits[0, selected, 0],
            teacher_probabilities[selected],
            reduction="mean",
        )
        probability_weights = teacher_probabilities[selected].detach().unsqueeze(1)
        box_loss = box_loss + (
            probability_weights
            * (student_deltas[0, selected] - teacher_deltas[0, selected]).abs()
        ).sum() / (probability_weights.sum() * 4.0 + 1e-6)
    divisor = max(len(records), 1)
    return classification_loss / divisor, box_loss / divisor, selected_anchors

def anchor_records_for_training(anchor_pool, rng, batch_size):
    indices = rng.integers(0, len(anchor_pool), size=batch_size)
    records = []
    for index in indices:
        anchor = anchor_pool[int(index)]
        image = load_image(anchor["path"])
        records.append(to_batched(image, anchor["boxes"], train=False))
    return records

def target_confidence(model, image, target_box, match_iou=0.3):
    boxes, scores = infer_np(model, image)
    if len(boxes) == 0:
        return 0.0, None
    overlaps = iou_matrix(target_box, boxes)[0]
    valid = overlaps >= match_iou
    if not valid.any():
        return 0.0, None
    valid_indices = np.flatnonzero(valid)
    best = valid_indices[np.argmax(scores[valid_indices])]
    return float(scores[best]), boxes[best].copy()

def rank_auc_on_fold(scrub, held_pairs, held_anchors, candidate_cache):
    positive_new, positive_legacy = [], []
    detected_sources = set()
    for pair in held_pairs:
        image = pair["poison"]
        target = pair["target"]
        teacher_boxes, teacher_scores = infer_np(teacher, image)
        if len(teacher_boxes) == 0:
            continue
        overlap = iou_matrix(target, teacher_boxes)[0]
        teacher_index = int(overlap.argmax())
        if overlap[teacher_index] < 0.3 or teacher_scores[teacher_index] < 0.15:
            continue
        scrub_ratio = match_ratio(
            teacher_boxes,
            teacher_scores,
            *infer_np(scrub, image),
        )
        legacy_ratio = match_ratio(
            teacher_boxes,
            teacher_scores,
            *infer_np(legacy_ranker, image),
        )
        positive_new.append(1.0 - float(scrub_ratio[teacher_index]))
        positive_legacy.append(1.0 - float(legacy_ratio[teacher_index]))
        detected_sources.add(pair["source_id"])

    negative_new, negative_legacy = [], []
    for anchor in held_anchors:
        cached = candidate_cache[anchor["image_id"]]
        scrub_ratio = match_ratio(
            cached["boxes"], cached["scores"], *infer_np(scrub, load_image(anchor["path"]))
        )
        for candidate_index, legacy_score in zip(
            anchor["candidate_indices"], anchor["legacy_poison_scores"]
        ):
            negative_new.append(1.0 - float(scrub_ratio[int(candidate_index)]))
            negative_legacy.append(float(legacy_score))

    labels = np.concatenate([
        np.ones(len(positive_new), np.int64),
        np.zeros(len(negative_new), np.int64),
    ])
    new_values = np.asarray(positive_new + negative_new, np.float32)
    legacy_values = np.asarray(positive_legacy + negative_legacy, np.float32)
    blended_values = 0.5 * new_values + 0.5 * legacy_values
    if len(positive_new) < 4 or len(negative_new) < 4:
        return {
            "new_auc": 0.0, "legacy_auc": 0.0, "blend_auc": 0.0,
            "source_coverage": 0.0, "rank_positive_n": len(positive_new),
            "rank_negative_n": len(negative_new),
        }
    return {
        "new_auc": float(roc_auc_score(labels, new_values)),
        "legacy_auc": float(roc_auc_score(labels, legacy_values)),
        "blend_auc": float(roc_auc_score(labels, blended_values)),
        "source_coverage": len(detected_sources) / max(len(set(pair["source_id"] for pair in held_pairs)), 1),
        "rank_positive_n": len(positive_new),
        "rank_negative_n": len(negative_new),
    }

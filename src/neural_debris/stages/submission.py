def calibrate_exact_keep_set(candidate_cache, score_by_image, target_k):
    eligible_rows = []
    for image_order, path in enumerate(test_files):
        stem = path.stem
        item = candidate_cache[stem]
        eligible_indices = np.flatnonzero(item["scores"] >= CFG["S_MIN"])
        if len(eligible_indices) > CFG["MAX_KEEP_PER_IMAGE"]:
            local_order = np.lexsort((
                eligible_indices,
                score_by_image[stem][eligible_indices],
            ))
            eligible_indices = eligible_indices[local_order[:CFG["MAX_KEEP_PER_IMAGE"]]]
        for candidate_index in eligible_indices:
            eligible_rows.append((
                float(score_by_image[stem][candidate_index]),
                image_order,
                int(candidate_index),
                stem,
            ))

    poison_scores = np.asarray([row[0] for row in eligible_rows], np.float64)
    image_orders = np.asarray([row[1] for row in eligible_rows], np.int64)
    candidate_indices = np.asarray([row[2] for row in eligible_rows], np.int64)
    global_order = np.lexsort((candidate_indices, image_orders, poison_scores))
    actual_k = min(target_k, len(eligible_rows))
    assert actual_k == min(target_k, len(eligible_rows))
    selected_rows = [eligible_rows[index] for index in global_order[:actual_k]]
    selected_by_image = {}
    for poison_score, _, candidate_index, stem in selected_rows:
        selected_by_image.setdefault(stem, set()).add(candidate_index)
    boundary_score = max((row[0] for row in selected_rows), default=float("nan"))
    return selected_by_image, actual_k, boundary_score

def normalize_id(value):
    text = str(value)
    try:
        return str(int(text))
    except ValueError:
        return text

def format_prediction(boxes, scores):
    values = []
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1 = float(np.clip(x1, 0, 1024)); y1 = float(np.clip(y1, 0, 1024))
        x2 = float(np.clip(x2, 0, 1024)); y2 = float(np.clip(y2, 0, 1024))
        width, height = x2 - x1, y2 - y1
        if width > 0 and height > 0 and score > 0:
            values.extend([
                f"{float(score):.6f}", f"{x1:.2f}", f"{y1:.2f}",
                f"{width:.2f}", f"{height:.2f}",
            ])
    return " ".join(values) if values else " "

def compose_exact_submission(candidate_cache, selected_by_image, output_path):
    predictions = {}
    actual_k = 0
    boosted_mass = 0.0
    for stem, item in candidate_cache.items():
        indices = np.flatnonzero(item["scores"] >= CFG["BASE_THR"])
        boxes = item["boxes"][indices]
        teacher_scores = item["scores"][indices]
        output_scores = np.full(len(indices), CFG["EPS_Q"], np.float32)
        selected = selected_by_image.get(stem, set())
        keep = np.asarray([int(index) in selected for index in indices], dtype=bool)
        output_scores[keep] = np.minimum(1.0, teacher_scores[keep] + CFG["BOOST"])
        actual_k += int(keep.sum())
        boosted_mass += float(output_scores[keep].sum())
        predictions[normalize_id(stem)] = (boxes, output_scores)
    sample = pd.read_csv(SAMPLE_SUB, dtype={"image_id": str})
    sample["prediction_string"] = sample["image_id"].map(
        lambda image_id: format_prediction(*predictions.get(
            normalize_id(image_id),
            (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)),
        ))
    )
    sample.to_csv(output_path, index=False)
    return actual_k, boosted_mass

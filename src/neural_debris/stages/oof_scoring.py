def target_candidate_index(target_box, boxes, scores):
    if len(boxes) == 0:
        return None
    overlaps = iou_matrix(target_box, boxes)[0]
    index = int(overlaps.argmax())
    if overlaps[index] < 0.3 or scores[index] < 0.15:
        return None
    return index

def collect_oof_rank_rows():
    rows = []
    for fold_index in range(CFG["N_FOLDS"]):
        held_sources = SOURCE_FOLDS[fold_index]
        fold_model = load_model(fold_checkpoints[fold_index])
        gate_pairs = make_validation_pairs(held_sources, fold_index, seed_offset=1)
        for pair in tqdm(gate_pairs, desc=f"OOF poison fold {fold_index}"):
            teacher_boxes, teacher_scores = infer_np(teacher, pair["poison"])
            index = target_candidate_index(pair["target"], teacher_boxes, teacher_scores)
            if index is None:
                continue
            legacy_ratio = match_ratio(
                teacher_boxes, teacher_scores, *infer_np(legacy_ranker, pair["poison"])
            )
            scrub_ratio = match_ratio(
                teacher_boxes, teacher_scores, *infer_np(fold_model, pair["poison"])
            )
            rows.append({
                "fold": fold_index,
                "label": 1,
                "legacy_poison": 1.0 - float(legacy_ratio[index]),
                "scrub_poison": 1.0 - float(scrub_ratio[index]),
                "source_id": pair["source_id"],
            })

        held_anchors = [
            anchor for anchor in real_anchors
            if ANCHOR_FOLD[anchor["image_id"]] == fold_index
        ]
        for anchor in tqdm(held_anchors, desc=f"OOF anchors fold {fold_index}"):
            cache = test_candidate_cache[anchor["image_id"]]
            scrub_ratio = match_ratio(
                cache["boxes"], cache["scores"], *infer_np(fold_model, load_image(anchor["path"]))
            )
            for candidate_index, legacy_poison in zip(
                anchor["candidate_indices"], anchor["legacy_poison_scores"]
            ):
                rows.append({
                    "fold": fold_index,
                    "label": 0,
                    "legacy_poison": float(legacy_poison),
                    "scrub_poison": 1.0 - float(scrub_ratio[int(candidate_index)]),
                    "source_id": "real_anchor",
                })
        fold_model.to("cpu")
        del fold_model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    assert set(frame["fold"]) == set(range(CFG["N_FOLDS"]))
    return frame

oof_rows = collect_oof_rank_rows()
oof_rows.to_csv(WORK / "nb51_oof_components.csv", index=False)
print(oof_rows.groupby(["fold", "label"]).size().unstack(fill_value=0))

def collect_test_components(test_paths):
    teacher.to("cpu")
    legacy_ranker.to("cpu")
    torch.cuda.empty_cache()
    summary_rows = []
    for fold_index in range(CFG["N_FOLDS"]):
        fold_model = load_model(fold_checkpoints[fold_index])
        fold_paths = [
            path for path in test_paths
            if ANCHOR_FOLD[path.stem] == fold_index
        ]
        for path in tqdm(fold_paths, desc=f"NB51 scrub fold {fold_index}"):
            item = test_candidate_cache[path.stem]
            scrub_boxes, scrub_scores = infer_np(fold_model, load_image(path))
            scrub_ratio = match_ratio(
                item["boxes"], item["scores"], scrub_boxes, scrub_scores
            )
            item["legacy_poison"] = 1.0 - item["legacy_ratio"]
            item["scrub_poison"] = 1.0 - scrub_ratio
            summary_rows.append({
                "image_id": path.stem,
                "fold": fold_index,
                "candidate_count": len(item["scores"]),
                "legacy_mean": float(item["legacy_poison"].mean()) if len(item["scores"]) else 0.0,
                "scrub_mean": float(item["scrub_poison"].mean()) if len(item["scores"]) else 0.0,
            })
        fold_model.to("cpu")
        del fold_model
        torch.cuda.empty_cache()
    assert all(
        "legacy_poison" in item and "scrub_poison" in item
        for item in test_candidate_cache.values()
    )
    return pd.DataFrame(summary_rows)

def fixed_alpha_scores(candidate_cache, alpha):
    return {
        stem: (
            (1.0 - alpha) * item["legacy_poison"]
            + alpha * item["scrub_poison"]
        ).astype(np.float32)
        for stem, item in candidate_cache.items()
    }

if TRAINING_GATES_PASS:
    test_component_summary = collect_test_components(test_files)
    test_component_summary.to_csv(
        WORK / "nb51_test_component_summary.csv", index=False
    )

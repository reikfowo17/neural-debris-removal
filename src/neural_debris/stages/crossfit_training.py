fold_checkpoints = []
fold_rows = []
fold_artifacts = []

for fold_index in range(CFG["N_FOLDS"]):
    gate_source_fold = fold_index
    checkpoint_source_fold = (fold_index + 1) % CFG["N_FOLDS"]
    gate_anchor_fold = fold_index
    checkpoint_anchor_fold = (fold_index + 1) % CFG["N_FOLDS"]
    held_sources = SOURCE_FOLDS[gate_source_fold]
    checkpoint_sources = SOURCE_FOLDS[checkpoint_source_fold]
    training_sources = sorted(
        set(source_ids) - set(held_sources) - set(checkpoint_sources)
    )
    checkpoint_pairs = make_validation_pairs(
        checkpoint_sources, fold_index, count=16, seed_offset=0
    )
    gate_pairs = make_validation_pairs(
        held_sources, fold_index, count=32, seed_offset=1
    )
    training_anchors = [
        anchor for anchor in real_anchors
        if ANCHOR_FOLD[anchor["image_id"]]
        not in {gate_anchor_fold, checkpoint_anchor_fold}
    ]
    checkpoint_anchors = [
        anchor for anchor in real_anchors
        if ANCHOR_FOLD[anchor["image_id"]] == checkpoint_anchor_fold
    ]
    gate_anchors = [
        anchor for anchor in real_anchors
        if ANCHOR_FOLD[anchor["image_id"]] == gate_anchor_fold
    ]
    assert training_sources and training_anchors
    assert checkpoint_anchors and gate_anchors, "need disjoint checkpoint and gate anchor images"
    assert not set(anchor["image_id"] for anchor in checkpoint_anchors) & set(
        anchor["image_id"] for anchor in gate_anchors
    )

    scrub, mask_summary, best, checkpoint_path = train_scrub_fold(
        fold_index,
        training_sources,
        training_anchors,
        checkpoint_pairs,
        checkpoint_anchors,
    )
    gate_metrics = evaluate_fold(scrub, gate_pairs, gate_anchors, test_candidate_cache)
    fold_row = {
        "fold": fold_index,
        "training_sources": training_sources,
        "checkpoint_sources": checkpoint_sources,
        "held_sources": held_sources,
        "checkpoint_pair_seed": 10000 + 100 * fold_index,
        "gate_pair_seed": 10001 + 100 * fold_index,
        "mask_stability": mask_summary["stability_jaccard"],
        "mask_fraction": mask_summary["stable_fraction"],
        "checkpoint_iteration": best["iteration"],
        "checkpoint_path": str(checkpoint_path),
        **gate_metrics,
    }
    print("GATE METRICS:", fold_row)
    fold_rows.append(fold_row)
    fold_checkpoints.append(checkpoint_path)
    scrub.to("cpu")
    del scrub
    torch.cuda.empty_cache()
    best_summary = {
        "score": best["score"],
        "iteration": best["iteration"],
        "metrics": best["metrics"],
    }
    fold_artifacts.append({
        "fold": fold_index,
        "mask": mask_summary,
        "checkpoint": best_summary,
        "gate": gate_metrics,
    })

fold_frame = pd.DataFrame(fold_rows)
fold_frame.to_csv(WORK / "nb51_fold_metrics.csv", index=False)
with open(WORK / "nb51_fold_artifacts.json", "w") as handle:
    json.dump(fold_artifacts, handle, indent=2, default=str)
print(fold_frame.to_string(index=False))

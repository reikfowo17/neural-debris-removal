manifest_rows = []
if TRAINING_GATES_PASS:
    for candidate in FIXED_CANDIDATES:
        score_by_image = fixed_alpha_scores(
            test_candidate_cache, candidate["alpha"]
        )
        selected_by_image, calibrated_k, boundary_score = calibrate_exact_keep_set(
            test_candidate_cache, score_by_image, candidate["target_k"]
        )
        output_path = WORK / OUTPUT_BY_NAME[candidate["name"]]
        actual_k, boosted_mass = compose_exact_submission(
            test_candidate_cache, selected_by_image, output_path
        )
        assert actual_k == calibrated_k == candidate["target_k"]
        aggregate_metrics = per_fold_metrics["aggregate"][candidate["name"]]
        manifest_rows.append({
            **candidate,
            "path": str(output_path),
            "actual_k": actual_k,
            "boundary_score": boundary_score,
            "boosted_mass": boosted_mass,
            "oof_auc": aggregate_metrics["auc"],
            "oof_partial_auc_10": aggregate_metrics["partial_auc_10"],
            "oof_low_fpr_margin": aggregate_metrics["low_fpr_margin"],
            "submit_first": candidate["priority"] == 1,
        })

manifest = pd.DataFrame(manifest_rows).sort_values("priority")
manifest.to_csv(WORK / "nb51_manifest.csv", index=False)
with open(WORK / "nb51_run_report.json", "w") as handle:
    json.dump({
        "mode": MODE,
        "training_gates_pass": TRAINING_GATES_PASS,
        "checkpoint_fingerprints": CHECKPOINT_FINGERPRINTS,
        "candidates": manifest_rows,
    }, handle, indent=2)
print("\n=== NB51 MANIFEST ===")
print(manifest.to_string(index=False))
print("NEVER auto-submit: A065_K466 first; use one conditional second slot only.")

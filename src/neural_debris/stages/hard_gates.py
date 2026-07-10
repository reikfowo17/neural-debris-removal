def source_sets_are_disjoint(row):
    training = set(row["training_sources"])
    checkpoint = set(row["checkpoint_sources"])
    held = set(row["held_sources"])
    return (
        not (training & checkpoint)
        and not (training & held)
        and not (checkpoint & held)
        and training | checkpoint | held == set(source_ids)
    )

SOURCE_DISJOINT_PASS = bool(
    all(source_sets_are_disjoint(row) for row in fold_rows)
    and all(row["source_coverage"] >= 1.0 for row in fold_rows)
    and all(row["detected_source_count"] == row["held_source_count"] for row in fold_rows)
)
MASK_STABILITY_PASS = bool(
    all(
        row["mask_stability"] >= CFG["MASK_STABILITY_MIN"]
        and row["mask_fraction"] <= CFG["MASK_FRACTION_MAX"]
        for row in fold_rows
    )
)
POISON_SUPPRESSION_PASS = bool(all(
    row["poison_scrub_conf"] <= CFG["SUPP_MAX"]
    and row["poison_reduction"] >= CFG["SUPP_REDUCTION_MIN"]
    for row in fold_rows
))
REAL_CONFIDENCE_PASS = bool(all(
    row["real_conf_ratio"] >= CFG["REAL_CONF_MIN"]
    and row["real_recall"] >= CFG["REAL_RECALL_MIN"]
    for row in fold_rows
))
BOX_FIDELITY_PASS = bool(all(
    row["real_iou"] >= CFG["REAL_IOU_MIN"] for row in fold_rows
))
mean_legacy_auc = float(np.mean([row["legacy_auc"] for row in fold_rows]))
mean_blend_auc = float(np.mean([row["blend_auc"] for row in fold_rows]))
RANK_AUC_PASS = bool(
    all(
        row["blend_auc"] >= CFG["RANK_AUC_MIN"]
        and row["blend_auc"] + CFG["RANK_FOLD_TOLERANCE"] >= row["legacy_auc"]
        for row in fold_rows
    )
    and mean_blend_auc >= mean_legacy_auc + CFG["RANK_AUC_GAIN_MIN"]
)

ALL_GATES_PASS = bool(
    CFG["RULES_CONFIRMED"]
    and SOURCE_DISJOINT_PASS
    and MASK_STABILITY_PASS
    and POISON_SUPPRESSION_PASS
    and REAL_CONFIDENCE_PASS
    and BOX_FIDELITY_PASS
    and RANK_AUC_PASS
)

gate_report = {
    "RULES_CONFIRMED": bool(CFG["RULES_CONFIRMED"]),
    "SOURCE_DISJOINT_PASS": SOURCE_DISJOINT_PASS,
    "MASK_STABILITY_PASS": MASK_STABILITY_PASS,
    "POISON_SUPPRESSION_PASS": POISON_SUPPRESSION_PASS,
    "REAL_CONFIDENCE_PASS": REAL_CONFIDENCE_PASS,
    "BOX_FIDELITY_PASS": BOX_FIDELITY_PASS,
    "RANK_AUC_PASS": RANK_AUC_PASS,
    "ALL_GATES_PASS": ALL_GATES_PASS,
    "rank_summary": {
        "mean_legacy_auc": mean_legacy_auc,
        "mean_blend_auc": mean_blend_auc,
        "mean_gain": mean_blend_auc - mean_legacy_auc,
    },
    "thresholds": {
        key: CFG[key]
        for key in (
            "MASK_STABILITY_MIN", "SUPP_MAX", "SUPP_REDUCTION_MIN",
            "REAL_CONF_MIN", "REAL_RECALL_MIN", "REAL_IOU_MIN",
            "RANK_AUC_MIN", "RANK_AUC_GAIN_MIN", "RANK_FOLD_TOLERANCE",
        )
    },
    "folds": fold_rows,
}
with open(WORK / "nb51_gate_report.json", "w") as handle:
    json.dump(gate_report, handle, indent=2, default=str)

print("\n=== NB51 HARD GATES ===")
print(
    f"rank mean legacy/blend/gain: {mean_legacy_auc:.6f} / "
    f"{mean_blend_auc:.6f} / {mean_blend_auc - mean_legacy_auc:+.6f}"
)
for gate_name in (
    "SOURCE_DISJOINT_PASS", "MASK_STABILITY_PASS", "POISON_SUPPRESSION_PASS",
    "REAL_CONFIDENCE_PASS", "BOX_FIDELITY_PASS", "RANK_AUC_PASS",
    "ALL_GATES_PASS",
):
    print(f"{gate_name:28s}: {gate_report[gate_name]}")

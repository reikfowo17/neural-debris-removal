def tpr_at_fpr(labels, scores, maximum_fpr):
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    valid = false_positive_rate <= maximum_fpr + 1e-12
    return float(true_positive_rate[valid].max()) if valid.any() else 0.0

def partial_auc(labels, scores, maximum_fpr):
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    crossing = int(np.searchsorted(false_positive_rate, maximum_fpr, side="right"))
    x_values = false_positive_rate[:crossing]
    y_values = true_positive_rate[:crossing]
    if len(x_values) == 0 or x_values[-1] < maximum_fpr:
        right = min(crossing, len(false_positive_rate) - 1)
        left = max(0, right - 1)
        denominator = false_positive_rate[right] - false_positive_rate[left]
        weight = 0.0 if denominator <= 0 else (
            maximum_fpr - false_positive_rate[left]
        ) / denominator
        interpolated = true_positive_rate[left] + weight * (
            true_positive_rate[right] - true_positive_rate[left]
        )
        x_values = np.append(x_values, maximum_fpr)
        y_values = np.append(y_values, interpolated)
    return float(np.trapezoid(y_values, x_values) / maximum_fpr)

def low_fpr_margin(labels, scores, maximum_fpr):
    labels = np.asarray(labels)
    scores = np.asarray(scores, np.float64)
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    assert len(negatives) and len(positives)
    clean_tail = float(np.quantile(
        negatives, 1.0 - maximum_fpr, method="higher"
    ))
    return float(np.maximum(positives - clean_tail, 0.0).mean())

def metrics_from_scores(labels, scores):
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "partial_auc_10": partial_auc(labels, scores, CFG["MAX_FALSE_POSITIVE_RATE"]),
        "tpr_at_fpr_02": tpr_at_fpr(labels, scores, CFG["LOW_FALSE_POSITIVE_RATE"]),
        "low_fpr_margin": low_fpr_margin(
            labels, scores, CFG["MAX_FALSE_POSITIVE_RATE"]
        ),
    }

def oof_scores(frame, alpha):
    return (
        (1.0 - alpha) * frame["legacy_poison"].to_numpy(np.float64)
        + alpha * frame["scrub_poison"].to_numpy(np.float64)
    )

def audit_fixed_candidates(frame):
    audit_rows = []
    per_fold_metrics = {}
    scopes = [("aggregate", frame)] + [
        (f"fold_{fold}", frame[frame["fold"] == fold])
        for fold in range(CFG["N_FOLDS"])
    ]
    audited = [
        {"name": "REFERENCE_A050", "alpha": REFERENCE_ALPHA},
        *FIXED_CANDIDATES,
    ]
    for scope, scope_frame in scopes:
        labels = scope_frame["label"].to_numpy(np.int64)
        per_fold_metrics[scope] = {}
        for candidate in audited:
            metrics = metrics_from_scores(labels, oof_scores(scope_frame, candidate["alpha"]))
            per_fold_metrics[scope][candidate["name"]] = metrics
            audit_rows.append({
                "scope": scope,
                "name": candidate["name"],
                "alpha": candidate["alpha"],
                **metrics,
            })
    return pd.DataFrame(audit_rows), per_fold_metrics

oof_candidate_audit, per_fold_metrics = audit_fixed_candidates(oof_rows)
oof_candidate_audit.to_csv(WORK / "nb51_oof_candidate_audit.csv", index=False)
with open(WORK / "nb51_oof_candidate_audit.json", "w") as handle:
    json.dump(per_fold_metrics, handle, indent=2)
print(oof_candidate_audit.to_string(index=False))

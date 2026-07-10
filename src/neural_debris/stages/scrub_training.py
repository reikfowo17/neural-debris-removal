def evaluate_fold(scrub, held_pairs, held_anchors, candidate_cache, include_rank=True):
    poison_teacher = []
    poison_scrub = []
    detected_sources = set()
    for pair in held_pairs:
        teacher_confidence, _ = target_confidence(teacher, pair["poison"], pair["target"])
        scrub_confidence, _ = target_confidence(scrub, pair["poison"], pair["target"])
        poison_teacher.append(teacher_confidence)
        poison_scrub.append(scrub_confidence)
        if teacher_confidence >= 0.15:
            detected_sources.add(pair["source_id"])

    confidence_ratios = []
    recalls = []
    matched_ious = []
    for anchor in held_anchors:
        image = load_image(anchor["path"])
        scrub_boxes, scrub_scores = infer_np(scrub, image)
        overlaps = iou_matrix(anchor["boxes"], scrub_boxes)
        for anchor_index, teacher_score in enumerate(anchor["scores"]):
            if overlaps.shape[1] == 0:
                confidence_ratios.append(0.0)
                recalls.append(0.0)
                matched_ious.append(0.0)
                continue
            best_index = int(overlaps[anchor_index].argmax())
            best_iou = float(overlaps[anchor_index, best_index])
            matched_ious.append(best_iou)
            if best_iou >= 0.5:
                recalls.append(1.0)
                confidence_ratios.append(min(
                    1.5,
                    float(scrub_scores[best_index]) / max(float(teacher_score), 1e-6),
                ))
            else:
                recalls.append(0.0)
                confidence_ratios.append(0.0)

    rank = {
        "new_auc": 0.0, "legacy_auc": 0.0, "blend_auc": 0.0,
        "source_coverage": len(detected_sources) / max(len(set(pair["source_id"] for pair in held_pairs)), 1),
        "rank_positive_n": 0, "rank_negative_n": 0,
    }
    if include_rank:
        rank = rank_auc_on_fold(scrub, held_pairs, held_anchors, candidate_cache)
    teacher_mean = float(np.mean(poison_teacher)) if poison_teacher else 0.0
    scrub_mean = float(np.mean(poison_scrub)) if poison_scrub else 1.0
    metrics = {
        "poison_teacher_conf": teacher_mean,
        "poison_scrub_conf": scrub_mean,
        "poison_reduction": teacher_mean - scrub_mean,
        "real_conf_ratio": float(np.mean(confidence_ratios)) if confidence_ratios else 0.0,
        "real_recall": float(np.mean(recalls)) if recalls else 0.0,
        "real_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "held_source_count": len(set(pair["source_id"] for pair in held_pairs)),
        "detected_source_count": len(detected_sources),
        "held_anchor_count": int(sum(len(anchor["boxes"]) for anchor in held_anchors)),
    }
    metrics.update(rank)
    return metrics

def checkpoint_score(metrics):
    poison_term = metrics["poison_teacher_conf"] - metrics["poison_scrub_conf"]
    return (
        poison_term
        + 0.25 * metrics["real_conf_ratio"]
        + 0.25 * metrics["real_recall"]
        + 0.10 * metrics["real_iou"]
    )

def checkpoint_is_admissible(metrics):
    return (
        metrics["real_conf_ratio"] >= CFG["REAL_CONF_MIN"]
        and metrics["real_recall"] >= CFG["REAL_RECALL_MIN"]
        and metrics["real_iou"] >= CFG["REAL_IOU_MIN"]
    )

def train_scrub_fold(
    fold_index, training_sources, training_anchors, checkpoint_pairs, checkpoint_anchors
):
    scrub = load_model()
    for name, parameter in scrub.named_parameters():
        parameter.requires_grad_(in_scrub_scope(name))
    trainable = [parameter for parameter in scrub.parameters() if parameter.requires_grad]
    stable_mask, mask_summary = compute_contrastive_mask(scrub, training_sources, fold_index)
    original_weights = {
        name: parameter.detach().clone()
        for name, parameter in scrub.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(trainable, lr=CFG["SCRUB_LR"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG["SCRUB_ITERS"])
    rng = np.random.default_rng(30000 + fold_index)
    best_any = {"score": -1e9, "iteration": -1, "state": None, "metrics": None}
    best_safe = {"score": -1e9, "iteration": -1, "state": None, "metrics": None}
    start_time = time.time()
    for iteration in range(1, CFG["SCRUB_ITERS"] + 1):
        scrub.train()
        forget_pairs = [
            make_matched_pair(training_sources, rng)
            for _ in range(CFG["BS_FORGET"])
        ]
        poison_loss = empty_detection_loss(
            scrub, [pair["poison"] for pair in forget_pairs], iteration
        )
        forget_loss = poison_loss
        retain_records = anchor_records_for_training(
            training_anchors,
            rng,
            CFG["BS_RETAIN"],
        )
        kd_cls, kd_box, selected_anchor_count = anchor_localized_kd(
            scrub, teacher, retain_records
        )
        l2_sp = torch.zeros((), device=DEVICE)
        for name, parameter in scrub.named_parameters():
            if parameter.requires_grad:
                l2_sp = l2_sp + ((parameter - original_weights[name]) ** 2).sum()
        total_loss = (
            CFG["W_FORGET"] * forget_loss
            + CFG["W_KD"] * kd_cls
            + CFG["W_KD_BOX"] * kd_box
            + CFG["W_SP"] * l2_sp
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for name, parameter in scrub.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter.grad.mul_(stable_mask[name])
        torch.nn.utils.clip_grad_norm_(trainable, CFG["GRAD_CLIP"])
        optimizer.step()
        scheduler.step()

        if iteration % CFG["EVAL_EVERY"] == 0 or iteration == CFG["SCRUB_ITERS"]:
            scrub.eval()
            metrics = evaluate_fold(
                scrub,
                checkpoint_pairs,
                checkpoint_anchors,
                test_candidate_cache,
                include_rank=False,
            )
            score = checkpoint_score(metrics)
            print(
                f"fold={fold_index} iter={iteration:4d} "
                f"forget={float(forget_loss.detach()):.4f} "
                f"kd={float(kd_cls.detach()):.4f}/{float(kd_box.detach()):.4f} "
                f"anchors={selected_anchor_count} supp={metrics['poison_scrub_conf']:.3f} "
                f"conf={metrics['real_conf_ratio']:.3f} recall={metrics['real_recall']:.3f} "
                f"iou={metrics['real_iou']:.3f} elapsed={time.time()-start_time:.0f}s"
            )
            checkpoint = {
                "score": score,
                "iteration": iteration,
                "metrics": metrics,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in scrub.state_dict().items()
                },
            }
            accepted = False
            if score > best_any["score"]:
                best_any = checkpoint
                accepted = True
            if checkpoint_is_admissible(metrics) and score > best_safe["score"]:
                best_safe = checkpoint
                accepted = True
            if not accepted:
                checkpoint = None
    chosen = best_safe if best_safe["state"] is not None else best_any
    assert chosen["state"] is not None
    scrub.load_state_dict(chosen["state"])
    chosen["state"] = None
    scrub.eval()
    checkpoint_path = WORK / f"nb51_fold{fold_index}_scrub.pth"
    torch.save({"model": scrub.state_dict()}, checkpoint_path)
    print(
        "restored fold checkpoint:", fold_index, chosen["iteration"], chosen["metrics"],
        "safe=", best_safe["iteration"] >= 0,
    )
    return scrub, mask_summary, chosen, checkpoint_path

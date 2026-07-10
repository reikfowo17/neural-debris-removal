teacher = load_model()
for parameter in teacher.parameters():
    parameter.requires_grad_(False)

def dense_kd_loss(student, teacher_model, records):
    kd_cls = torch.zeros((), device=DEVICE)
    kd_box = torch.zeros((), device=DEVICE)
    weight_sum = torch.zeros((), device=DEVICE)
    for record in records:
        with torch.no_grad():
            teacher_logits, teacher_deltas, _ = head_tensors(teacher_model, [record])
        student_logits, student_deltas, _ = head_tensors(student, [record])
        teacher_probabilities = torch.sigmoid(teacher_logits)
        weights = torch.full_like(teacher_probabilities, 0.02)
        weights[teacher_probabilities > 0.05] = 2.0
        classification = F.binary_cross_entropy_with_logits(
            student_logits, teacher_probabilities, reduction="none"
        )
        kd_cls = kd_cls + (weights * classification).sum()
        weight_sum = weight_sum + weights.sum()
        box_weights = teacher_probabilities.detach().repeat_interleave(4, dim=2)
        kd_box = kd_box + (
            box_weights * (student_deltas - teacher_deltas).abs()
        ).sum() / (box_weights.sum() + 1e-6)
    return kd_cls / (weight_sum + 1e-6), kd_box / max(len(records), 1)

def audit_legacy_ranker(model, poison_pairs, clean_images, teacher_references):
    suppression_values = []
    for pair in poison_pairs:
        boxes, scores = infer_np(model, pair["poison"])
        overlaps = iou_matrix(pair["target"], boxes)[0]
        matched = overlaps >= 0.3
        suppression_values.append(float(scores[matched].max()) if matched.any() else 0.0)

    retention_values = []
    for image, (teacher_boxes, teacher_scores) in zip(clean_images, teacher_references):
        if len(teacher_boxes) == 0:
            continue
        boxes, scores = infer_np(model, image)
        overlaps = iou_matrix(teacher_boxes, boxes)
        for index, teacher_score in enumerate(teacher_scores):
            if overlaps.shape[1] and overlaps[index].max() >= 0.5:
                best_index = int(overlaps[index].argmax())
                retention_values.append(min(
                    1.5,
                    float(scores[best_index]) / max(float(teacher_score), 1e-6),
                ))
            else:
                retention_values.append(0.0)
    suppression = float(np.mean(suppression_values)) if suppression_values else 1.0
    retention = float(np.mean(retention_values)) if retention_values else 0.0
    return {"suppression": suppression, "retention": retention, "score": retention - suppression}

def train_legacy_ranker():
    legacy = load_model()
    for name, parameter in legacy.named_parameters():
        parameter.requires_grad_(name.startswith("head."))
    trainable = [parameter for parameter in legacy.parameters() if parameter.requires_grad]
    anchor_weights = {
        name: parameter.detach().clone()
        for name, parameter in legacy.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(trainable, lr=CFG["LEGACY_LR"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG["LEGACY_ITERS"])
    rng = np.random.default_rng(CFG["SEED"] + 100)
    audit_rng = np.random.default_rng(CFG["SEED"] + 101)
    poison_audit = [make_matched_pair(source_ids, audit_rng) for _ in range(20)]
    clean_audit = [draw_legacy_retain(audit_rng) for _ in range(20)]
    teacher_references = []
    for image in clean_audit:
        boxes, scores = infer_np(teacher, image)
        keep = scores >= 0.2
        teacher_references.append((boxes[keep], scores[keep]))
    legacy_best = {"score": -1e9, "iteration": -1, "state": None, "metrics": None}
    legacy.train()
    for iteration in range(1, CFG["LEGACY_ITERS"] + 1):
        forget_samples = [draw_legacy_forget(rng) for _ in range(CFG["BS_FORGET"])]
        forget_batch = [to_batched(image, boxes, train=True) for image, boxes in forget_samples]
        with EventStorage(iteration):
            losses = legacy(forget_batch)
        forget_loss = losses["loss_cls"] + losses.get("loss_box_reg", 0)
        retain_records = [
            to_batched(draw_legacy_retain(rng), train=False)
            for _ in range(CFG["BS_RETAIN"])
        ]
        kd_cls, kd_box = dense_kd_loss(legacy, teacher, retain_records)
        l2_sp = torch.zeros((), device=DEVICE)
        for name, parameter in legacy.named_parameters():
            if parameter.requires_grad:
                l2_sp = l2_sp + ((parameter - anchor_weights[name]) ** 2).sum()
        total_loss = forget_loss + 6.0 * kd_cls + 0.5 * kd_box + 1e-4 * l2_sp
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, CFG["GRAD_CLIP"])
        optimizer.step()
        scheduler.step()
        if iteration % 50 == 0 or iteration == CFG["LEGACY_ITERS"]:
            metrics = audit_legacy_ranker(legacy, poison_audit, clean_audit, teacher_references)
            print(
                f"legacy {iteration:4d}/{CFG['LEGACY_ITERS']} | "
                f"forget={float(forget_loss.detach()):.4f} kd={float(kd_cls.detach()):.4f} "
                f"supp={metrics['suppression']:.3f} ret={metrics['retention']:.3f}"
            )
            if metrics["score"] > legacy_best["score"]:
                legacy_best = {
                    "score": metrics["score"],
                    "iteration": iteration,
                    "metrics": metrics,
                    "state": {
                        name: value.detach().cpu().clone()
                        for name, value in legacy.state_dict().items()
                    },
                }
    assert legacy_best["state"] is not None
    legacy.load_state_dict(legacy_best["state"])
    legacy_best["state"] = None
    legacy.eval()
    legacy.zero_grad(set_to_none=True)
    for parameter in legacy.parameters():
        parameter.requires_grad_(False)
    print("restored legacy checkpoint:", legacy_best)
    torch.save({"model": legacy.state_dict()}, WORK / "nb51_legacy_ranker.pth")
    return legacy

legacy_ranker = train_legacy_ranker()

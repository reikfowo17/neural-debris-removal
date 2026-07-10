def in_scrub_scope(name):
    return (
        name.startswith("head.")
        or name.startswith("backbone.fpn")
        or name.startswith("backbone.bottom_up.res5")
    )

def empty_detection_loss(model, images, iteration):
    records = [
        to_batched(image, np.zeros((0, 4), np.float32), train=True)
        for image in images
    ]
    with EventStorage(iteration):
        losses = model(records)
    return losses["loss_cls"] + losses.get("loss_box_reg", 0)

def streaming_topk_threshold(tensors, fraction, steps):
    values = list(tensors)
    total_values = int(sum(value.numel() for value in values))
    target_count = max(1, int(math.ceil(total_values * fraction)))
    _ = steps
    if not all(bool(torch.isfinite(value).all().item()) for value in values):
        return float("inf"), 0, 0
    global_top = None
    for value in values:
        local_count = min(target_count, value.numel())
        local_top = torch.topk(value.reshape(-1), local_count, sorted=False).values
        if global_top is None:
            global_top = local_top
        else:
            merged = torch.cat([global_top, local_top])
            keep_count = min(target_count, merged.numel())
            global_top = torch.topk(merged, keep_count, sorted=False).values
    assert global_top is not None and global_top.numel() == target_count
    threshold = float(global_top.min().item())
    del global_top
    if threshold <= 0.0:
        strict_selected = int(sum(
            torch.count_nonzero(value > 0.0).item() for value in values
        ))
        return 0.0, strict_selected, 0
    strict_selected = int(sum(
        torch.count_nonzero(value > threshold).item() for value in values
    ))
    tie_budget = target_count - strict_selected
    assert 0 <= tie_budget <= target_count
    return threshold, strict_selected, tie_budget

def build_budgeted_mask(tensors, threshold, tie_budget, chunk_size=1_000_000):
    mask = {
        name: (value > threshold).to(torch.uint8)
        for name, value in tensors.items()
    }
    remaining = int(tie_budget)
    if remaining <= 0 or not math.isfinite(threshold):
        return mask
    for name, value in tensors.items():
        flat_value = value.reshape(-1)
        flat_mask = mask[name].reshape(-1)
        for start in range(0, flat_value.numel(), chunk_size):
            if remaining <= 0:
                break
            end = min(start + chunk_size, flat_value.numel())
            tie_positions = torch.nonzero(
                flat_value[start:end] == threshold, as_tuple=False
            ).reshape(-1)
            take = min(remaining, tie_positions.numel())
            if take:
                flat_mask[start + tie_positions[:take]] = 1
                remaining -= take
        if remaining <= 0:
            break
    assert remaining == 0, "global top-k tie budget could not be allocated"
    return mask

def compute_contrastive_mask(model, allowed_sources, fold_index):
    scope_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if in_scrub_scope(name)
    ]
    assert scope_parameters, "empty scrub scope"
    parameter_list = [parameter for _, parameter in scope_parameters]
    repeat_masks = []
    repeat_stats = []
    for repeat_index in range(CFG["SAL_REPEATS"]):
        rng = np.random.default_rng(20000 + 1000 * fold_index + repeat_index)
        accumulation = {
            name: torch.zeros_like(parameter, device=DEVICE)
            for name, parameter in scope_parameters
        }
        model.train()
        for batch_index in range(CFG["SAL_BATCHES"]):
            pairs = [
                make_matched_pair(allowed_sources, rng)
                for _ in range(CFG["BS_FORGET"])
            ]
            poison_loss = empty_detection_loss(
                model, [pair["poison"] for pair in pairs], batch_index
            )
            poison_gradients = torch.autograd.grad(
                poison_loss, parameter_list, allow_unused=True
            )
            control_loss = empty_detection_loss(
                model, [pair["control"] for pair in pairs], batch_index
            )
            control_gradients = torch.autograd.grad(
                control_loss, parameter_list, allow_unused=True
            )
            for (name, parameter), poison_gradient, control_gradient in zip(
                scope_parameters, poison_gradients, control_gradients
            ):
                if poison_gradient is None:
                    poison_gradient = torch.zeros_like(parameter)
                if control_gradient is None:
                    control_gradient = torch.zeros_like(parameter)
                accumulation[name] += (poison_gradient - control_gradient).abs()
            model.zero_grad(set_to_none=True)
        threshold, strict_selected, tie_budget = streaming_topk_threshold(
            accumulation.values(), CFG["MASK_TOPK"], CFG["SAL_THRESHOLD_STEPS"]
        )
        current_mask = build_budgeted_mask(accumulation, threshold, tie_budget)
        selected_from_mask = int(sum(value.sum().item() for value in current_mask.values()))
        selected = strict_selected + tie_budget
        assert selected_from_mask == selected
        repeat_masks.append(current_mask)
        repeat_stats.append({
            "repeat": repeat_index,
            "threshold": float(threshold),
            "selected": selected,
        })
        del accumulation
        torch.cuda.empty_cache()

    pairwise_jaccard = []
    for first_index in range(len(repeat_masks)):
        for second_index in range(first_index + 1, len(repeat_masks)):
            intersection = 0
            union = 0
            for name, _ in scope_parameters:
                first = repeat_masks[first_index][name].bool()
                second = repeat_masks[second_index][name].bool()
                intersection += int((first & second).sum().item())
                union += int((first | second).sum().item())
            pairwise_jaccard.append(intersection / max(union, 1))
    stability = float(np.mean(pairwise_jaccard)) if pairwise_jaccard else 1.0
    stable_mask = {}
    for name, _ in scope_parameters:
        votes = torch.stack([mask[name] for mask in repeat_masks], dim=0).sum(dim=0)
        stable_mask[name] = (votes >= CFG["MASK_MIN_VOTES"]).float()
    selected_stable = int(sum(value.sum().item() for value in stable_mask.values()))
    total_scope = int(sum(value.numel() for value in stable_mask.values()))
    assert selected_stable > 0, "stable contrastive mask selected no weights"
    summary = {
        "fold": fold_index,
        "stability_jaccard": stability,
        "stable_selected": selected_stable,
        "scope_total": total_scope,
        "stable_fraction": selected_stable / max(total_scope, 1),
        "repeats": repeat_stats,
    }
    print("contrastive mask:", summary)
    return stable_mask, summary

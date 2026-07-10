import ast
import json
import math
from pathlib import Path

import torch


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook/experiments/49_contrastive_real_anchor_scrub.ipynb"


def load_notebook():
    with NOTEBOOK.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def notebook_source(notebook):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def load_saliency_helpers():
    source = notebook_source(load_notebook())
    tree = ast.parse(source)
    helper_names = {"streaming_topk_threshold", "build_budgeted_mask"}
    helper_nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), "<saliency-test>", "exec"), namespace)
    return namespace


def test_notebook_exists_and_code_cells_parse():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")


def test_notebook_contains_required_nb49_protocol():
    source = notebook_source(load_notebook())
    required_symbols = [
        "split_source_folds",
        "mine_real_anchors",
        "split_anchor_folds",
        "compute_contrastive_mask",
        "train_scrub_fold",
        "evaluate_fold",
        "aggregate_crossfit_score",
        "ALL_GATES_PASS",
        "SOURCE_DISJOINT_PASS",
        "POISON_SUPPRESSION_PASS",
        "REAL_CONFIDENCE_PASS",
        "BOX_FIDELITY_PASS",
    ]
    for symbol in required_symbols:
        assert symbol in source


def test_export_is_guarded_by_all_gates():
    source = notebook_source(load_notebook())
    assert "if ALL_GATES_PASS:" in source
    assert "nb49_crossfit_rank_Kcal.csv" in source
    assert "NEVER auto-submit" in source


def test_validation_does_not_use_synthetic_retention_gate():
    source = notebook_source(load_notebook())
    assert 'RETENTION_SOURCE = "real_teacher_anchors"' in source
    assert "sample_retain" not in source


def test_coordinate_transform_uses_detectron_transform():
    source = notebook_source(load_notebook())
    assert "tfm.apply_box" in source
    assert "boxes_to_level_cells" not in source


def test_retinanet_raw_outputs_use_version_stable_helpers():
    source = notebook_source(load_notebook())
    assert "def permute_head_output" in source
    assert "from detectron2.layers import cat, permute_to_N_HWA_K" not in source
    assert "generated_anchors = model.anchor_generator(features)" in source
    assert "anchor_levels = generated_anchors" in source
    assert "Boxes.cat(anchor_levels).tensor" in source


def test_saliency_mask_fails_closed_on_zero_or_dense_gradients():
    source = notebook_source(load_notebook())
    assert "MASK_FRACTION_MAX" in source
    assert "(value > threshold).to(torch.uint8)" in source
    assert 'row["mask_fraction"] <= CFG["MASK_FRACTION_MAX"]' in source


def test_saliency_threshold_streams_without_large_global_tensor():
    source = notebook_source(load_notebook())
    assert "import math" in source
    assert "def streaming_topk_threshold" in source
    assert "SAL_THRESHOLD_STEPS" in source
    assert "threshold, strict_selected, tie_budget = streaming_topk_threshold" in source
    assert "def build_budgeted_mask" in source
    assert "torch.topk" in source
    assert "torch.quantile(" not in source
    assert "torch.cat([value.flatten().float() for value in accumulation.values()])" not in source


def test_streaming_mask_handles_ties_without_exceeding_budget():
    helpers = load_saliency_helpers()
    tensors = {
        "first": torch.cat([torch.zeros(90), torch.ones(10)]),
        "second": torch.ones(100),
    }
    threshold, strict_selected, tie_budget = helpers["streaming_topk_threshold"](
        tensors.values(), fraction=0.05, steps=32
    )
    mask = helpers["build_budgeted_mask"](tensors, threshold, tie_budget)
    selected = sum(int(value.sum().item()) for value in mask.values())
    assert strict_selected == 0
    assert selected == math.ceil(200 * 0.05)
    assert selected <= math.ceil(200 * 0.05)


def test_legacy_ranker_preserves_nb46_forget_mixture():
    source = notebook_source(load_notebook())
    assert "if draw < 0.15:" in source
    assert "if draw < 0.40:" in source
    assert "def audit_legacy_ranker" in source
    assert 'legacy_best = {"score": -1e9' in source
    assert 'legacy.load_state_dict(legacy_best["state"])' in source
    assert "parameter.requires_grad_(False)" in source
    assert "legacy.zero_grad(set_to_none=True)" in source


def test_cells_follow_runtime_dependency_order():
    source = notebook_source(load_notebook())
    ordered_markers = [
        "def make_matched_pair",
        "def train_legacy_ranker",
        "def mine_real_anchors",
        "def compute_contrastive_mask",
        "def train_scrub_fold",
        "SOURCE_DISJOINT_PASS =",
        "def aggregate_crossfit_score",
    ]
    positions = [source.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)


def test_checkpoint_artifacts_do_not_serialize_model_state():
    source = notebook_source(load_notebook())
    assert '"checkpoint": best,' not in source
    assert "best_summary =" in source
    assert '"checkpoint": best_summary,' in source


def test_crossfit_models_are_loaded_one_fold_at_a_time():
    source = notebook_source(load_notebook())
    assert "fold_checkpoints = []" in source
    assert "fold_models.append(scrub)" not in source
    assert "fold_checkpoints.append(checkpoint_path)" in source
    assert "fold_model = load_model(checkpoint_path)" in source
    assert 'scrub.to("cpu")' in source
    assert "torch.cuda.empty_cache()" in source


def test_stale_submission_artifacts_are_removed_before_training():
    source = notebook_source(load_notebook())
    assert "STALE_EXPORTS" in source
    assert "stale_path.unlink(missing_ok=True)" in source


def test_competition_rule_evidence_unlocks_real_anchor_distillation():
    source = notebook_source(load_notebook())
    assert "RULE_EVIDENCE" in source
    assert "Unlabeled Test Set can be used" in source
    assert "RULES_CONFIRMED=True" in source
    assert "RULES_CONFIRMED=False" not in source
    assert 'assert CFG["RULES_CONFIRMED"]' in source


def test_checkpoint_selection_prefers_fidelity_admissible_state():
    source = notebook_source(load_notebook())
    assert "def checkpoint_is_admissible" in source
    assert 'best_safe = {"score": -1e9' in source
    assert 'chosen = best_safe if best_safe["state"] is not None else best_any' in source
    assert 'checkpoint["state"] = None' not in source
    assert 'accepted = False' in source


def test_rank_auc_runs_only_on_final_gate_evaluation():
    source = notebook_source(load_notebook())
    assert "def evaluate_fold(scrub, held_pairs, held_anchors, candidate_cache, include_rank=True):" in source
    assert "if include_rank:" in source
    assert "include_rank=False" in source


def test_rank_gate_uses_noninferiority_per_fold_and_gain_in_aggregate():
    source = notebook_source(load_notebook())
    assert "RANK_FOLD_TOLERANCE" in source
    assert 'mean_legacy_auc = float(np.mean([row["legacy_auc"] for row in fold_rows]))' in source
    assert 'mean_blend_auc = float(np.mean([row["blend_auc"] for row in fold_rows]))' in source
    assert 'row["blend_auc"] + CFG["RANK_FOLD_TOLERANCE"] >= row["legacy_auc"]' in source
    assert "mean_blend_auc >= mean_legacy_auc + CFG[\"RANK_AUC_GAIN_MIN\"]" in source
    assert 'row["blend_auc"] >= row["legacy_auc"] + CFG["RANK_AUC_GAIN_MIN"]' not in source


def test_runtime_fold_metrics_pass_corrected_rank_gate():
    rows = [
        {"legacy_auc": 0.9989130434782609, "blend_auc": 1.0},
        {"legacy_auc": 1.0, "blend_auc": 1.0},
        {"legacy_auc": 1.0, "blend_auc": 1.0},
        {"legacy_auc": 0.9545454545454546, "blend_auc": 0.9960474308300395},
    ]
    tolerance = 1e-12
    minimum_auc = 0.88
    minimum_gain = 0.01
    mean_legacy = sum(row["legacy_auc"] for row in rows) / len(rows)
    mean_blend = sum(row["blend_auc"] for row in rows) / len(rows)
    assert all(
        row["blend_auc"] >= minimum_auc
        and row["blend_auc"] + tolerance >= row["legacy_auc"]
        for row in rows
    )
    assert mean_blend >= mean_legacy + minimum_gain

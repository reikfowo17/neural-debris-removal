import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook/experiments/50_nested_alpha_k_sweep.ipynb"


def load_notebook():
    with NOTEBOOK.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def code_source():
    notebook = load_notebook()
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_exists_and_all_code_cells_parse():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")


def test_nb50_is_standalone_and_trains_its_own_rankers():
    source = code_source()
    assert 'MODE = "standalone_train_and_select"' in source
    assert "def train_legacy_ranker" in source
    assert "def train_scrub_fold" in source
    assert "LEGACY_ITERS" in source
    assert "SCRUB_ITERS" in source
    assert "EventStorage(" in source


def test_nb50_does_not_require_nb49_output_as_input():
    source = code_source()
    assert "resolve_nb49_checkpoints" not in source
    assert "NB49_INPUT_HINT" not in source
    assert "NB49_CHECKPOINTS" not in source
    assert "/kaggle/input/**/nb49_legacy_ranker.pth" not in source
    assert "nb50_legacy_ranker.pth" in source
    assert 'checkpoint_path = WORK / f"nb50_fold{fold_index}_scrub.pth"' in source
    assert 'torch.save({"model": legacy.state_dict()}, WORK / "nb50_legacy_ranker.pth")' in source


def test_nb50_clears_all_stale_standalone_artifacts_before_training():
    source = code_source()
    assert 'for pattern in ("nb50_*.csv", "nb50_*.json", "nb50_*.pth"):' in source
    assert "for stale_path in WORK.glob(pattern):" in source
    assert "stale_path.unlink(missing_ok=True)" in source


def test_nested_alpha_selection_excludes_target_fold():
    source = code_source()
    assert "def choose_nested_alphas" in source
    assert 'frame["fold"] != target_fold' in source
    assert "selected_alpha_by_fold" in source
    assert "ALPHA_GRID = [0.50, 0.65, 0.80, 1.00]" in source


def test_alpha_selection_uses_low_false_positive_operating_metrics():
    source = code_source()
    assert "def tpr_at_fpr" in source
    assert "def partial_auc" in source
    assert "def low_fpr_margin" in source
    assert "MAX_FALSE_POSITIVE_RATE" in source
    assert "partial_auc_10" in source
    assert "tpr_at_fpr_02" in source
    assert "low_fpr_margin" in source


def test_oof_validation_uses_source_and_anchor_disjoint_folds():
    source = code_source()
    assert "SOURCE_FOLDS" in source
    assert "ANCHOR_FOLD" in source
    assert "collect_oof_rank_rows" in source
    assert "held_sources = SOURCE_FOLDS[fold_index]" in source
    assert 'ANCHOR_FOLD[anchor["image_id"]] == fold_index' in source


def test_crossfit_test_scoring_uses_fold_specific_alpha():
    source = code_source()
    assert "def build_nested_crossfit_scores" in source
    assert "alpha = alpha_by_fold[fold_index]" in source
    assert "legacy_poison = 1.0 - cache[\"legacy_ratio\"]" in source
    assert "scrub_poison = 1.0 - scrub_ratio" in source


def test_k_sweep_is_bounded_and_marks_one_primary_candidate():
    source = code_source()
    assert "K_SWEEP = [440, 450, 466, 480, 490]" in source
    assert "PRIMARY_K = 466" in source
    assert "nb50_nested_alpha_K466.csv" in source
    assert '"submit_priority": "primary"' in source
    assert "nb50_manifest.csv" in source
    assert "NEVER auto-submit" in source


def test_selection_fails_closed_if_nested_score_does_not_beat_nb49_blend():
    source = code_source()
    assert "NESTED_SELECTION_PASS" in source
    assert "REFERENCE_ALPHA = 0.50" in source
    assert "per_fold_comparison" in source
    assert 'nested_metrics["low_fpr_margin"] >= reference_metrics["low_fpr_margin"] + CFG["NESTED_GAIN_MIN"]' in source
    assert 'nested_metrics["partial_auc_10"] >= reference_metrics["partial_auc_10"] + CFG["NESTED_GAIN_MIN"]' not in source
    assert "if TRAINING_GATES_PASS and NESTED_SELECTION_PASS:" in source
    assert "NB50 FAIL CLOSED" in source


def test_nested_oof_metrics_use_direct_score_vector_without_duplicate_columns():
    source = code_source()
    assert "def metrics_from_scores" in source
    assert 'metrics_from_scores(labels, output["nested_score"].to_numpy' in source
    assert 'rename(columns={"nested_score": "scrub_poison"})' not in source


def test_oof_uses_the_same_source_disjoint_pair_generator_as_training():
    source = code_source()
    assert "def make_validation_pairs" in source
    assert "make_validation_pairs(held_sources, fold_index, seed_offset=1)" in source
    assert 'patch_record = source_records[source_id]["patches"][' in source


def test_nested_selection_runs_only_after_training_gates_pass():
    source = code_source()
    assert "ALL_GATES_PASS" in source
    assert "if ALL_GATES_PASS:" in source
    assert "NESTED_SELECTION_PASS = bool(" in source
    assert "TRAINING_GATES_PASS" in source
    assert "if TRAINING_GATES_PASS and NESTED_SELECTION_PASS:" in source

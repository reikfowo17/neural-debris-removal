import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook/experiments/51_deadline_fixed_alpha_sweep.ipynb"


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


def test_nb51_is_standalone_and_trains_before_sweeping():
    source = code_source()
    assert 'MODE = "standalone_train_fixed_alpha_sweep"' in source
    assert "def train_legacy_ranker" in source
    assert "def train_scrub_fold" in source
    assert "LEGACY_ITERS" in source
    assert "SCRUB_ITERS" in source
    assert "EventStorage(" in source


def test_nb51_creates_its_own_checkpoints_without_saved_input():
    source = code_source()
    assert "resolve_nb50_saved_run" not in source
    assert "/kaggle/input/**/nb50_legacy_ranker.pth" not in source
    assert "NB50_FILES" not in source
    assert "nb51_legacy_ranker.pth" in source
    assert 'checkpoint_path = WORK / f"nb51_fold{fold_index}_scrub.pth"' in source
    assert 'torch.save({"model": legacy.state_dict()}, WORK / "nb51_legacy_ranker.pth")' in source


def test_standalone_training_hard_gates_must_pass():
    source = code_source()
    assert "ALL_GATES_PASS" in source
    assert "TRAINING_GATES_PASS = False" in source
    assert "if ALL_GATES_PASS:" in source
    assert "TRAINING_GATES_PASS = True" in source
    assert "if TRAINING_GATES_PASS:" in source


def test_fixed_candidates_are_preregistered_and_unambiguous():
    source = code_source()
    assert 'FIXED_CANDIDATES = [' in source
    assert '{"name": "A065_K466", "alpha": 0.65, "target_k": 466, "priority": 1}' in source
    assert '{"name": "A080_K466", "alpha": 0.80, "target_k": 466, "priority": 2}' in source
    assert '{"name": "A065_K450", "alpha": 0.65, "target_k": 450, "priority": 3}' in source
    assert '{"name": "A065_K480", "alpha": 0.65, "target_k": 480, "priority": 4}' in source
    assert "ALPHA_GRID" not in source
    assert "choose_nested_alphas" not in source


def test_oof_audit_compares_every_candidate_to_nb49_reference():
    source = code_source()
    assert "def collect_oof_rank_rows" in source
    assert "def audit_fixed_candidates" in source
    assert "REFERENCE_ALPHA = 0.50" in source
    assert "per_fold_metrics" in source
    assert "low_fpr_margin" in source
    assert "nb51_oof_candidate_audit.csv" in source


def test_test_scoring_uses_new_fold_checkpoints_and_fixed_alpha():
    source = code_source()
    assert "def collect_test_components" in source
    assert "load_model(fold_checkpoints[fold_index])" in source
    assert "legacy_poison" in source
    assert "scrub_poison" in source
    assert "def fixed_alpha_scores" in source


def test_threshold_calibration_is_exact_and_deterministic():
    source = code_source()
    assert "def calibrate_exact_keep_set" in source
    assert "np.lexsort" in source
    assert "target_k" in source
    assert "actual_k == min(target_k, len(eligible_rows))" in source


def test_exports_four_named_candidates_and_manifest():
    source = code_source()
    for name in ("A065_K466", "A080_K466", "A065_K450", "A065_K480"):
        assert f"nb51_{name}.csv" in source
    assert "nb51_manifest.csv" in source
    assert '"submit_first": candidate["priority"] == 1' in source
    assert "NEVER auto-submit" in source


def test_nb51_clears_only_its_own_stale_outputs():
    source = code_source()
    assert 'for pattern in ("nb51_*.csv", "nb51_*.json", "nb51_*.pth"):' in source
    assert 'WORK.glob(pattern)' in source
    assert 'WORK.glob("nb50_' not in source

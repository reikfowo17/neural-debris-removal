import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "52_sparse_exact_k_sweep.ipynb"


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


def test_nb52_is_standalone_and_retrains_rankers():
    source = code_source()
    assert 'MODE = "standalone_sparse_exact_k_sweep"' in source
    assert "def train_legacy_ranker" in source
    assert "def train_scrub_fold" in source
    assert "LEGACY_ITERS" in source
    assert "SCRUB_ITERS" in source
    assert 'torch.save({"model": legacy.state_dict()}, WORK / "nb52_legacy_ranker.pth")' in source
    assert 'checkpoint_path = WORK / f"nb52_fold{fold_index}_scrub.pth"' in source


def test_sparse_candidates_are_preregistered_with_one_primary():
    source = code_source()
    assert '{"name": "SPARSE_A065_K466", "alpha": 0.65, "target_k": 466, "priority": 1}' in source
    assert '{"name": "SPARSE_A065_K450", "alpha": 0.65, "target_k": 450, "priority": 2}' in source
    assert '{"name": "SPARSE_A065_K480", "alpha": 0.65, "target_k": 480, "priority": 3}' in source
    assert '{"name": "SPARSE_A080_K466", "alpha": 0.80, "target_k": 466, "priority": 4}' in source
    assert 'sum(candidate["priority"] == 1 for candidate in FIXED_CANDIDATES) == 1' in source


def test_training_and_oof_audit_fail_closed():
    source = code_source()
    assert "ALL_GATES_PASS" in source
    assert "TRAINING_GATES_PASS = False" in source
    assert "if ALL_GATES_PASS:" in source
    assert "if TRAINING_GATES_PASS:" in source
    assert "def collect_oof_rank_rows" in source
    assert "def audit_fixed_candidates" in source
    assert "REFERENCE_ALPHA = 0.50" in source


def test_sparse_composer_outputs_only_selected_boxes():
    source = code_source()
    assert "def compose_sparse_submission" in source
    assert "selected_indices = sorted(selected_by_image.get(stem, set()))" in source
    assert 'boxes = item["boxes"][selected_indices]' in source
    assert 'teacher_scores = item["scores"][selected_indices]' in source
    assert "output_scores = np.minimum(1.0, teacher_scores + CFG[\"BOOST\"])" in source
    assert "EPS_Q" not in source
    assert "np.full(len(indices)" not in source


def test_sparse_exact_k_is_deterministic_and_capped_per_image():
    source = code_source()
    assert "def calibrate_exact_keep_set" in source
    assert "np.lexsort" in source
    assert "MAX_KEEP_PER_IMAGE=2" in source
    assert "actual_k == min(target_k, len(eligible_rows))" in source
    assert "assert actual_k == calibrated_k == candidate[\"target_k\"]" in source


def test_sparse_manifest_reports_removed_low_tier_mass():
    source = code_source()
    assert "candidate_pool_count" in source
    assert "omitted_candidate_count" in source
    assert "dense_eps_mass_avoided" in source
    assert 'dense_eps_mass_avoided = omitted_candidate_count * DENSE_REFERENCE_EPS' in source
    assert "DENSE_REFERENCE_EPS = 0.02" in source


def test_exports_multiple_files_but_marks_only_one_submit_first():
    source = code_source()
    for name in (
        "SPARSE_A065_K466",
        "SPARSE_A065_K450",
        "SPARSE_A065_K480",
        "SPARSE_A080_K466",
    ):
        assert f"nb52_{name}.csv" in source
    assert "nb52_manifest.csv" in source
    assert '"submit_first": candidate["priority"] == 1' in source
    assert 'assert int(manifest["submit_first"].sum()) == 1' in source
    assert "NEVER auto-submit" in source


def test_nb52_clears_only_its_own_stale_outputs():
    source = code_source()
    assert 'for pattern in ("nb52_*.csv", "nb52_*.json", "nb52_*.pth"):' in source
    assert "WORK.glob(pattern)" in source
    assert 'WORK.glob("nb51_' not in source


def test_no_saved_nb51_checkpoint_input_dependency():
    source = code_source()
    assert "resolve_nb51" not in source
    assert "NB51_FILES" not in source
    assert "/kaggle/input/**/nb51_" not in source


def test_nb52_has_minimal_markdown_and_compact_runtime_logging():
    notebook = load_notebook()
    markdown_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    source = code_source()
    assert len(markdown_cells) == 1
    assert sum(len("".join(cell["source"])) for cell in markdown_cells) < 500
    assert "for key, value in CFG.items()" not in source
    assert 'print("GATE METRICS:"' not in source
    assert "fold_frame.to_string" not in source
    assert 'print("contrastive mask:"' not in source
    assert "oof_candidate_audit.to_string(index=False)" not in source


def test_nb52_keeps_compact_decision_outputs():
    source = code_source()
    assert "NB52 SUMMARY" in source
    assert "NB52 OOF AUDIT" in source
    assert "NB52 SPARSE MANIFEST" in source

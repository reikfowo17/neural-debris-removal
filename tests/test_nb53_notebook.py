import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "53_counterfactual_response_fingerprint.ipynb"


def load_notebook():
    with NOTEBOOK.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def code_source():
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_exists_and_all_code_cells_parse():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")


def test_nb53_is_independent_and_inference_only():
    source = code_source()
    assert 'MODE = "standalone_counterfactual_response"' in source
    for old_version in ("nb49_", "nb50_", "nb51_", "nb52_"):
        assert old_version not in source
    assert "/kaggle/input/**/nb" not in source
    assert "train_legacy_ranker" not in source
    assert "train_scrub_fold" not in source
    assert ".backward(" not in source
    assert "optimizer" not in source
    assert "torch.save" not in source


def test_nb53_uses_only_competition_data_and_poisoned_model():
    source = code_source()
    assert "annotations_coco.json" in source
    assert "sample_submission.csv" in source
    assert "POISONED_WEIGHTS" in source
    assert "def resolve_competition_data" in source
    assert "def load_model" in source
    assert "REQUIRED_SAVED_FILES" not in source
    assert "resolve_nb" not in source


def test_nb53_builds_new_counterfactual_response_signal():
    source = code_source()
    assert "def inpaint_box" in source
    assert "def build_counterfactual_stack" in source
    assert "amplitude_35" in source
    assert "amplitude_65" in source
    assert "low_frequency_only" in source
    assert "high_frequency_only" in source
    assert "def response_fingerprint" in source
    assert "frequency_interaction" in source


def test_nb53_has_source_disjoint_validation_and_real_anchors():
    source = code_source()
    assert "def split_source_folds" in source
    assert "def mine_stable_real_anchors" in source
    assert "source_id not in poison_source_ids" in source
    assert "def crossfit_response_audit" in source
    assert "def fit_robust_calibrator" in source
    assert "def score_fingerprint" in source
    assert "RESPONSE_GATES_PASS" in source


def test_nb53_preserves_candidates_and_uses_continuous_rescoring():
    source = code_source()
    assert "def collect_test_fingerprints" in source
    assert 'keep = scores >= CFG["BASE_THR"]' in source
    assert "def compose_response_submission" in source
    assert 'output_scores = np.maximum(CFG["EPS"], calibrated_scores)' in source
    assert 'assert output_count == candidate_count' in source
    assert "calibrate_exact_keep_set" not in source
    assert "MAX_KEEP_PER_IMAGE" not in source
    assert "compose_sparse_submission" not in source


def test_nb53_preregisters_one_primary_and_two_hedges():
    source = code_source()
    assert '{"name": "CF_BALANCED", "strength": 0.78, "power": 1.50, "priority": 1}' in source
    assert '{"name": "CF_CONSERVATIVE", "strength": 0.62, "power": 1.75, "priority": 2}' in source
    assert '{"name": "CF_AGGRESSIVE", "strength": 0.90, "power": 1.25, "priority": 3}' in source
    assert 'sum(candidate["priority"] == 1 for candidate in RESPONSE_CANDIDATES) == 1' in source


def test_nb53_exports_only_its_own_artifacts():
    source = code_source()
    assert "nb53_CF_BALANCED.csv" in source
    assert "nb53_CF_CONSERVATIVE.csv" in source
    assert "nb53_CF_AGGRESSIVE.csv" in source
    assert "nb53_response_oof.csv" in source
    assert "nb53_manifest.csv" in source
    assert "nb53_run_report.json" in source
    assert 'for pattern in ("nb53_*.csv", "nb53_*.json"):' in source
    assert '"submit_first": candidate["priority"] == 1' in source
    assert 'assert int(manifest["submit_first"].sum()) == 1' in source
    assert "NEVER auto-submit" in source


def test_nb53_is_compact_and_unexecuted():
    notebook = load_notebook()
    markdown_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]
    assert len(markdown_cells) == 1
    assert len("".join(markdown_cells[0]["source"])) < 500
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
NB55 = ROOT / "notebook/experiments/55_conservative_floor_sweep.ipynb"
NB56 = ROOT / "notebook/experiments/56_pcgrad_boxkd_repair.ipynb"


def notebook_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return notebook, "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )


def load_functions(path, names, namespace):
    _, source = notebook_source(path)
    tree = ast.parse(source)
    by_name = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names and node.name not in by_name:
            by_name[node.name] = node
    nodes = list(by_name.values())
    found = {node.name for node in nodes}
    assert found == set(names), f"missing helpers: {set(names) - found}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def toy_cache():
    return {
        "1": {"boxes": np.array([[0, 0, 2, 2], [2, 0, 4, 2], [4, 0, 6, 2]], np.float32),
              "scores": np.array([0.90, 0.50, 0.40], np.float32)},
        "02": {"boxes": np.array([[0, 2, 2, 4], [2, 2, 4, 4], [4, 2, 6, 4]], np.float32),
               "scores": np.array([0.80, 0.60, 0.20], np.float32)},
    }


def nb55_helpers():
    namespace = {"np": np, "pd": pd, "json": json}
    namespace["CFG"] = {"S_MIN": 0.30, "MAX_KEEP_PER_IMAGE": 2,
                         "BASE_THR": 0.10, "BOOST": 0.15}
    namespace["TAIL_FLOORS"] = (0.03, 0.04, 0.05)
    return load_functions(
        NB55,
        {"calibrate_exact_keep_set", "compose_floor_frame", "normalize_id", "format_prediction",
         "publish_floor_bundle", "validate_submission_csv"},
        namespace,
    )


def test_notebooks_parse_and_are_unexecuted():
    for path in (NB55, NB56):
        notebook, _ = notebook_source(path)
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])), filename=f"{path.name}:{index}")
                if path == NB56:
                    assert cell.get("execution_count") is None
                    assert cell.get("outputs") == []
                    assert cell.get("metadata") == {}


def test_nb55_selection_is_ascending_capped_and_preserves_original_indices():
    helpers = nb55_helpers()
    cache = toy_cache()
    scores = {"1": np.array([0.9, 0.1, 0.2]), "02": np.array([0.8, 0.05, 0.01])}
    selected, actual, boundary = helpers["calibrate_exact_keep_set"](
        cache, scores, target_k=3, test_stems=["1", "02"]
    )
    assert actual == 3
    assert selected == {"1": {1, 2}, "02": {1}}
    assert all(len(indices) <= 2 for indices in selected.values())
    assert boundary == pytest.approx(0.2)


def test_nb55_composer_uses_baseline_universe_boost_and_exact_floor():
    helpers = nb55_helpers()
    cache = toy_cache()
    # id and image_id are independent, matching competition sample_submission.
    sample = pd.DataFrame({"id": [10, 11], "image_id": ["1", "2"], "prediction_string": ["", ""]})
    selected = {"1": {1}, "02": {0}}
    frames = {}
    for floor in (0.03, 0.04, 0.05):
        frame, audit = helpers["compose_floor_frame"](cache, selected, floor, sample)
        frames[floor] = frame
        assert list(frame.columns) == ["id", "image_id", "prediction_string"]
        assert frame["id"].tolist() == [10, 11]
        assert frame["image_id"].tolist() == ["1", "2"]
        assert audit["selected_count"] == 2
        values = [np.asarray([float(v) for v in text.split()[::5]]) for text in frame.prediction_string]
        assert values[0].tolist() == pytest.approx([floor, 0.65, floor])
        assert values[1].tolist() == pytest.approx([0.95, floor, floor])
    assert frames[0.03].prediction_string.tolist() != frames[0.04].prediction_string.tolist()
    assert frames[0.04].prediction_string.tolist() != frames[0.05].prediction_string.tolist()


def test_nb55_submission_csv_preserves_sample_id_image_id_mapping(tmp_path):
    helpers = nb55_helpers()
    cache = toy_cache()
    sample = pd.DataFrame({
        "id": [100, 101],
        "image_id": [2, 1],  # intentionally not equal to id
        "prediction_string": ["old-a", "old-b"],
        "extra": ["drop-me", "drop-me"],
    })
    selected = {"1": {1}, "02": {0}}
    frame, audit = helpers["compose_floor_frame"](cache, selected, 0.03, sample)
    assert list(frame.columns) == ["id", "image_id", "prediction_string"]
    assert frame["id"].tolist() == [100, 101]
    assert frame["image_id"].tolist() == ["2", "1"]
    assert audit["selected_count"] == 2
    path = tmp_path / "nb55_F003_A065_K466.csv"
    frame.to_csv(path, index=False)
    assert helpers["validate_submission_csv"](path, expected_rows=2)
    reloaded = pd.read_csv(path)
    assert reloaded["id"].tolist() == [100, 101]
    assert reloaded["image_id"].astype(str).tolist() == ["2", "1"]


def test_nb55_format_prediction_is_exactly_nb51_compatible():
    formatter = nb55_helpers()["format_prediction"]
    boxes = np.array([[-2.123, 1.234, 1028.0, 4.999], [5, 5, 4, 7]], np.float32)
    scores = np.array([0.12345678, 0.9], np.float32)
    assert formatter(boxes, scores) == "0.123457 0.00 1.23 1024.00 3.77"


def test_nb55_bundle_publish_rolls_back_finals_and_temps_on_mid_publish_failure(tmp_path):
    publish = nb55_helpers()["publish_floor_bundle"]
    frames = {
        floor: pd.DataFrame({"id": [1], "image_id": ["1"], "prediction_string": [str(floor)]})
        for floor in (0.03, 0.04, 0.05)
    }
    outputs = {floor: tmp_path / f"floor_{floor}.csv" for floor in frames}
    report_path = tmp_path / "report.json"
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        source.replace(destination)

    with pytest.raises(OSError, match="injected publish failure"):
        publish(frames, {"ok": True}, outputs, report_path,
                validate_frame=lambda frame, floor: None, replace_func=fail_second_replace)
    assert not any(path.exists() for path in [*outputs.values(), report_path])
    assert not list(tmp_path.glob("*.tmp"))


def test_nb55_composer_rejects_normalized_id_collisions_and_bad_schema():
    helpers = nb55_helpers()
    cache = toy_cache()
    collision = dict(cache)
    collision["2"] = collision["02"]
    sample = pd.DataFrame({"id": [1, 2], "image_id": ["1", "2"], "prediction_string": ["", ""]})
    with pytest.raises(AssertionError, match="normalized image-id collision"):
        helpers["compose_floor_frame"](collision, {}, 0.03, sample)
    with pytest.raises(AssertionError, match="submission schema"):
        helpers["compose_floor_frame"](cache, {}, 0.03, sample.drop(columns="id"))


def test_nb55_embeds_nb51_training_gates_crossfit_and_streaming():
    _, source = notebook_source(NB55)
    for marker in ("compute_contrastive_mask", "mine_real_anchors", "collect_oof_rank_rows",
                   "SOURCE_DISJOINT_PASS", "MASK_STABILITY_PASS", "RANK_AUC_PASS",
                   "checkpoint_path = WORK / f\"nb55_fold{fold_index}_scrub.pth\""):
        assert marker in source
    assert 'fold_model = load_model(fold_checkpoints[fold_index])' in source
    assert 'fold_model.to("cpu")' in source
    assert "torch.cuda.empty_cache()" in source
    assert "resolve_nb50_saved_run" not in source


def test_nb55_cleans_stale_files_and_uses_atomic_publication():
    _, source = notebook_source(NB55)
    assert 'for pattern in ("nb55_*.csv", "nb55_*.json", "nb55_*.pth", "nb55_*.tmp")' in source
    assert "def atomic_write_csv(" in source
    assert "temporary.replace(output_path)" in source
    assert "if TRAINING_GATES_PASS:" in source


def nb56_helpers():
    return load_functions(
        NB56,
        {"iou_matrix", "global_masked_asymmetric_pcgrad", "split_nested_folds", "source_sets_are_disjoint",
         "validate_fold_cache_scope", "validate_fold_legacy_provenance", "target_box_match", "candidate_match_stability",
         "rank_auc_from_rows"},
        {"torch": torch, "np": np, "roc_auc_score": roc_auc_score, "hashlib": hashlib},
    )


def test_global_pcgrad_does_not_project_local_conflict_when_global_dot_is_aligned():
    pcgrad = nb56_helpers()["global_masked_asymmetric_pcgrad"]
    projected = pcgrad(
        [torch.tensor([1.0]), torch.tensor([1.0])],
        [torch.tensor([-1.0]), torch.tensor([2.0])],
        [torch.ones(1), torch.ones(1)],
    )
    assert torch.equal(projected[0], torch.tensor([1.0]))
    assert torch.equal(projected[1], torch.tensor([1.0]))


def test_global_pcgrad_projects_global_conflict_on_stable_coordinates():
    pcgrad = nb56_helpers()["global_masked_asymmetric_pcgrad"]
    projected = pcgrad(
        [torch.tensor([1.0]), torch.tensor([0.0])],
        [torch.tensor([-1.0]), torch.tensor([1.0])],
        [torch.ones(1), torch.ones(1)],
    )
    assert torch.cat(projected).tolist() == pytest.approx([0.5, 0.5])
    assert torch.dot(torch.cat(projected), torch.tensor([-1.0, 1.0])).item() == pytest.approx(0.0)


def test_nested_split_is_disjoint_and_held_data_never_trains():
    helpers = nb56_helpers()
    rows = helpers["split_nested_folds"]([f"s{i}" for i in range(12)], 4, 7)
    assert all(helpers["source_sets_are_disjoint"](row) for row in rows)
    for row in rows:
        assert not set(row["training_sources"]) & set(row["held_sources"])
        assert not set(row["training_sources"]) & set(row["checkpoint_sources"])


def test_nb56_actual_fold_legacy_metadata_excludes_held_and_checkpoint_sources():
    helpers = nb56_helpers()
    source_ids = [f"s{i}" for i in range(12)]
    protocol = helpers["split_nested_folds"](source_ids, 4, 7)
    artifact = {"fold": 0, "legacy_training_sources": protocol[0]["training_sources"]}
    cache_rows = [{"legacy_fold": 0, "legacy_training_sources": tuple(protocol[0]["training_sources"])}]
    assert helpers["validate_fold_legacy_provenance"](protocol[0], artifact, cache_rows)
    _, source = notebook_source(NB56)
    assert 'FOLD_PROTOCOL = split_nested_folds(source_ids, CFG["N_FOLDS"], CFG["SEED"])' in source
    assert 'fold_protocol = FOLD_PROTOCOL[fold_index]' in source
    assert 'legacy_training_sources = fold_protocol["training_sources"]' in source
    assert 'set(legacy_training_sources) == set(training_sources)' in source
    assert '"legacy_training_sources": legacy_training_sources' in source
    assert '"legacy_checkpoint_path": str(legacy_checkpoint_path)' in source
    assert 'fold_candidate_cache[path.stem]["legacy_fold"] = fold_index' in source
    assert 'legacy_ranker.to("cpu")' in source


def test_nb56_fold_evaluation_resolves_nonassigned_anchors_from_fold_cache():
    validate = nb56_helpers()["validate_fold_cache_scope"]
    fold_cache = {
        "assigned": {"legacy_fold": 1},
        "39": {"legacy_fold": 1},
        "checkpoint": {"legacy_fold": 1},
    }
    # Global export cache already contains prior-fold images; that must not fail fold 1.
    global_cache = {
        "prior": {"legacy_fold": 0},
        "assigned": fold_cache["assigned"],
    }
    checkpoint_anchors = [{"image_id": "checkpoint"}]
    gate_anchors = [{"image_id": "39"}]
    assigned_ids = {"assigned"}

    assert "39" not in global_cache
    assert fold_cache[gate_anchors[0]["image_id"]]["legacy_fold"] == 1
    assert validate(
        fold_cache, global_cache, checkpoint_anchors, gate_anchors, assigned_ids, fold_index=1
    )
    with pytest.raises(AssertionError):
        # Leak a non-assigned fold-1 image into the export cache.
        bad_global = dict(global_cache)
        bad_global["39"] = {"legacy_fold": 1}
        validate(
            fold_cache, bad_global, checkpoint_anchors, gate_anchors, assigned_ids, fold_index=1
        )


def test_nb56_checkpoint_and_gate_evaluation_use_fold_local_cache_only():
    _, source = notebook_source(NB56)
    training = source[source.index("def train_scrub_fold("):source.index("fold_checkpoints = []")]
    assert "fold_candidate_cache" in training
    assert "checkpoint_anchors,\n                fold_candidate_cache," in training
    fold_loop = source[source.index("fold_checkpoints = []"):source.index("fold_frame = pd.DataFrame")]
    assert "evaluate_fold(scrub, gate_pairs, gate_anchors, fold_candidate_cache)" in fold_loop
    assert "assert checkpoint_anchor_ids <= set(fold_candidate_cache)" in fold_loop
    assert "assert gate_anchor_ids <= set(fold_candidate_cache)" in fold_loop
    assert "assigned_fold_ids = {" in fold_loop
    assert "assert assigned_fold_ids <= set(test_candidate_cache)" in fold_loop
    assert 'item.get("legacy_fold") == fold_index and stem not in assigned_fold_ids' in fold_loop
    assert 'fold_candidate_cache[path.stem]["legacy_fold"] = fold_index' in fold_loop


def test_target_box_matching_and_candidate_stability_execute():
    helpers = nb56_helpers()
    target = np.array([[0, 0, 10, 10]], np.float32)
    boxes = np.array([[20, 20, 30, 30], [1, 1, 9, 9]], np.float32)
    scores = np.array([0.9, 0.4], np.float32)
    assert helpers["target_box_match"](target, boxes, scores, 0.5) == pytest.approx(0.4)
    stability = helpers["candidate_match_stability"](boxes, scores, boxes.copy(), scores * 0.9)
    assert stability["match_fraction"] == pytest.approx(1.0)
    assert stability["mean_iou"] == pytest.approx(1.0)


def test_rank_auc_executes_sklearn_roc_auc_on_positive_and_real_rows():
    rank_auc = nb56_helpers()["rank_auc_from_rows"]
    rows = [{"label": 1, "score": 0.9}, {"label": 1, "score": 0.8},
            {"label": 0, "score": 0.2}, {"label": 0, "score": 0.1}]
    assert rank_auc(rows) == pytest.approx(roc_auc_score([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]))


def test_nb56_embeds_full_nb49_protocol_and_real_kd():
    _, source = notebook_source(NB56)
    for marker in ("make_matched_pair", "(poison_gradient - control_gradient).abs()", "compute_contrastive_mask",
                   "mine_real_anchors", "ANCHOR_MAX_POISON_SCORE", "anchor_localized_kd",
                   "teacher_probabilities", "teacher_deltas", "target_box_match", "roc_auc_score",
                   "candidate_match_stability", "N_FOLDS=4"):
        assert marker in source
    assert "forget_loss.backward(retain_graph=True)" in source
    assert "retain_loss.backward()" in source
    assert "global_masked_asymmetric_pcgrad" in source
    assert "global_forget = torch.cat" in source
    assert "global_retain = torch.cat" in source
    assert "parameter.grad.mul_(stable_mask[name])" in source
    assert "decoded_iou_loss = decoded_box_iou_fidelity" in source
    assert "+ 0.25 * decoded_iou_loss" in source
    assert '"candidate_match_fraction"' in source
    assert "CANDIDATE_STABILITY_PASS" in source
    assert '"candidate_score_ratio"' in source
    assert "CANDIDATE_SCORE_RATIO_PASS" in source
    assert 'CFG["CANDIDATE_SCORE_RATIO_MIN"]' in source


def test_nb56_exports_crossfit_ranker_not_raw_repaired_detections():
    _, source = notebook_source(NB56)
    assert "collect_test_components" in source
    assert "calibrate_exact_keep_set" in source
    assert "compose_ranked_submission" in source
    publish = source[source.index("def compose_ranked_submission"):]
    assert 'model([image_record(path)])' not in publish


def test_nb56_rank_output_is_cleaned_before_gates_and_written_atomically():
    _, source = notebook_source(NB56)
    cleanup = '"nb56_PCGRAD_RANK_K467.csv"'
    assert cleanup in source
    assert "unlink(missing_ok=True)" in source
    assert "temporary.replace(output_path)" in source
    assert source.index(cleanup) < source.index("if ALL_GATES_PASS:")


def test_nb56_cleanup_is_first_code_cell_before_install_or_training():
    notebook, _ = notebook_source(NB56)
    first = "".join(next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")["source"])
    assert "nb56_PCGRAD_RANK_K467.csv" in first
    assert "nb56_fail" in first
    assert "unlink(missing_ok=True)" in first
    assert "pip install" not in first

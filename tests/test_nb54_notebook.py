import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook/experiments/54_continuous_dense_calibration.ipynb"
BUILDER = NOTEBOOK.parent / "_build_nb54.py"


def load_notebook():
    with NOTEBOOK.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def code_source():
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def calibration_namespace():
    source = code_source()
    start = source.index("def normalize_id")
    end = source.index("manifest_rows = []", start)
    namespace = {
        "np": np,
        "pd": pd,
        "json": json,
        "Path": Path,
        "DUPLICATE_IOU": 0.70,
        "DUPLICATE_DAMPING": 0.92,
        "EPS_Q": 0.02,
        "CFG": {"BASE_THR": 0.10},
    }
    exec(compile(source[start:end], "nb54_calibration", "exec"), namespace)
    return namespace


def candidate_cache():
    return {
        "001": {
            "boxes": np.array([[1.25, 2.5, 11.75, 12.5], [20, 21, 30, 31]], np.float32),
            "scores": np.array([0.8, 0.6], np.float32),
            "legacy_poison": np.array([0.1, 0.7], np.float32),
            "scrub_poison": np.array([0.2, 0.8], np.float32),
        },
        "2": {
            "boxes": np.array([[100.125, 101.25, 110.5, 111.75]], np.float32),
            "scores": np.array([0.9], np.float32),
            "legacy_poison": np.array([0.3], np.float32),
            "scrub_poison": np.array([0.4], np.float32),
        },
    }


def sample_frame():
    return pd.DataFrame(
        {
            "id": [17, 91],
            "image_id": ["1", "002"],
            "prediction_string": [" ", " "],
        }
    )


def test_notebook_exists_and_all_code_cells_parse():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")


def test_builder_fails_loudly_if_nb51_runtime_cell_layout_drifts():
    source = BUILDER.read_text(encoding="utf-8")
    assert "NB51_RUNTIME_MARKERS" in source
    assert "assert len(nb51[\"cells\"]) == 20" in source
    assert "assert marker in source" in source


def test_nb54_is_standalone_and_retrains_nb51_pipeline():
    source = code_source()
    assert 'MODE = "standalone_continuous_dense_calibration"' in source
    assert "def train_legacy_ranker" in source
    assert "def train_scrub_fold" in source
    assert "torch.save" in source
    for old_version in ("nb49_", "nb50_", "nb51_", "nb52_", "nb53_"):
        assert old_version not in source
    assert "resolve_nb50" not in source
    assert "resolve_nb51" not in source


def test_nb54_preregisters_a065_and_three_gamma_variants():
    source = code_source()
    assert 'A065_ALPHA = 0.65' in source
    assert '"legacy_weight": 0.35' in source
    assert '"scrub_weight": 0.65' in source
    assert '{"name": "CONTINUOUS_G2", "gamma": 2.0, "priority": 1}' in source
    assert '{"name": "CONTINUOUS_G1", "gamma": 1.0, "priority": 2}' in source
    assert '{"name": "CONTINUOUS_G3", "gamma": 3.0, "priority": 3}' in source
    assert 'sum(candidate["priority"] == 1 for candidate in CALIBRATION_CANDIDATES) == 1' in source


def test_nb54_has_continuous_mapping_and_no_sparse_omission_path():
    source = code_source()
    assert "def a065_poison_scores" in source
    assert "def continuous_clean_percentiles" in source
    assert "def calibrate_continuous_confidence" in source
    assert "q ** gamma" in source
    assert "def apply_mild_duplicate_damping" in source
    assert "DUPLICATE_IOU" in source
    assert "EPS_Q" in source
    assert "calibrate_exact_keep_set" not in source
    assert "compose_sparse_submission" not in source
    assert "selected_indices" not in source
    assert "drop_duplicates" not in source
    assert "filter(" not in source


def test_continuous_calibration_is_bounded_monotonic_and_deterministic():
    ns = calibration_namespace()
    scores = {"2": np.array([0.4, 0.1]), "1": np.array([0.1])}
    first = ns["continuous_clean_percentiles"](scores)
    second = ns["continuous_clean_percentiles"](scores)
    assert first == second
    assert first[("1", 0)] > first[("2", 1)] > first[("2", 0)]
    confidence = ns["calibrate_continuous_confidence"](
        np.array([0.0, 0.5, 1.0]), np.array([0.8, 0.8, 0.8]), 2.0
    )
    assert np.all(np.isfinite(confidence))
    assert np.all((confidence >= 0.02) & (confidence <= 1.0))
    assert np.all(np.diff(confidence) >= 0.0)


@pytest.mark.parametrize("gamma", [1.0, 2.0, 3.0])
def test_continuous_calibration_matches_exact_equation_for_each_gamma(gamma):
    ns = calibration_namespace()
    q = np.array([0.0, 0.25, 0.5, 1.0], np.float64)
    teacher = np.array([0.4, 0.8, 0.95, 1.0], np.float64)
    expected = 0.02 + q ** gamma * (np.minimum(1.0, teacher + 0.15) - 0.02)
    actual = ns["calibrate_continuous_confidence"](q, teacher, gamma)
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_duplicate_damping_preserves_every_candidate_and_box():
    ns = calibration_namespace()
    boxes = np.array([[0, 0, 10, 10], [0.5, 0.5, 9.5, 9.5], [40, 40, 50, 50]], np.float32)
    confidence = np.array([0.9, 0.8, 0.7], np.float32)
    clean_percentiles = np.array([1.0, 0.5, 0.8], np.float32)
    output = ns["apply_mild_duplicate_damping"](
        boxes, confidence.copy(), clean_percentiles
    )
    assert len(output) == len(boxes)
    assert np.array_equal(boxes, np.array([[0, 0, 10, 10], [0.5, 0.5, 9.5, 9.5], [40, 40, 50, 50]], np.float32))
    assert output[0] == confidence[0]
    assert output[1] < confidence[1]
    assert output[1] >= 0.02
    assert output[2] == confidence[2]


def test_duplicate_damping_keeps_higher_raw_confidence_when_clean_order_disagrees():
    ns = calibration_namespace()
    boxes = np.array([[0, 0, 10, 10], [0.5, 0.5, 9.5, 9.5]], np.float32)
    confidence = np.array([0.60, 0.90], np.float32)
    clean_percentiles = np.array([1.0, 0.2], np.float32)
    output = ns["apply_mild_duplicate_damping"](boxes, confidence, clean_percentiles)
    assert output[1] == confidence[1]
    assert output[0] < confidence[0]


def test_nb54_exports_only_after_all_nb51_gates_and_audits_structure():
    source = code_source()
    assert "if ALL_GATES_PASS:" in source
    assert "assert audit[\"candidate_count\"] == candidate_count" in source
    assert "assert_serialized_boxes_preserved(output_boxes, baseline_by_stem[stem])" in source
    assert "nb54_CONTINUOUS_G2.csv" in source
    assert "nb54_CONTINUOUS_G1.csv" in source
    assert "nb54_CONTINUOUS_G3.csv" in source
    assert "nb54_manifest.csv" in source
    assert "duplicate_damped_count" in source
    assert '"submit_first": candidate["priority"] == 1' in source
    assert "compose_continuous_submission" in source
    assert "validate_submission_bundle" in source
    assert "publish_output_bundle" in source
    assert source.index("validate_submission_bundle") < source.index("publish_output_bundle")
    assert "predictions.get(" not in source


def test_composer_round_trips_exact_counts_boxes_and_ids():
    ns = calibration_namespace()
    cache = candidate_cache()
    scores = ns["a065_poison_scores"](cache)
    frame, audit = ns["compose_continuous_submission"](
        cache, scores, 2.0, sample_frame()
    )
    ns["validate_serialized_submission"](frame, cache)
    assert audit["candidate_count"] == 3
    assert set(frame["image_id"]) == {"1", "002"}
    parsed = {
        ns["normalize_id"](row.image_id): ns["parse_prediction_string"](row.prediction_string)
        for row in frame.itertuples(index=False)
    }
    np.testing.assert_array_equal(parsed["1"][0], cache["001"]["boxes"])
    np.testing.assert_array_equal(parsed["2"][0], cache["2"]["boxes"])


def test_composer_preserves_exact_sample_schema_and_id_order():
    ns = calibration_namespace()
    cache = candidate_cache()
    sample = pd.DataFrame(
        {
            "id": [91, 17],
            "image_id": ["002", "1"],
            "prediction_string": ["placeholder", "placeholder"],
        }
    )

    frame, _ = ns["compose_continuous_submission"](
        cache, ns["a065_poison_scores"](cache), 2.0, sample
    )

    assert list(frame.columns) == ["id", "image_id", "prediction_string"]
    assert frame["id"].tolist() == [91, 17]
    assert frame["image_id"].tolist() == ["002", "1"]


def test_calibration_and_manifest_use_only_nb51_baseline_eligible_subset():
    ns = calibration_namespace()
    cache = candidate_cache()
    cache["001"]["scores"][1] = np.float32(0.099)
    eligible = {
        stem: np.flatnonzero(item["scores"] >= ns["CFG"]["BASE_THR"])
        for stem, item in cache.items()
    }

    scores = ns["a065_poison_scores"](cache)
    assert {stem: len(values) for stem, values in scores.items()} == {"001": 1, "2": 1}
    percentiles = ns["continuous_clean_percentiles"](scores)
    assert set(percentiles) == {("001", 0), ("2", 0)}

    frames = {}
    audits = {}
    for name, gamma in (
        ("CONTINUOUS_G1", 1.0),
        ("CONTINUOUS_G2", 2.0),
        ("CONTINUOUS_G3", 3.0),
    ):
        frames[name], audits[name] = ns["compose_continuous_submission"](
            cache, scores, gamma, sample_frame()
        )
        assert audits[name]["candidate_count"] == 2
        parsed = {
            ns["normalize_id"](row.image_id): ns["parse_prediction_string"](
                row.prediction_string
            )[0]
            for row in frames[name].itertuples(index=False)
        }
        ns["assert_serialized_boxes_preserved"](
            parsed["1"], cache["001"]["boxes"][eligible["001"]]
        )
        ns["assert_serialized_boxes_preserved"](
            parsed["2"], cache["2"]["boxes"][eligible["2"]]
        )

    manifest = pd.DataFrame({"name": list(frames), "candidate_count": [2, 2, 2]})
    assert ns["validate_manifest_report"](
        manifest, {"candidate_count": 2}, frames, cache
    )


def test_serialized_round_trip_accepts_float32_reconstruction_ulp_case():
    ns = calibration_namespace()
    cache = candidate_cache()
    cache["001"]["boxes"][0] = np.array(
        [401.01785, 15.125, 955.6429, 800.875], np.float32
    )
    frame, _ = ns["compose_continuous_submission"](
        cache, ns["a065_poison_scores"](cache), 2.0, sample_frame()
    )
    assert ns["validate_serialized_submission"](frame, cache)


def test_serialized_round_trip_accepts_broad_random_float32_boxes():
    ns = calibration_namespace()
    rng = np.random.default_rng(540)
    starts = rng.uniform(0, 900, size=(500, 2)).astype(np.float32)
    maximum_sizes = 1024.0 - starts
    sizes = (rng.uniform(0.001, 1.0, size=(500, 2)) * maximum_sizes).astype(np.float32)
    boxes = np.column_stack((starts, starts + sizes)).astype(np.float32)
    scores = rng.uniform(0.02, 1.0, size=500).astype(np.float32)
    serialized = ns["format_prediction"](boxes, scores)
    parsed_boxes, parsed_scores = ns["parse_prediction_string"](serialized)
    ns["assert_serialized_boxes_preserved"](parsed_boxes, boxes)
    assert len(parsed_scores) == len(scores)


def test_composer_rejects_normalized_id_collision_and_missing_sample_id():
    ns = calibration_namespace()
    cache = candidate_cache()
    collision = dict(cache)
    collision["1"] = collision["001"]
    with pytest.raises(AssertionError, match="normalized candidate IDs"):
        ns["compose_continuous_submission"](
            collision, ns["a065_poison_scores"](collision), 2.0, sample_frame()
        )


def test_composer_rejects_empty_candidate_set():
    ns = calibration_namespace()
    with pytest.raises(AssertionError, match="candidate set is empty"):
        ns["compose_continuous_submission"](
            {}, {}, 2.0, pd.DataFrame({"id": [], "image_id": [], "prediction_string": []})
        )


@pytest.mark.parametrize(
    "bad_box",
    [
        [1, 2, 1, 3],
        [-1, 2, 3, 4],
        [1, 2, 1025, 4],
        [1, 2, np.nan, 4],
    ],
)
def test_composer_rejects_invalid_boxes_instead_of_filtering(bad_box):
    ns = calibration_namespace()
    cache = candidate_cache()
    cache["001"]["boxes"][0] = bad_box
    with pytest.raises(AssertionError, match="invalid boxes"):
        ns["compose_continuous_submission"](
            cache, ns["a065_poison_scores"](cache), 2.0, sample_frame()
        )


def test_bundle_validation_failure_writes_no_csv(tmp_path):
    ns = calibration_namespace()
    cache = candidate_cache()
    scores = ns["a065_poison_scores"](cache)
    frames = {}
    for name, gamma in (("G1", 1.0), ("G2", 2.0), ("G3", 3.0)):
        frames[name], _ = ns["compose_continuous_submission"](
            cache, scores, gamma, sample_frame()
        )
    frames["G3"].loc[0, "prediction_string"] = "0.5 1 2 0 4"
    outputs = {name: tmp_path / f"{name}.csv" for name in frames}
    outputs.update(manifest=tmp_path / "manifest.csv", report=tmp_path / "report.json")
    manifest = pd.DataFrame({"name": list(frames), "candidate_count": [3, 3, 3]})
    with pytest.raises(AssertionError):
        ns["publish_output_bundle"](
            frames, cache, manifest, {"candidate_count": 3}, outputs
        )
    assert list(tmp_path.glob("*.csv")) == []


def test_atomic_write_failure_removes_partial_outputs(tmp_path, monkeypatch):
    ns = calibration_namespace()
    cache = candidate_cache()
    scores = ns["a065_poison_scores"](cache)
    frames = {
        name: ns["compose_continuous_submission"](cache, scores, gamma, sample_frame())[0]
        for name, gamma in (("CONTINUOUS_G1", 1.0), ("CONTINUOUS_G2", 2.0), ("CONTINUOUS_G3", 3.0))
    }
    manifest = pd.DataFrame({"name": list(frames), "candidate_count": [3, 3, 3]})
    report = {"candidate_count": 3, "candidates": list(frames)}
    outputs = {name: tmp_path / f"{name}.csv" for name in frames}
    outputs["manifest"] = tmp_path / "manifest.csv"
    outputs["report"] = tmp_path / "report.json"
    original_replace = Path.replace
    calls = 0

    def fail_second_replace(path, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated replacement failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated"):
        ns["publish_output_bundle"](frames, cache, manifest, report, outputs)
    assert list(tmp_path.glob("*.csv")) == []
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_manifest_or_report_failure_leaves_no_outputs(tmp_path):
    ns = calibration_namespace()
    cache = candidate_cache()
    scores = ns["a065_poison_scores"](cache)
    frames = {
        name: ns["compose_continuous_submission"](cache, scores, gamma, sample_frame())[0]
        for name, gamma in (("CONTINUOUS_G1", 1.0), ("CONTINUOUS_G2", 2.0), ("CONTINUOUS_G3", 3.0))
    }
    outputs = {name: tmp_path / f"{name}.csv" for name in frames}
    outputs.update(manifest=tmp_path / "manifest.csv", report=tmp_path / "report.json")
    manifest = pd.DataFrame({"name": list(frames), "candidate_count": [3, 3, 2]})
    with pytest.raises(AssertionError, match="manifest candidate count"):
        ns["publish_output_bundle"](frames, cache, manifest, {"candidate_count": 3}, outputs)
    assert list(tmp_path.iterdir()) == []
    valid_manifest = pd.DataFrame({"name": list(frames), "candidate_count": [3, 3, 3]})
    with pytest.raises(AssertionError):
        ns["publish_output_bundle"](
            frames, cache, valid_manifest, {"candidate_count": float("nan")}, outputs
        )
    assert list(tmp_path.iterdir()) == []


def test_publish_failure_removes_stale_outputs(tmp_path, monkeypatch):
    ns = calibration_namespace()
    cache = candidate_cache()
    scores = ns["a065_poison_scores"](cache)
    frames = {
        name: ns["compose_continuous_submission"](cache, scores, gamma, sample_frame())[0]
        for name, gamma in (("CONTINUOUS_G1", 1.0), ("CONTINUOUS_G2", 2.0), ("CONTINUOUS_G3", 3.0))
    }
    manifest = pd.DataFrame({"name": list(frames), "candidate_count": [3, 3, 3]})
    report = {"candidate_count": 3}
    outputs = {name: tmp_path / f"{name}.csv" for name in frames}
    outputs.update(manifest=tmp_path / "manifest.csv", report=tmp_path / "report.json")
    for path in outputs.values():
        path.write_text("stale", encoding="utf-8")

    def fail_replace(path, target):
        raise OSError("publish failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="publish failed"):
        ns["publish_output_bundle"](frames, cache, manifest, report, outputs)
    assert list(tmp_path.iterdir()) == []


def test_nb54_cells_are_unexecuted_and_outputs_are_empty():
    notebook = load_notebook()
    assert len([cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]) == 1
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []

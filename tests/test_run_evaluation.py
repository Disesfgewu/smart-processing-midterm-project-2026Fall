"""Tests for #10's scoring/export wiring. Pure path only (score_and_export):
no PIL, no model -- the full run_evaluation() I/O wrapper is exercised by
the opt-in real test in test_run_evaluation_real.py instead."""
import json

from smart_processing.evaluation_dataset import EvaluationEntry, GroundTruthBox
from smart_processing.prompt_evaluation import PredictedBox
from smart_processing.run_evaluation import export_results, score_and_export


def _entry(category="occlusion", prompt_style="basic_noun", prompt="a cat", gt=None):
    return EvaluationEntry(
        image="evaluation/x.jpg",
        image_license="CC0 (verified)",
        category=category,
        prompt_style=prompt_style,
        prompt=prompt,
        ground_truth_boxes=gt or [],
    )


def test_score_and_export_reports_scores_as_plain_dicts():
    entry = _entry(gt=[GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))])
    preds = [PredictedBox(label="a cat", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    result = score_and_export([(entry, preds)])
    assert result["scores"]["overall"]["true_positives"] == 1
    assert result["scores"]["by_category"]["occlusion"]["true_positives"] == 1
    assert result["scores"]["by_prompt_style"]["basic_noun"]["true_positives"] == 1


def test_score_and_export_omits_perfect_entries_from_failed_examples():
    entry = _entry(gt=[GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))])
    preds = [PredictedBox(label="a cat", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    result = score_and_export([(entry, preds)])
    assert result["failed_examples"] == []


def test_score_and_export_includes_failed_entries_with_config_and_boxes():
    gt = [GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    entry = _entry(gt=gt, prompt="a cat")
    preds = [PredictedBox(label="a dog", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    result = score_and_export([(entry, preds)], threshold=0.25)
    assert len(result["failed_examples"]) == 1
    failed = result["failed_examples"][0]
    assert failed["image"] == "evaluation/x.jpg"
    assert failed["config"] == {"threshold": 0.25}
    assert failed["ground_truth_boxes"] == [{"label": "cat", "box_xyxy": [0.0, 0.0, 10.0, 10.0]}]
    assert failed["predicted_boxes"][0]["label"] == "a dog"
    assert failed["score"]["false_positives"] == 1
    assert failed["score"]["false_negatives"] == 1


def test_export_results_writes_valid_json(tmp_path):
    entry = _entry(gt=[])
    result = score_and_export([(entry, [])])
    out_path = tmp_path / "results.json"
    export_results(result, str(out_path))
    loaded = json.loads(out_path.read_text())
    assert loaded == result

"""Tests for #10's scorer against docs/evaluation_methodology.md's spec."""
import pytest

from smart_processing.evaluation_dataset import EvaluationEntry, GroundTruthBox
from smart_processing.prompt_evaluation import PredictedBox, iou, score_dataset, score_entry


def _entry(ground_truth_boxes, prompt="a cat"):
    return EvaluationEntry(
        image="x.jpg",
        image_license="CC0 (verified)",
        category="similar_objects",
        prompt_style="basic_noun",
        prompt=prompt,
        ground_truth_boxes=ground_truth_boxes,
    )


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection 5x10=50, union 100+100-50=150
    assert iou(a, b) == pytest.approx(50.0 / 150.0)


def test_score_entry_exact_match_is_one_true_positive():
    gt = [GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [PredictedBox(label="a cat", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    result = score_entry(_entry(gt), pred)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 0, 0)
    assert result.precision == result.recall == result.f1 == pytest.approx(1.0)


def test_score_entry_label_mismatch_is_false_positive():
    gt = [GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [PredictedBox(label="a dog", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    result = score_entry(_entry(gt), pred)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 1, 1)


def test_score_entry_low_iou_is_false_positive_and_false_negative():
    gt = [GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [PredictedBox(label="a cat", score=0.9, box_xyxy=(50.0, 50.0, 60.0, 60.0))]
    result = score_entry(_entry(gt), pred)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 1, 1)


def test_score_entry_duplicate_predictions_only_one_true_positive():
    """A second prediction overlapping an already-matched box is a false
    positive, not a second true positive (prevents inflating recall)."""
    gt = [GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred = [
        PredictedBox(label="a cat", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0)),
        PredictedBox(label="a cat", score=0.8, box_xyxy=(0.0, 0.0, 10.0, 10.0)),
    ]
    result = score_entry(_entry(gt), pred)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)


def test_score_entry_greedy_prefers_higher_score_order():
    """Two predictions both overlap the single ground-truth box; the
    higher-scoring one should claim the match, not whichever is listed
    first in the predictions list."""
    gt = [GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    low_first = [
        PredictedBox(label="a cat", score=0.4, box_xyxy=(0.0, 0.0, 10.0, 10.0)),
        PredictedBox(label="a cat", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0)),
    ]
    result = score_entry(_entry(gt), low_first)
    assert (result.true_positives, result.false_positives) == (1, 1)


def test_score_entry_empty_ground_truth_and_empty_predictions_is_perfect():
    result = score_entry(_entry([]), [])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 0, 0)
    assert result.precision == result.recall == result.f1 == pytest.approx(1.0)


def test_score_entry_empty_ground_truth_with_prediction_is_false_positive():
    pred = [PredictedBox(label="a cat", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    result = score_entry(_entry([]), pred)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 1, 0)
    assert result.precision == 0.0


def test_score_dataset_micro_averages_across_entries():
    gt1 = [GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    gt2 = [
        GroundTruthBox(label="dog", box_xyxy=(0.0, 0.0, 10.0, 10.0)),
        GroundTruthBox(label="dog", box_xyxy=(20.0, 20.0, 30.0, 30.0)),
    ]
    entry1 = _entry(gt1, prompt="a cat")
    entry2 = _entry(gt2, prompt="a dog")
    pred1 = [PredictedBox(label="a cat", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    pred2 = [PredictedBox(label="a dog", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    result = score_dataset([(entry1, pred1), (entry2, pred2)])
    overall = result["overall"]
    # 2 true positives total, 0 false positives, 1 false negative (gt2's 2nd box unmatched)
    assert (overall.true_positives, overall.false_positives, overall.false_negatives) == (2, 0, 1)


def test_score_dataset_breaks_down_by_category_and_prompt_style():
    gt = [GroundTruthBox(label="cat", box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    entry = EvaluationEntry(
        image="x.jpg",
        image_license="CC0 (verified)",
        category="occlusion",
        prompt_style="fine_grained",
        prompt="a cat",
        ground_truth_boxes=gt,
    )
    pred = [PredictedBox(label="a cat", score=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0))]
    result = score_dataset([(entry, pred)])
    assert set(result["by_category"]) == {"occlusion"}
    assert set(result["by_prompt_style"]) == {"fine_grained"}
    assert result["by_category"]["occlusion"].true_positives == 1

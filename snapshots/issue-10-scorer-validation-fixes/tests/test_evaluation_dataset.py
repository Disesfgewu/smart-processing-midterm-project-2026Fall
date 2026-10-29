"""Fast tests (no model, no network): evaluation dataset schema/loader."""
import json

import pytest

from smart_processing.evaluation_dataset import (
    EvaluationEntry,
    GroundTruthBox,
    load_annotations,
    summarize,
)


def test_ground_truth_box_rejects_non_increasing_coords():
    with pytest.raises(ValueError):
        GroundTruthBox(label="cat", box_xyxy=(50, 50, 10, 10))


def test_ground_truth_box_rejects_wrong_length_box():
    with pytest.raises(ValueError):
        GroundTruthBox(label="cat", box_xyxy=(1, 2, 3))


def test_ground_truth_box_rejects_negative_coords():
    with pytest.raises(ValueError):
        GroundTruthBox(label="cat", box_xyxy=(-5, 0, 10, 10))


def test_ground_truth_box_rejects_non_finite_coords():
    with pytest.raises(ValueError):
        GroundTruthBox(label="cat", box_xyxy=(0, 0, float("inf"), 10))


def test_evaluation_entry_rejects_blank_image():
    with pytest.raises(ValueError):
        EvaluationEntry(
            image="  ", image_license="CC0", category="small_objects", prompt_style="basic_noun", prompt="a cat"
        )


def test_evaluation_entry_rejects_non_string_image():
    with pytest.raises(ValueError):
        EvaluationEntry(
            image=None, image_license="CC0", category="small_objects", prompt_style="basic_noun", prompt="a cat"
        )


def test_ground_truth_box_rejects_blank_label():
    with pytest.raises(ValueError):
        GroundTruthBox(label="  ", box_xyxy=(0, 0, 10, 10))


def test_evaluation_entry_rejects_invalid_category():
    with pytest.raises(ValueError):
        EvaluationEntry(
            image="x.jpg", image_license="CC0", category="bogus", prompt_style="basic_noun", prompt="a cat"
        )


def test_evaluation_entry_rejects_invalid_prompt_style():
    with pytest.raises(ValueError):
        EvaluationEntry(
            image="x.jpg",
            image_license="CC0",
            category="small_objects",
            prompt_style="bogus",
            prompt="a cat",
        )


def test_evaluation_entry_rejects_blank_license():
    with pytest.raises(ValueError):
        EvaluationEntry(
            image="x.jpg", image_license="  ", category="small_objects", prompt_style="basic_noun", prompt="a cat"
        )


def test_evaluation_entry_accepts_empty_ground_truth_boxes():
    entry = EvaluationEntry(
        image="x.jpg",
        image_license="CC0",
        category="occlusion",
        prompt_style="fine_grained",
        prompt="a flying saucer",
    )
    assert entry.ground_truth_boxes == []


def test_load_annotations_round_trips(tmp_path):
    data = [
        {
            "image": "a.jpg",
            "image_license": "CC BY-SA 4.0",
            "category": "small_objects",
            "prompt_style": "basic_noun",
            "prompt": "a bolt",
            "ground_truth_boxes": [{"label": "bolt", "box_xyxy": [1, 2, 3, 4]}],
        }
    ]
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))

    entries = load_annotations(str(path))

    assert len(entries) == 1
    assert entries[0].prompt == "a bolt"
    assert entries[0].ground_truth_boxes[0].label == "bolt"
    assert entries[0].ground_truth_boxes[0].box_xyxy == (1, 2, 3, 4)


def test_load_annotations_rejects_non_list(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        load_annotations(str(path))


def test_load_annotations_reports_entry_index_on_error(tmp_path):
    data = [{"image": "a.jpg"}]  # missing required fields
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="entry 0"):
        load_annotations(str(path))


def test_load_annotations_rejects_null_field_as_value_error_not_attribute_error(tmp_path):
    data = [
        {
            "image": None,
            "image_license": "CC0",
            "category": "small_objects",
            "prompt_style": "basic_noun",
            "prompt": "a cat",
            "ground_truth_boxes": [],
        }
    ]
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):  # must not escape as AttributeError
        load_annotations(str(path))


def test_summarize_counts_distinct_images_and_breakdowns():
    entries = [
        EvaluationEntry(
            image="a.jpg", image_license="CC0", category="small_objects", prompt_style="basic_noun", prompt="x"
        ),
        EvaluationEntry(
            image="a.jpg",
            image_license="CC0",
            category="small_objects",
            prompt_style="fine_grained",
            prompt="y",
        ),
        EvaluationEntry(
            image="b.jpg", image_license="CC0", category="occlusion", prompt_style="basic_noun", prompt="z"
        ),
    ]
    summary = summarize(entries)
    assert summary["distinct_images"] == 2
    assert summary["total_entries"] == 3
    assert summary["by_category"] == {"small_objects": 2, "occlusion": 1}
    assert summary["by_prompt_style"] == {"basic_noun": 2, "fine_grained": 1}

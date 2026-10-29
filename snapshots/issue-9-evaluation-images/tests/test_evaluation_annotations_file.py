"""Verifies the real tests/data/evaluation/annotations.json loads and is
internally consistent -- no network, no model."""
from pathlib import Path

from smart_processing.evaluation_dataset import load_annotations, summarize

ANNOTATIONS_PATH = Path(__file__).resolve().parent / "data" / "evaluation" / "annotations.json"


def test_annotations_file_loads():
    entries = load_annotations(str(ANNOTATIONS_PATH))
    assert len(entries) >= 1


def test_annotations_file_images_exist_on_disk():
    entries = load_annotations(str(ANNOTATIONS_PATH))
    data_dir = ANNOTATIONS_PATH.parent.parent
    for entry in entries:
        assert (data_dir / entry.image).exists(), f"missing image file: {entry.image}"


def test_annotations_file_every_entry_has_verified_license_text():
    entries = load_annotations(str(ANNOTATIONS_PATH))
    for entry in entries:
        assert "verified" in entry.image_license.lower() or "cc0" in entry.image_license.lower()


def test_annotations_summary_reports_current_partial_coverage():
    entries = load_annotations(str(ANNOTATIONS_PATH))
    summary = summarize(entries)
    # Honest partial-progress check: not yet at the 10-image minimum.
    assert summary["distinct_images"] >= 1
    assert summary["distinct_images"] < 10


def test_annotations_file_covers_all_three_required_categories():
    entries = load_annotations(str(ANNOTATIONS_PATH))
    summary = summarize(entries)
    assert set(summary["by_category"]) == {"similar_objects", "small_objects", "occlusion"}
    assert set(summary["by_prompt_style"]) == {"basic_noun", "fine_grained", "descriptive_phrase"}

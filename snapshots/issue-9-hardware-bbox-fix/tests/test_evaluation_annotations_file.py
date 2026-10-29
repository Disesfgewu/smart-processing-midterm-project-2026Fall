"""Verifies the real tests/data/evaluation/annotations.json loads and is
internally consistent -- no network, no model."""
from pathlib import Path

import pytest

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


def test_annotations_file_boxes_fit_within_the_actual_loaded_image():
    """Regression test for a real PM-Reviwer finding: a ground-truth box
    for similar_objects_hardware.jpg exceeded the image's real bounds
    (y1=2804 > 2448) because it had been computed in the EXIF-rotated
    display orientation while PIL's Image.open() (no exif_transpose)
    loads the raw, unrotated frame -- a coordinate-space mismatch a
    naive "box_xyxy is internally well-ordered" check doesn't catch.
    Checks every entry's boxes against the image's REAL loaded pixel
    dimensions, not just internal self-consistency."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    entries = load_annotations(str(ANNOTATIONS_PATH))
    data_dir = ANNOTATIONS_PATH.parent.parent
    dims_cache: dict[str, tuple[int, int]] = {}
    for entry in entries:
        if entry.image not in dims_cache:
            with Image.open(data_dir / entry.image) as img:
                dims_cache[entry.image] = img.size  # (width, height), no exif_transpose
        width, height = dims_cache[entry.image]
        for box in entry.ground_truth_boxes:
            x0, y0, x1, y1 = box.box_xyxy
            assert 0 <= x0 < x1 <= width, (
                f"{entry.image}: box {box.box_xyxy} x-range exceeds real image width {width}"
            )
            assert 0 <= y0 < y1 <= height, (
                f"{entry.image}: box {box.box_xyxy} y-range exceeds real image height {height}"
            )

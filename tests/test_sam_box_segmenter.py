"""Fast tests (no model download): input validation only."""
import pytest

from smart_processing.adapters.sam_box_segmenter import validate_segment_inputs
from smart_processing.types import BoundingBox

VALID_BOX = BoundingBox(10, 10, 100, 100)


def test_accepts_valid_box_within_bounds():
    validate_segment_inputs((480, 640), VALID_BOX)  # must not raise


def test_rejects_segment_before_set_image():
    with pytest.raises(RuntimeError):
        validate_segment_inputs(None, VALID_BOX)


def test_rejects_box_exceeding_width():
    with pytest.raises(ValueError):
        validate_segment_inputs((480, 50), VALID_BOX)  # width=50 < box.x1=100


def test_rejects_box_exceeding_height():
    with pytest.raises(ValueError):
        validate_segment_inputs((50, 640), VALID_BOX)  # height=50 < box.y1=100


def test_accepts_box_touching_exact_image_edge():
    edge_box = BoundingBox(0, 0, 640, 480)
    validate_segment_inputs((480, 640), edge_box)  # must not raise

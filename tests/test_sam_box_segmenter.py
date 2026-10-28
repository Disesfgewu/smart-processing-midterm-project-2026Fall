"""Fast tests (no model download): input validation plus set_image()/
segment() state transitions and output construction, using a stub
predictor in place of segment_anything.SamPredictor."""
import numpy as np
import pytest

from smart_processing.adapters.sam_box_segmenter import SamBoxSegmenter, validate_segment_inputs
from smart_processing.types import BoundingBox, SegmentationMask

VALID_BOX = BoundingBox(10, 10, 100, 100)
VALID_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


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


@pytest.mark.parametrize("bad_box", [None, (10, 10, 100, 100), "box", 42])
def test_rejects_non_bounding_box_type(bad_box):
    with pytest.raises(ValueError):
        validate_segment_inputs((480, 640), bad_box)


class _StubPredictor:
    def __init__(self, mask_shape, score):
        self._mask_shape = mask_shape
        self._score = score

    def set_image(self, image):
        pass

    def predict(self, box, multimask_output):
        mask = np.ones((1, *self._mask_shape), dtype=bool)
        return mask, np.array([self._score]), None


def _make_segmenter(mask_shape=(480, 640), score=0.9):
    segmenter = SamBoxSegmenter.__new__(SamBoxSegmenter)
    segmenter._predictor = _StubPredictor(mask_shape, score)
    segmenter.device = "cpu"
    segmenter._image_shape = None
    return segmenter


def test_set_image_records_shape_and_rejects_invalid_image():
    segmenter = _make_segmenter()
    segmenter.set_image(VALID_IMAGE)
    assert segmenter._image_shape == (480, 640)

    with pytest.raises(ValueError):
        segmenter.set_image(np.zeros((480, 640), dtype=np.uint8))  # not HWC-3


def test_segment_returns_aligned_segmentation_mask():
    segmenter = _make_segmenter(mask_shape=(480, 640), score=0.75)
    segmenter.set_image(VALID_IMAGE)
    result = segmenter.segment(VALID_BOX)

    assert isinstance(result, SegmentationMask)
    assert (result.height, result.width) == (480, 640)
    assert result.score == 0.75


@pytest.mark.parametrize("score", [1.5, -0.5])
def test_segment_clamps_out_of_range_score(score):
    segmenter = _make_segmenter(mask_shape=(480, 640), score=score)
    segmenter.set_image(VALID_IMAGE)
    result = segmenter.segment(VALID_BOX)
    assert 0.0 <= result.score <= 1.0


def test_segment_rejects_mask_shape_mismatch():
    # Stub predictor returns a mask shaped for a different image than the
    # one set_image() recorded -- must fail loudly, not silently misalign.
    segmenter = _make_segmenter(mask_shape=(100, 100), score=0.9)
    segmenter.set_image(VALID_IMAGE)  # records (480, 640)
    with pytest.raises(ValueError):
        segmenter.segment(VALID_BOX)

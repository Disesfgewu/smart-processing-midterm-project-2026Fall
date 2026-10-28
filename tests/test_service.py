"""Unit tests for #23's service request-handling logic. No PIL/model here
-- prompts/threshold validation happens before image decoding (see
handle_infer's docstring), so these paths are testable in a PIL-free dev
venv. The full real flow (image decode + real detection + real HTTP
round-trip) is in the opt-in tests/test_service_real.py instead."""
import base64

import pytest

from smart_processing.service import ServiceError, handle_infer, handle_segment
from smart_processing.types import BoundingBox, SegmentationMask


class _NeverCalledDetector:
    def detect(self, image, prompts, threshold=0.1):
        raise AssertionError("detector.detect() should never be reached in this test")


class _NeverCalledSegmenter:
    def set_image(self, image):
        raise AssertionError("segmenter.set_image() should never be reached in this test")

    def segment(self, box):
        raise AssertionError("segmenter.segment() should never be reached in this test")


class _StubSegmenter:
    """Returns a fixed mask without any real model -- used to test the
    success path and response shape without needing a real SAM checkpoint."""

    def __init__(self):
        self.seen_image = None
        self.seen_box = None

    def set_image(self, image):
        self.seen_image = image

    def segment(self, box):
        self.seen_box = box
        import numpy as np

        mask = np.zeros(self.seen_image.shape[:2], dtype=bool)
        mask[0:1, 0:1] = True
        return SegmentationMask(mask=mask, score=0.95)


@pytest.mark.parametrize("bad_prompts", [None, [], "a cat", [123], [""]])
def test_handle_infer_rejects_invalid_prompts(bad_prompts):
    with pytest.raises(ServiceError):
        handle_infer(_NeverCalledDetector(), {"image_base64": "anything", "prompts": bad_prompts})


@pytest.mark.parametrize("bad_threshold", ["0.5", None, True, -0.1, 1.1])
def test_handle_infer_rejects_invalid_threshold(bad_threshold):
    with pytest.raises(ServiceError):
        handle_infer(
            _NeverCalledDetector(),
            {"image_base64": "anything", "prompts": ["a cat"], "threshold": bad_threshold},
        )


def test_handle_infer_defaults_threshold_when_omitted():
    """Threshold is optional; its absence must not itself raise before
    reaching image decoding (which will raise instead, since
    image_base64 here is intentionally invalid -- this test only checks
    that the *threshold* default doesn't block that)."""
    with pytest.raises(ServiceError, match="image_base64"):
        handle_infer(_NeverCalledDetector(), {"image_base64": "not-valid-base64!!", "prompts": ["a cat"]})


def test_handle_infer_rejects_missing_image_base64():
    with pytest.raises(ServiceError):
        handle_infer(_NeverCalledDetector(), {"prompts": ["a cat"]})


def test_handle_infer_rejects_non_base64_image_string():
    with pytest.raises(ServiceError):
        handle_infer(
            _NeverCalledDetector(), {"image_base64": "not valid base64 at all!", "prompts": ["a cat"]}
        )


@pytest.mark.parametrize("bad_body", [[], None, 123, "a string", True])
def test_handle_infer_rejects_non_dict_body(bad_body):
    """PM-Reviwer finding: a syntactically valid JSON body that isn't an
    object (e.g. `[]`, `null`, `123`) made body.get() raise AttributeError,
    an unhandled exception the HTTP layer reported as a 500 instead of the
    400 a malformed request deserves."""
    with pytest.raises(ServiceError, match="JSON object"):
        handle_infer(_NeverCalledDetector(), bad_body)


@pytest.mark.parametrize("bad_body", [[], None, 123, "a string", True])
def test_handle_segment_rejects_non_dict_body(bad_body):
    with pytest.raises(ServiceError, match="JSON object"):
        handle_segment(_NeverCalledSegmenter(), bad_body)


def test_handle_segment_rejects_missing_box_xyxy():
    with pytest.raises(ServiceError, match="box_xyxy"):
        handle_segment(_NeverCalledSegmenter(), {"image_base64": "anything"})


@pytest.mark.parametrize("bad_box", [[1, 2, 3], [1, 2, 3, 4, 5], "a box", None, 123])
def test_handle_segment_rejects_malformed_box_xyxy_shape(bad_box):
    with pytest.raises(ServiceError, match="box_xyxy"):
        handle_segment(_NeverCalledSegmenter(), {"box_xyxy": bad_box})


def test_handle_segment_rejects_invalid_box():
    """box_xyxy must still fail validation (reusing #3's BoundingBox
    checks) even with no image_base64 key at all -- confirms box
    validation, not missing-image handling, is what raises here."""
    bad_box = [10.0, 10.0, 5.0, 5.0]  # x1 < x0, y1 < y0
    with pytest.raises(ServiceError, match="box_xyxy"):
        handle_segment(_NeverCalledSegmenter(), {"box_xyxy": bad_box})


def test_handle_segment_rejects_missing_image_base64():
    with pytest.raises(ServiceError, match="image_base64"):
        handle_segment(_NeverCalledSegmenter(), {"box_xyxy": [1.0, 1.0, 2.0, 2.0]})


def test_handle_segment_rejects_non_base64_image_string():
    with pytest.raises(ServiceError, match="image_base64"):
        handle_segment(
            _NeverCalledSegmenter(), {"box_xyxy": [1.0, 1.0, 2.0, 2.0], "image_base64": "not valid base64!"}
        )


def test_handle_segment_returns_mask_and_score_on_success():
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.full((20, 20, 3), 128, dtype=np.uint8))
    buffer = __import__("io").BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    segmenter = _StubSegmenter()
    result = handle_segment(segmenter, {"image_base64": image_base64, "box_xyxy": [1.0, 1.0, 5.0, 5.0]})

    assert result["image_width"] == 20
    assert result["image_height"] == 20
    assert result["score"] == 0.95
    assert isinstance(result["mask_base64"], str) and result["mask_base64"]
    assert segmenter.seen_box == BoundingBox(1.0, 1.0, 5.0, 5.0)

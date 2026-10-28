"""Contract tests for #3 — no model downloads, no torch/transformers import."""
import numpy as np
import pytest

from smart_processing.types import (
    BackendConfig,
    BoundingBox,
    Detection,
    DetectionResult,
    SegmentationMask,
)


def test_bounding_box_valid():
    box = BoundingBox(10, 20, 30, 40)
    assert box.as_xyxy() == (10, 20, 30, 40)


@pytest.mark.parametrize("x0,y0,x1,y1", [(30, 20, 10, 40), (10, 40, 30, 20), (10, 20, 10, 40)])
def test_bounding_box_rejects_non_increasing_coords(x0, y0, x1, y1):
    with pytest.raises(ValueError):
        BoundingBox(x0, y0, x1, y1)


def test_detection_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        Detection(box=BoundingBox(0, 0, 10, 10), score=1.5, label="cat")


def test_detection_result_rejects_box_exceeding_image_bounds():
    oversized = Detection(box=BoundingBox(0, 0, 999, 999), score=0.5, label="cat")
    with pytest.raises(ValueError):
        DetectionResult(image_width=640, image_height=480, detections=[oversized])


def test_detection_result_accepts_in_bounds_detection():
    d = Detection(box=BoundingBox(0, 0, 100, 100), score=0.9, label="cat")
    result = DetectionResult(image_width=640, image_height=480, detections=[d])
    assert len(result.detections) == 1


def test_segmentation_mask_requires_bool_dtype():
    with pytest.raises(ValueError):
        SegmentationMask(mask=np.zeros((4, 4), dtype=np.uint8), score=0.9)


def test_segmentation_mask_requires_2d():
    with pytest.raises(ValueError):
        SegmentationMask(mask=np.zeros((1, 4, 4), dtype=np.bool_), score=0.9)


def test_segmentation_mask_reports_dims():
    mask = SegmentationMask(mask=np.zeros((480, 640), dtype=np.bool_), score=0.99)
    assert (mask.height, mask.width) == (480, 640)


def test_backend_config_defaults_to_cpu():
    assert BackendConfig().device == "cpu"
    assert BackendConfig(device="mps", model_name="google/owlvit-base-patch32").model_name

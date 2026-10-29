"""Fast tests (no model download): pipeline orchestration, structured
output, and overlay drawing, using stub TextDetector/BoxSegmenter."""
import json

import numpy as np
import pytest

from smart_processing.pipeline import (
    DetectSegmentPipeline,
    PipelineDetection,
    PipelineResult,
    save_overlay_image,
    save_structured_output,
)
from smart_processing.types import BoundingBox, Detection, DetectionResult, SegmentationMask

IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


class _StubDetector:
    def __init__(self, detections, image_width=640, image_height=480):
        self._detections = detections
        self._image_width = image_width
        self._image_height = image_height
        self.calls = []

    def detect(self, image, prompts, threshold=0.1):
        self.calls.append((prompts, threshold))
        return DetectionResult(
            image_width=self._image_width, image_height=self._image_height, detections=self._detections
        )


class _StubSegmenter:
    def __init__(self):
        self.set_image_calls = 0
        self.segment_calls = []

    def set_image(self, image):
        self.set_image_calls += 1

    def segment(self, box):
        self.segment_calls.append(box)
        mask = np.zeros((480, 640), dtype=np.bool_)
        mask[int(box.y0) : int(box.y1), int(box.x0) : int(box.x1)] = True
        return SegmentationMask(mask=mask, score=0.9)


def _detection(x0, y0, x1, y1, label="cat", score=0.8):
    return Detection(box=BoundingBox(x0, y0, x1, y1), score=score, label=label)


def test_run_with_no_detections_skips_segmenter_entirely():
    detector = _StubDetector([])
    segmenter = _StubSegmenter()
    pipeline = DetectSegmentPipeline(detector, segmenter)

    result = pipeline.run(IMAGE, ["a cat"])

    assert result.results == []
    assert (result.image_width, result.image_height) == (640, 480)
    assert segmenter.set_image_calls == 0  # no point calling it with nothing to segment
    assert segmenter.segment_calls == []


def test_run_rejects_detector_image_dim_mismatch():
    # Detector claims a different image size than what was actually passed in.
    detector = _StubDetector([], image_width=320, image_height=240)
    segmenter = _StubSegmenter()
    pipeline = DetectSegmentPipeline(detector, segmenter)

    with pytest.raises(ValueError):
        pipeline.run(IMAGE, ["a cat"])  # IMAGE is 640x480
    assert segmenter.set_image_calls == 0


def test_run_passes_each_detection_box_to_segmenter():
    detections = [_detection(10, 10, 50, 50), _detection(100, 100, 200, 200, label="dog")]
    detector = _StubDetector(detections)
    segmenter = _StubSegmenter()
    pipeline = DetectSegmentPipeline(detector, segmenter)

    result = pipeline.run(IMAGE, ["a cat", "a dog"], threshold=0.3)

    assert detector.calls == [(["a cat", "a dog"], 0.3)]
    assert segmenter.set_image_calls == 1  # set once, not once per detection
    assert segmenter.segment_calls == [d.box for d in detections]
    assert len(result.results) == 2
    assert all(isinstance(r, PipelineDetection) for r in result.results)
    assert result.results[0].detection.label == "cat"
    assert result.results[1].detection.label == "dog"


def test_save_structured_output_writes_expected_json(tmp_path):
    detection = _detection(10, 10, 50, 50, label="cat", score=0.8)
    mask = np.zeros((480, 640), dtype=np.bool_)
    mask[10:50, 10:50] = True
    result = PipelineResult(
        image_width=640,
        image_height=480,
        results=[PipelineDetection(detection=detection, mask=SegmentationMask(mask=mask, score=0.95))],
    )

    out = tmp_path / "result.json"
    save_structured_output(result, str(out))

    payload = json.loads(out.read_text())
    assert payload["image_width"] == 640
    assert payload["image_height"] == 480
    assert payload["detections"][0]["label"] == "cat"
    assert payload["detections"][0]["box_xyxy"] == [10, 10, 50, 50]
    assert payload["detections"][0]["mask_pixel_count"] == 1600  # 40x40


def test_save_structured_output_creates_missing_parent_dirs(tmp_path):
    result = PipelineResult(image_width=640, image_height=480, results=[])
    out = tmp_path / "nested" / "dir" / "result.json"
    save_structured_output(result, str(out))
    assert out.exists()


def test_pipeline_result_rejects_mask_shape_mismatch():
    detection = _detection(10, 10, 50, 50)
    wrong_shape_mask = SegmentationMask(mask=np.zeros((100, 100), dtype=np.bool_), score=0.9)
    with pytest.raises(ValueError):
        PipelineResult(
            image_width=640,
            image_height=480,
            results=[PipelineDetection(detection=detection, mask=wrong_shape_mask)],
        )


def test_pipeline_result_rejects_box_exceeding_image_dims():
    oversized = _detection(10, 10, 999, 999)
    mask = SegmentationMask(mask=np.zeros((480, 640), dtype=np.bool_), score=0.9)
    with pytest.raises(ValueError):
        PipelineResult(
            image_width=640, image_height=480, results=[PipelineDetection(detection=oversized, mask=mask)]
        )


def test_save_overlay_image_requires_pil():
    pytest.importorskip("PIL")
    import tempfile

    from PIL import Image as PILImage

    detection = _detection(10, 10, 50, 50)
    mask = np.zeros((480, 640), dtype=np.bool_)
    mask[10:50, 10:50] = True
    result = PipelineResult(
        image_width=640,
        image_height=480,
        results=[PipelineDetection(detection=detection, mask=SegmentationMask(mask=mask, score=0.9))],
    )

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        save_overlay_image(IMAGE, result, tmp.name)
        saved = np.array(PILImage.open(tmp.name).convert("RGB"))
        assert saved.shape == IMAGE.shape
        assert saved[30, 30].tolist() != [0, 0, 0]  # mask region got colored


def test_save_overlay_image_rejects_image_shape_mismatch():
    pytest.importorskip("PIL")
    wrong_size_image = np.zeros((100, 100, 3), dtype=np.uint8)
    empty_result = PipelineResult(image_width=640, image_height=480, results=[])
    with pytest.raises(ValueError):
        save_overlay_image(wrong_size_image, empty_result, "/tmp/unused.png")


def test_save_overlay_image_creates_missing_parent_dirs(tmp_path):
    pytest.importorskip("PIL")
    empty_result = PipelineResult(image_width=640, image_height=480, results=[])
    out = tmp_path / "nested" / "dir" / "overlay.png"
    save_overlay_image(IMAGE, empty_result, str(out))
    assert out.exists()

"""Fast tests (no model download): counting logic using stub detector/
segmenter, same pattern as tests/test_pipeline.py."""
import numpy as np

from smart_processing.manufacturing_counting import count_detections, run_counting_demo
from smart_processing.pipeline import DetectSegmentPipeline, PipelineDetection, PipelineResult
from smart_processing.types import BoundingBox, Detection, DetectionResult, SegmentationMask

IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def _pipeline_detection(label, score=0.8):
    box = BoundingBox(10, 10, 50, 50)
    mask = np.zeros((480, 640), dtype=np.bool_)
    mask[10:50, 10:50] = True
    return PipelineDetection(
        detection=Detection(box=box, score=score, label=label),
        mask=SegmentationMask(mask=mask, score=0.9),
    )


def test_count_detections_counts_everything_with_no_label_filter():
    result = PipelineResult(
        image_width=640,
        image_height=480,
        results=[_pipeline_detection("a cat"), _pipeline_detection("a dog"), _pipeline_detection("a cat")],
    )
    assert count_detections(result) == 3


def test_count_detections_filters_by_exact_label():
    result = PipelineResult(
        image_width=640,
        image_height=480,
        results=[_pipeline_detection("a cat"), _pipeline_detection("a dog"), _pipeline_detection("a cat")],
    )
    assert count_detections(result, "a cat") == 2
    assert count_detections(result, "a dog") == 1
    assert count_detections(result, "a bird") == 0


def test_count_detections_label_filter_is_case_insensitive():
    result = PipelineResult(image_width=640, image_height=480, results=[_pipeline_detection("A Cat")])
    assert count_detections(result, "a cat") == 1


def test_count_detections_empty_result_is_zero():
    result = PipelineResult(image_width=640, image_height=480, results=[])
    assert count_detections(result) == 0
    assert count_detections(result, "a cat") == 0


class _StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image, prompts, threshold=0.1):
        return DetectionResult(image_width=640, image_height=480, detections=self._detections)


class _StubSegmenter:
    def set_image(self, image):
        pass

    def segment(self, box):
        mask = np.zeros((480, 640), dtype=np.bool_)
        mask[int(box.y0) : int(box.y1), int(box.x0) : int(box.x1)] = True
        return SegmentationMask(mask=mask, score=0.9)


def test_run_counting_demo_reports_per_prompt_counts():
    detections = [
        Detection(box=BoundingBox(10, 10, 50, 50), score=0.9, label="a bolt"),
        Detection(box=BoundingBox(60, 60, 100, 100), score=0.8, label="a bolt"),
        Detection(box=BoundingBox(110, 110, 150, 150), score=0.7, label="a nut"),
    ]
    pipeline = DetectSegmentPipeline(_StubDetector(detections), _StubSegmenter())

    summary = run_counting_demo(pipeline, IMAGE, ["a bolt", "a nut"])

    assert summary["counts"] == {"a bolt": 2, "a nut": 1}
    assert summary["total"] == 3
    assert summary["prompts"] == ["a bolt", "a nut"]
    assert isinstance(summary["result"], PipelineResult)


def test_run_counting_demo_changing_targets_gives_different_counts():
    detections = [
        Detection(box=BoundingBox(10, 10, 50, 50), score=0.9, label="a bolt"),
        Detection(box=BoundingBox(60, 60, 100, 100), score=0.8, label="a bolt"),
    ]
    pipeline = DetectSegmentPipeline(_StubDetector(detections), _StubSegmenter())

    bolt_summary = run_counting_demo(pipeline, IMAGE, ["a bolt"])
    nut_summary = run_counting_demo(pipeline, IMAGE, ["a nut"])  # same scene, different target

    assert bolt_summary["counts"]["a bolt"] == 2
    assert nut_summary["counts"]["a nut"] == 0  # target changed, count reflects the new target

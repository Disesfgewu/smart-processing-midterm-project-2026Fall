"""Unit tests for #11's benchmark wiring error path. Fake detector/segmenter,
no PIL/torch -- the real wiring is exercised by the opt-in
test_run_benchmarks_real.py instead."""
import numpy as np
import pytest

from smart_processing.run_benchmarks import run_pipeline_benchmarks
from smart_processing.types import BoundingBox, Detection, DetectionResult


class _EmptyDetector:
    def detect(self, image, prompts, threshold=0.1):
        return DetectionResult(image_width=image.shape[1], image_height=image.shape[0], detections=[])


class _FakeSegmenter:
    def set_image(self, image):
        pass

    def segment(self, box):
        raise AssertionError("segment() should never be called when there are no detections")


def test_run_pipeline_benchmarks_raises_when_nothing_detected():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        run_pipeline_benchmarks(_EmptyDetector(), _FakeSegmenter(), image, ["a cat"])

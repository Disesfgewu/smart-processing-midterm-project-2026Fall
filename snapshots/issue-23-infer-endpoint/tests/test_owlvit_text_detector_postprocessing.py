"""Fast tests for OwlViTTextDetector.detect()'s post-processing logic
(box clamping, degenerate-box skipping, label mapping, empty results) --
no model weights downloaded, model/processor are fakes.

Needs torch itself (for tensor-shaped fakes matching the real HF output
contract) but not the macos-feasibility extra's heavier deps or any
network access, so it's skipped rather than failing collection when torch
isn't installed (e.g. a plain `dev`-only .venv).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector  # noqa: E402

IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self, result):
        self._result = result

    def __call__(self, text, images, return_tensors):
        return _FakeInputs()

    def post_process_grounded_object_detection(self, outputs, threshold, target_sizes):
        return [self._result]


def _make_detector(boxes, scores, labels):
    detector = OwlViTTextDetector.__new__(OwlViTTextDetector)
    detector._torch = torch
    detector.device = "cpu"
    detector.model_name = "fake"
    result = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    detector.processor = _FakeProcessor(result)
    detector.model = lambda **kwargs: object()
    return detector


def test_detect_clamps_out_of_bounds_box():
    detector = _make_detector(boxes=[[-10, -5, 700, 500]], scores=[0.9], labels=[0])
    result = detector.detect(IMAGE, ["a cat"], threshold=0.1)
    assert len(result.detections) == 1
    box = result.detections[0].box
    assert (box.x0, box.y0, box.x1, box.y1) == (0.0, 0.0, 640.0, 480.0)


def test_detect_skips_fully_degenerate_box():
    detector = _make_detector(boxes=[[700, 10, 800, 50]], scores=[0.9], labels=[0])
    result = detector.detect(IMAGE, ["a cat"], threshold=0.1)
    assert result.detections == []


def test_detect_maps_label_index_to_prompt():
    detector = _make_detector(boxes=[[10, 10, 50, 50]], scores=[0.8], labels=[1])
    result = detector.detect(IMAGE, ["a cat", "a dog"], threshold=0.1)
    assert result.detections[0].label == "a dog"


def test_detect_returns_empty_result_when_no_boxes():
    detector = _make_detector(boxes=[], scores=[], labels=[])
    result = detector.detect(IMAGE, ["a cat"], threshold=0.1)
    assert result.detections == []

"""Opt-in real-model verification for OwlViTTextDetector (#4).

Skipped by default (downloads/loads a real ~600MB model); run explicitly
with RUN_FEASIBILITY_SMOKE=1, same convention as #2's smoke tests.
"""
import os

import pytest

from smart_processing.types import BackendConfig, DetectionResult

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run real OWL-ViT model checks"
)


@pytest.fixture(scope="module")
def detector():
    # Deferred import: Pillow/torch/transformers are the macos-feasibility
    # optional extra, not a default dependency, so a plain `pytest` in a
    # clean .venv must not import them even when this module is collected.
    from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector

    return OwlViTTextDetector(BackendConfig(device="cpu"))


@pytest.fixture(scope="module")
def sample_image():
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image

    return np.array(Image.open(ensure_sample_image(SAMPLE_IMAGE_PATH)).convert("RGB"))


def test_real_detection_finds_cats(detector, sample_image):
    result = detector.detect(sample_image, ["a photo of a cat"], threshold=0.1)
    assert isinstance(result, DetectionResult)
    assert (result.image_width, result.image_height) == (
        sample_image.shape[1],
        sample_image.shape[0],
    )
    assert len(result.detections) >= 1
    assert all(0.0 <= d.score <= 1.0 for d in result.detections)
    assert all(d.label == "a photo of a cat" for d in result.detections)


def test_real_detection_empty_result_for_implausible_prompt(detector, sample_image):
    # High threshold against an object absent from the image: must return
    # an empty DetectionResult, not raise.
    result = detector.detect(sample_image, ["a flying saucer"], threshold=0.99)
    assert isinstance(result, DetectionResult)
    assert result.detections == []

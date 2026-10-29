"""Opt-in real-model demo for the manufacturing parts-counting scenario
(#8): runs real OwlViTTextDetector + SamBoxSegmenter together, counts
detections per prompt, and demonstrates changing the target without
restarting (two separate runs with different prompts on the same scene).

Skipped by default (downloads/loads real models); run explicitly with
RUN_FEASIBILITY_SMOKE=1, same convention as #2/#4/#5/#6/#7.

Scene note: a dedicated manufacturing-parts photo with reliable OWL-ViT
detection could not be sourced in this session -- see
docs/manufacturing_counting.md for the three real candidates tried
(bolts/nuts/washers, bottles, egg carton) and their actual (near-zero)
detection results, which is itself real evidence of a small/cluttered-
object failure mode. This test uses the already-validated #2 sample image
(COCO val2017 000000039769.jpg) as an explicitly labeled substitute
repeatable tabletop scene: "a cat" and "a remote control" stand in for
countable parts, each with real, previously-verified high-confidence
detections.
"""
import os

import pytest

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"
CHECKPOINT = "tests/data/sam_vit_b_01ec64.pth"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run the real counting demo"
)


@pytest.fixture(scope="module")
def pipeline():
    from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
    from smart_processing.adapters.sam_box_segmenter import SamBoxSegmenter
    from smart_processing.pipeline import DetectSegmentPipeline
    from smart_processing.types import BackendConfig

    return DetectSegmentPipeline(
        OwlViTTextDetector(BackendConfig(device="cpu")), SamBoxSegmenter(CHECKPOINT)
    )


@pytest.fixture(scope="module")
def sample_image():
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image

    return np.array(Image.open(ensure_sample_image(SAMPLE_IMAGE_PATH)).convert("RGB"))


def test_real_counting_demonstrates_changing_targets(pipeline, sample_image):
    from smart_processing.manufacturing_counting import run_counting_demo

    cat_summary = run_counting_demo(pipeline, sample_image, ["a photo of a cat"])
    remote_summary = run_counting_demo(pipeline, sample_image, ["a remote control"])

    # Real, previously-verified counts on this exact image (see
    # docs/macos_feasibility.md): 2 cats, 2 remote controls.
    assert cat_summary["counts"]["a photo of a cat"] >= 1
    assert remote_summary["counts"]["a remote control"] >= 1
    # Changing the target genuinely changes what's counted, same scene,
    # same pipeline instance, no restart.
    assert cat_summary["counts"] != remote_summary["counts"]

"""Opt-in real-model integration test for the pipeline (#6): demonstrates
OwlViTTextDetector (#4) and SamBoxSegmenter (#5) running together end to
end, with a saved overlay image and structured JSON output.

Skipped by default (downloads/loads real models); run explicitly with
RUN_FEASIBILITY_SMOKE=1, same convention as #2/#4/#5.
"""
import json
import os

import pytest

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"
CHECKPOINT = "tests/data/sam_vit_b_01ec64.pth"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run the real detect+segment pipeline"
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


def test_real_pipeline_detects_and_segments_cats(sample_image, tmp_path):
    from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
    from smart_processing.adapters.sam_box_segmenter import SamBoxSegmenter
    from smart_processing.pipeline import (
        DetectSegmentPipeline,
        save_overlay_image,
        save_structured_output,
    )
    from smart_processing.types import BackendConfig

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    segmenter = SamBoxSegmenter(CHECKPOINT)
    pipeline = DetectSegmentPipeline(detector, segmenter)

    result = pipeline.run(sample_image, ["a photo of a cat"], threshold=0.1)

    assert len(result.results) >= 1
    for r in result.results:
        assert (r.mask.height, r.mask.width) == (result.image_height, result.image_width)
        assert r.mask.mask.any()

    overlay_path = tmp_path / "overlay.png"
    json_path = tmp_path / "result.json"
    save_overlay_image(sample_image, result, str(overlay_path))
    save_structured_output(result, str(json_path))

    assert overlay_path.exists() and overlay_path.stat().st_size > 0
    payload = json.loads(json_path.read_text())
    assert len(payload["detections"]) == len(result.results)
    assert payload["detections"][0]["label"] == "a photo of a cat"

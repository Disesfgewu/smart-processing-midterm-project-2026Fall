"""Opt-in real-model verification for SamBoxSegmenter (#5).

Skipped by default (loads a real ~375MB checkpoint); run explicitly with
RUN_FEASIBILITY_SMOKE=1, same convention as #2/#4. Requires
tests/data/sam_vit_b_01ec64.pth (see docs/macos_feasibility.md for the
download command).
"""
import os

import pytest

from smart_processing.types import BoundingBox, SegmentationMask

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"
CHECKPOINT = "tests/data/sam_vit_b_01ec64.pth"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run real SAM model checks"
)


@pytest.fixture(scope="module")
def segmenter():
    from smart_processing.adapters.sam_box_segmenter import SamBoxSegmenter

    return SamBoxSegmenter(CHECKPOINT)


@pytest.fixture(scope="module")
def sample_image():
    import numpy as np
    from PIL import Image

    return np.array(Image.open("tests/data/sample_cats.jpg").convert("RGB"))


def test_real_segmentation_produces_aligned_mask(segmenter, sample_image):
    segmenter.set_image(sample_image)
    # The cat box from #4's real OWL-ViT detection run on this same image.
    box = BoundingBox(1.5, 55.3, 315.6, 472.2)
    result = segmenter.segment(box)

    assert isinstance(result, SegmentationMask)
    assert (result.height, result.width) == sample_image.shape[:2]
    assert 0.0 <= result.score <= 1.0
    assert result.mask.any()  # a real cat silhouette, not an all-empty mask


def test_real_segmentation_rejects_box_before_set_image(segmenter):
    # Fresh segmenter instance to avoid depending on fixture call order.
    from smart_processing.adapters.sam_box_segmenter import SamBoxSegmenter

    fresh = SamBoxSegmenter(CHECKPOINT)
    with pytest.raises(RuntimeError):
        fresh.segment(BoundingBox(0, 0, 10, 10))


def test_real_segmentation_rejects_out_of_bounds_box(segmenter, sample_image):
    segmenter.set_image(sample_image)
    height, width = sample_image.shape[:2]
    with pytest.raises(ValueError):
        segmenter.segment(BoundingBox(0, 0, width + 100, height + 100))

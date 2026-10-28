"""Opt-in real-model benchmark run for #11, wiring the harness (#45) to the
real OwlViTTextDetector/SamBoxSegmenter/DetectSegmentPipeline (#4/#5/#6).

Skipped by default (downloads/loads real models); run explicitly with
RUN_FEASIBILITY_SMOKE=1, same convention as #2/#4/#5/#6.
"""
import os

import pytest

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"
CHECKPOINT = "tests/data/sam_vit_b_01ec64.pth"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run the real pipeline benchmarks"
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


def test_real_pipeline_benchmarks_produce_real_timings(sample_image, tmp_path):
    from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
    from smart_processing.adapters.sam_box_segmenter import SamBoxSegmenter
    from smart_processing.run_benchmarks import export_benchmark_results, run_pipeline_benchmarks
    from smart_processing.types import BackendConfig

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    segmenter = SamBoxSegmenter(CHECKPOINT)

    results = run_pipeline_benchmarks(
        detector, segmenter, sample_image, ["a photo of a cat"], warmup=1, samples=3
    )

    assert set(results) == {"detection", "segmentation", "end_to_end"}
    for name, result in results.items():
        assert result.latency.mean_seconds > 0, f"{name} reported zero latency"
        assert len(result.latency.samples) == 3
        assert result.platform_info.device == "cpu"
        # end_to_end composes detection+segmentation, so it must not be
        # faster than detection alone -- a real sanity check on the wiring,
        # not just "did it run."
    assert results["end_to_end"].latency.mean_seconds >= results["detection"].latency.mean_seconds * 0.5

    out_path = tmp_path / "benchmark_results.json"
    export_benchmark_results(results, str(out_path))
    assert out_path.exists() and out_path.stat().st_size > 0

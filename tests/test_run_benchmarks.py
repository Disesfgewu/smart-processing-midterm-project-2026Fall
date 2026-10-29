"""Unit tests for #11's benchmark wiring error path and run-metadata
export. Fake detector/segmenter, no PIL/torch -- the real wiring is
exercised by the opt-in test_run_benchmarks_real.py instead."""
import json

import numpy as np
import pytest

from smart_processing.performance_harness import BenchmarkResult, LatencyStats, PeakRssMeasurement, PlatformInfo
from smart_processing.run_benchmarks import build_run_metadata, export_benchmark_results, run_pipeline_benchmarks
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


class _NamedDetector:
    model_name = "google/owlvit-base-patch32"

    def detect(self, image, prompts, threshold=0.1):
        raise AssertionError("not called in this test")


def test_build_run_metadata_captures_the_actual_run_configuration():
    """PM-Reviwer finding: a run with 0 failures had no record anywhere
    of what was actually run (model, device, threshold, prompts, sample
    counts) -- this is the reproducibility fix."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    metadata = build_run_metadata(
        _NamedDetector(),
        _FakeSegmenter(),
        image,
        ["a cat", "a dog"],
        threshold=0.25,
        device="cpu",
        warmup=3,
        samples=7,
        checkpoint_path="/path/to/sam_vit_b.pth",
    )
    assert metadata == {
        "model_name": "google/owlvit-base-patch32",
        "segmenter_checkpoint_path": "/path/to/sam_vit_b.pth",
        "device": "cpu",
        "threshold": 0.25,
        "prompts": ["a cat", "a dog"],
        "warmup": 3,
        "samples": 7,
        "image_shape": [10, 10, 3],
    }


def test_build_run_metadata_tolerates_a_detector_without_model_name():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    metadata = build_run_metadata(
        _EmptyDetector(), _FakeSegmenter(), image, ["a cat"], 0.1, "cpu", 2, 5, checkpoint_path=None
    )
    assert metadata["model_name"] is None
    assert metadata["segmenter_checkpoint_path"] is None


def _fake_benchmark_result(name: str) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        platform_info=PlatformInfo(system="Darwin", machine="arm64", python_version="3.11.16", device="cpu"),
        latency=LatencyStats(samples=(0.01, 0.02, 0.015), warmup_count=2),
        peak_rss=PeakRssMeasurement(before_bytes=1000, after_bytes=1500),
    )


def test_export_benchmark_results_includes_run_metadata_when_given(tmp_path):
    out_path = tmp_path / "results.json"
    export_benchmark_results(
        {"detection": _fake_benchmark_result("detection")},
        str(out_path),
        run_metadata={"model_name": "m", "device": "cpu"},
    )
    exported = json.loads(out_path.read_text())
    assert exported["run_metadata"] == {"model_name": "m", "device": "cpu"}
    assert exported["detection"]["name"] == "detection"


def test_export_benchmark_results_omits_run_metadata_key_when_not_given(tmp_path):
    """Backward-compatible: existing callers that never pass run_metadata
    get exactly the same export shape as before this fix."""
    out_path = tmp_path / "results.json"
    export_benchmark_results({"detection": _fake_benchmark_result("detection")}, str(out_path))
    exported = json.loads(out_path.read_text())
    assert "run_metadata" not in exported

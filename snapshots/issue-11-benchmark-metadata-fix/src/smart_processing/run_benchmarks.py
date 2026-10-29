"""Wires #11's measurement harness to the real detection/segmentation/
pipeline calls (#4/#5/#6), producing actual macOS latency/FPS/memory
numbers -- never fabricated, never claimed as Jetson results. Heavy
imports (PIL/torch) stay deferred so importing this module never
requires the macos-feasibility optional extra."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from smart_processing.performance_harness import BenchmarkResult, run_benchmark


def build_run_metadata(
    detector,
    segmenter,
    image,
    prompts: list[str],
    threshold: float,
    device: str,
    warmup: int,
    samples: int,
    checkpoint_path: str | None = None,
) -> dict:
    """Captures the actual run configuration for reproducibility (PM-
    Reviwer finding: platform_info alone -- system/machine/python/
    device -- doesn't record model identity, image identity, prompts,
    threshold, or sample/warmup counts, so two exported results can't
    be told apart or reproduced without external notes). Reads only
    attributes the adapters already expose (`detector.model_name`);
    `checkpoint_path` is accepted explicitly since SamBoxSegmenter
    doesn't retain its own checkpoint path as a public attribute, and
    this module shouldn't reach into its private state to get one."""
    return {
        "model_name": getattr(detector, "model_name", None),
        "segmenter_checkpoint_path": checkpoint_path,
        "device": device,
        "threshold": threshold,
        "prompts": list(prompts),
        "warmup": warmup,
        "samples": samples,
        "image_shape": list(image.shape),
    }


def run_pipeline_benchmarks(
    detector,
    segmenter,
    image,
    prompts: list[str],
    threshold: float = 0.1,
    device: str = "cpu",
    warmup: int = 2,
    samples: int = 5,
) -> dict[str, BenchmarkResult]:
    """Benchmarks detection, segmentation, and the full pipeline as three
    separate named stages, per #11's acceptance criteria ("measure
    detection, segmentation and end-to-end latency, FPS and memory
    separately"). Detection and end-to-end each re-run inference for every
    sample (the real cost a caller pays); segmentation is benchmarked on a
    fixed box from one real detection run, isolating SAM's own cost from
    OWL-ViT's."""
    from smart_processing.pipeline import DetectSegmentPipeline

    detection_result = detector.detect(image, prompts, threshold=threshold)
    if not detection_result.detections:
        raise ValueError(
            "no detections found for the given image/prompts; cannot benchmark "
            "segmentation or end-to-end stages without at least one real box"
        )
    first_box = detection_result.detections[0].box

    def detect_once():
        detector.detect(image, prompts, threshold=threshold)

    def segment_once():
        segmenter.set_image(image)
        segmenter.segment(first_box)

    pipeline = DetectSegmentPipeline(detector, segmenter)

    def end_to_end_once():
        pipeline.run(image, prompts, threshold=threshold)

    return {
        "detection": run_benchmark("detection", detect_once, device=device, warmup=warmup, samples=samples),
        "segmentation": run_benchmark(
            "segmentation", segment_once, device=device, warmup=warmup, samples=samples
        ),
        "end_to_end": run_benchmark(
            "end_to_end", end_to_end_once, device=device, warmup=warmup, samples=samples
        ),
    }


def export_benchmark_results(
    results: dict[str, BenchmarkResult], out_path: str, run_metadata: dict | None = None
) -> None:
    serializable = {
        name: {
            "name": r.name,
            "platform_info": asdict(r.platform_info),
            "latency": {
                "warmup_count": r.latency.warmup_count,
                "samples_seconds": list(r.latency.samples),
                "mean_seconds": r.latency.mean_seconds,
                "median_seconds": r.latency.median_seconds,
                "stdev_seconds": r.latency.stdev_seconds,
                "min_seconds": r.latency.min_seconds,
                "max_seconds": r.latency.max_seconds,
                "fps": r.latency.fps,
            },
            "peak_rss_before_bytes": r.peak_rss.before_bytes,
            "peak_rss_after_bytes": r.peak_rss.after_bytes,
            "peak_rss_delta_bytes": r.peak_rss.delta_bytes,
        }
        for name, r in results.items()
    }
    # PM-Reviwer finding: a run with 0 failures/full-success results had
    # no record anywhere of what was actually run (model, device,
    # threshold, prompts, sample counts) -- `run_metadata` is optional
    # so existing callers aren't forced to supply it, but every stage
    # export now carries it when a caller does (same top-level-metadata
    # pattern as #10's run_evaluation.py export).
    if run_metadata is not None:
        serializable["run_metadata"] = run_metadata
    Path(out_path).write_text(json.dumps(serializable, indent=2))

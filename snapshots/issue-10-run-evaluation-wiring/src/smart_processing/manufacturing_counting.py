"""Manufacturing parts-counting scenario (#8).

count_detections() is pure and reusable by any caller (camera loop,
static-image demo, future batch tooling). run_counting_demo() wires a real
DetectSegmentPipeline against a real image/prompt and returns a structured
result -- used by the real-model test and the scripted demo.
"""
from __future__ import annotations

from smart_processing.pipeline import PipelineResult


def count_detections(result: PipelineResult, label: str | None = None) -> int:
    """Counts detections in a pipeline result, optionally filtered to an
    exact label match (case-insensitive). With label=None, counts all
    detections regardless of which prompt matched."""
    if label is None:
        return len(result.results)
    label = label.lower()
    return sum(1 for r in result.results if r.detection.label.lower() == label)


def run_counting_demo(pipeline, image, prompts: list[str], threshold: float = 0.1) -> dict:
    """Runs the pipeline once and returns a structured count-per-prompt
    summary alongside the full PipelineResult, so callers can both report
    counts and save overlay/structured output from the same run."""
    result = pipeline.run(image, prompts, threshold)
    counts = {prompt: count_detections(result, prompt) for prompt in prompts}
    return {"prompts": prompts, "counts": counts, "total": len(result.results), "result": result}

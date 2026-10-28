"""Wires #10's scorer to a real detector run against #9's annotated dataset,
and exports a results table plus failed examples with their configuration
(#10's acceptance: "export comparison tables and failed examples with
configurations"). Heavy imports (PIL) stay deferred so importing this module
never requires the macos-feasibility optional extra.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from smart_processing.evaluation_dataset import EvaluationEntry, load_annotations
from smart_processing.prompt_evaluation import IOU_THRESHOLD, EntryScore, PredictedBox, score_dataset


class _Detector(Protocol):
    def detect(self, image, prompts: list[str], threshold: float): ...


def _load_image_array(images_dir: Path, relative_path: str):
    import numpy as np
    from PIL import Image

    return np.array(Image.open(images_dir / relative_path).convert("RGB"))


def run_evaluation(
    annotations_path: str,
    detector: _Detector,
    images_dir: str | None = None,
    threshold: float = 0.1,
) -> dict:
    """Runs `detector` against every entry in `annotations_path`'s dataset,
    scores the results, and returns `{"scores": ..., "failed_examples": [...]}`.
    `images_dir` defaults to annotations_path's grandparent (tests/data),
    matching entry.image's "evaluation/<file>.jpg" relative paths.

    I/O wrapper around `score_and_export`, which holds the pure (no PIL, no
    model) scoring/export logic and is unit-testable on its own with a
    fabricated predictions list.

    Attaches top-level run metadata (model identity, device, dataset
    path/entry count, detection/IoU thresholds) to the result regardless
    of whether the run passed or failed -- PM-Reviwer finding on #43: a
    perfect run previously exported no configuration at all, and even a
    failing run's config was incomplete (threshold only, no model/
    device/dataset identity), making exported comparisons impossible to
    reproduce or attribute to a specific detector/dataset combination."""
    annotations_file = Path(annotations_path)
    base_dir = Path(images_dir) if images_dir else annotations_file.resolve().parent.parent

    entries = load_annotations(annotations_path)
    entries_with_predictions: list[tuple[EvaluationEntry, list[PredictedBox]]] = []
    for entry in entries:
        image = _load_image_array(base_dir, entry.image)
        result = detector.detect(image, [entry.prompt], threshold=threshold)
        predictions = [
            PredictedBox(label=d.label, score=d.score, box_xyxy=d.box.as_xyxy())
            for d in result.detections
        ]
        entries_with_predictions.append((entry, predictions))

    run_metadata = {
        "annotations_path": str(annotations_path),
        "images_dir": str(base_dir),
        "entry_count": len(entries),
        "model_name": getattr(detector, "model_name", "unknown"),
        "device": getattr(detector, "device", "unknown"),
        "detection_threshold": threshold,
        "iou_threshold": IOU_THRESHOLD,
    }
    return score_and_export(entries_with_predictions, run_metadata=run_metadata)


def score_and_export(
    entries_with_predictions: list[tuple[EvaluationEntry, list[PredictedBox]]],
    run_metadata: dict | None = None,
) -> dict:
    """Pure scoring + failed-examples assembly: no image loading, no model.
    Unit-tested directly with fabricated (entry, predictions) pairs.

    `run_metadata` is attached to the result's top level unconditionally
    (both perfect and failing runs); callers who don't have a detector to
    introspect (e.g. unit tests) may omit it and get a minimal default
    with only `entry_count` and `iou_threshold` filled in."""
    from smart_processing.prompt_evaluation import score_entry

    if run_metadata is None:
        run_metadata = {"entry_count": len(entries_with_predictions), "iou_threshold": IOU_THRESHOLD}

    scores = score_dataset(entries_with_predictions)

    failed_examples = []
    for entry, predictions in entries_with_predictions:
        entry_score = score_entry(entry, predictions)
        if entry_score.false_positives > 0 or entry_score.false_negatives > 0:
            failed_examples.append(
                {
                    "image": entry.image,
                    "category": entry.category,
                    "prompt_style": entry.prompt_style,
                    "prompt": entry.prompt,
                    "ground_truth_boxes": [
                        {"label": g.label, "box_xyxy": list(g.box_xyxy)}
                        for g in entry.ground_truth_boxes
                    ],
                    "predicted_boxes": [
                        {"label": p.label, "score": p.score, "box_xyxy": list(p.box_xyxy)}
                        for p in predictions
                    ],
                    "score": asdict(entry_score),
                }
            )

    return {
        "run_metadata": run_metadata,
        "scores": _scores_to_dict(scores),
        "failed_examples": failed_examples,
    }


def _scores_to_dict(scores: dict) -> dict:
    def _convert(value):
        return asdict(value) if isinstance(value, EntryScore) else {k: asdict(v) for k, v in value.items()}

    return {key: _convert(value) for key, value in scores.items()}


def export_results(result: dict, out_path: str) -> None:
    Path(out_path).write_text(json.dumps(result, indent=2))

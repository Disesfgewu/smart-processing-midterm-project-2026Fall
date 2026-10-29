"""Prompt evaluation scorer (#10).

Pure IoU-matching and micro-averaged precision/recall/F1, implementing
docs/evaluation_methodology.md's spec exactly. No model inference and no
pipeline coupling here: callers run a detector themselves (#4/#6) and pass
its predicted boxes in; this module only scores them against
tests/data/evaluation/annotations.json's ground truth (#9).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from smart_processing.evaluation_dataset import EvaluationEntry

IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class PredictedBox:
    """One detector output: label is expected to be the full prompt text
    used for the query (#39's fix), not a short ground-truth noun."""

    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class EntryScore:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def score_entry(entry: EvaluationEntry, predictions: list[PredictedBox]) -> EntryScore:
    """Greedy one-to-one matching: predictions sorted by descending score;
    each matched to its best-IoU unmatched ground-truth box at IoU>=0.5,
    with the predicted label compared against entry.prompt (case-
    insensitive, stripped) rather than any ground-truth box's label."""
    expected_label = entry.prompt.strip().lower()
    ordered = sorted(predictions, key=lambda p: p.score, reverse=True)
    unmatched_gt = list(entry.ground_truth_boxes)
    true_positives = 0
    false_positives = 0
    for pred in ordered:
        if pred.label.strip().lower() != expected_label:
            false_positives += 1
            continue
        best_idx, best_iou = -1, IOU_THRESHOLD
        for i, gt in enumerate(unmatched_gt):
            score = iou(pred.box_xyxy, gt.box_xyxy)
            if score >= IOU_THRESHOLD and (best_idx == -1 or score > best_iou):
                best_idx, best_iou = i, score
        if best_idx >= 0:
            true_positives += 1
            del unmatched_gt[best_idx]
        else:
            false_positives += 1
    false_negatives = len(unmatched_gt)
    precision, recall, f1 = _precision_recall_f1(true_positives, false_positives, false_negatives)
    return EntryScore(true_positives, false_positives, false_negatives, precision, recall, f1)


def _aggregate(scores: list[EntryScore]) -> EntryScore:
    tp = sum(s.true_positives for s in scores)
    fp = sum(s.false_positives for s in scores)
    fn = sum(s.false_negatives for s in scores)
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    return EntryScore(tp, fp, fn, precision, recall, f1)


def score_dataset(
    entries_with_predictions: list[tuple[EvaluationEntry, list[PredictedBox]]],
) -> dict:
    """Overall micro-averaged metrics plus per-category and per-prompt-style
    breakdowns, per docs/evaluation_methodology.md point 6."""
    per_entry = [score_entry(entry, preds) for entry, preds in entries_with_predictions]
    by_category: dict[str, list[EntryScore]] = {}
    by_prompt_style: dict[str, list[EntryScore]] = {}
    for (entry, _), entry_score in zip(entries_with_predictions, per_entry):
        by_category.setdefault(entry.category, []).append(entry_score)
        by_prompt_style.setdefault(entry.prompt_style, []).append(entry_score)
    return {
        "overall": _aggregate(per_entry),
        "by_category": {k: _aggregate(v) for k, v in by_category.items()},
        "by_prompt_style": {k: _aggregate(v) for k, v in by_prompt_style.items()},
    }

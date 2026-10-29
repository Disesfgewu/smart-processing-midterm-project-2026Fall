"""Ground-truth evaluation dataset schema and loader (#9).

See docs/evaluation_methodology.md for the full format spec, the accuracy
metric and matching rules this dataset is prepared for (#10 implements the
scorer). No model code here; pure data loading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_CATEGORIES = {"similar_objects", "small_objects", "occlusion"}
VALID_PROMPT_STYLES = {"basic_noun", "descriptive_phrase", "fine_grained"}


@dataclass(frozen=True)
class GroundTruthBox:
    label: str
    box_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must be a non-blank string")
        x0, y0, x1, y1 = self.box_xyxy
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"box_xyxy must have x0<x1 and y0<y1, got {self.box_xyxy}")


@dataclass(frozen=True)
class EvaluationEntry:
    image: str
    image_license: str
    category: str
    prompt_style: str
    prompt: str
    ground_truth_boxes: list[GroundTruthBox] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {VALID_CATEGORIES}, got {self.category!r}")
        if self.prompt_style not in VALID_PROMPT_STYLES:
            raise ValueError(
                f"prompt_style must be one of {VALID_PROMPT_STYLES}, got {self.prompt_style!r}"
            )
        if not self.prompt.strip():
            raise ValueError("prompt must be a non-blank string")
        if not self.image_license.strip():
            raise ValueError("image_license must be a non-blank string (document how it was verified)")


def load_annotations(path: str) -> list[EvaluationEntry]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of entries, got {type(raw)}")
    entries = []
    for i, item in enumerate(raw):
        try:
            boxes = [
                GroundTruthBox(label=b["label"], box_xyxy=tuple(b["box_xyxy"]))
                for b in item.get("ground_truth_boxes", [])
            ]
            entries.append(
                EvaluationEntry(
                    image=item["image"],
                    image_license=item["image_license"],
                    category=item["category"],
                    prompt_style=item["prompt_style"],
                    prompt=item["prompt"],
                    ground_truth_boxes=boxes,
                )
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"annotations.json entry {i} is invalid: {e}") from e
    return entries


def summarize(entries: list[EvaluationEntry]) -> dict:
    """Coverage summary: distinct images, counts per category/prompt_style --
    used to check #9's "10 images, 3 categories, 3+ prompt styles" minimum
    without re-deriving it by hand each time."""
    images = {e.image for e in entries}
    by_category: dict[str, int] = {}
    by_prompt_style: dict[str, int] = {}
    for e in entries:
        by_category[e.category] = by_category.get(e.category, 0) + 1
        by_prompt_style[e.prompt_style] = by_prompt_style.get(e.prompt_style, 0) + 1
    return {
        "distinct_images": len(images),
        "total_entries": len(entries),
        "by_category": by_category,
        "by_prompt_style": by_prompt_style,
    }

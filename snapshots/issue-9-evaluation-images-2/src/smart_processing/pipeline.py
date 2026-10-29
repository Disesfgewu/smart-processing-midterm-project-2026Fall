"""Text -> boxes -> masks pipeline (#6), composing #3's TextDetector and
BoxSegmenter Protocols. No model code here; adapters (#4/#5) are injected,
so this module never needs torch/transformers to import.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from smart_processing.types import (
    BoxSegmenter,
    Detection,
    ImageArray,
    SegmentationMask,
    TextDetector,
    validate_image,
)

_OVERLAY_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


@dataclass(frozen=True)
class PipelineDetection:
    detection: Detection
    mask: SegmentationMask


@dataclass(frozen=True)
class PipelineResult:
    image_width: int
    image_height: int
    results: list[PipelineDetection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width/image_height must be positive")
        for r in self.results:
            if (r.mask.height, r.mask.width) != (self.image_height, self.image_width):
                raise ValueError(
                    f"mask shape {(r.mask.height, r.mask.width)} does not match "
                    f"image dims {(self.image_height, self.image_width)}"
                )
            if r.detection.box.x1 > self.image_width or r.detection.box.y1 > self.image_height:
                raise ValueError(
                    f"detection box {r.detection.box.as_xyxy()} exceeds image dims "
                    f"{(self.image_width, self.image_height)}"
                )


class DetectSegmentPipeline:
    def __init__(self, detector: TextDetector, segmenter: BoxSegmenter):
        self._detector = detector
        self._segmenter = segmenter

    def run(self, image: ImageArray, prompts: list[str], threshold: float = 0.1) -> PipelineResult:
        height, width = validate_image(image)
        detection_result = self._detector.detect(image, prompts, threshold)
        if (detection_result.image_width, detection_result.image_height) != (width, height):
            raise ValueError(
                f"detector reported image dims "
                f"{(detection_result.image_width, detection_result.image_height)} "
                f"but the input image is {(width, height)}"
            )

        if not detection_result.detections:
            return PipelineResult(image_width=width, image_height=height, results=[])

        self._segmenter.set_image(image)
        results = [
            PipelineDetection(detection=d, mask=self._segmenter.segment(d.box))
            for d in detection_result.detections
        ]
        return PipelineResult(image_width=width, image_height=height, results=results)


def save_structured_output(result: PipelineResult, path: str) -> None:
    payload = {
        "image_width": result.image_width,
        "image_height": result.image_height,
        "detections": [
            {
                "label": r.detection.label,
                "score": r.detection.score,
                "box_xyxy": list(r.detection.box.as_xyxy()),
                "mask_score": r.mask.score,
                "mask_pixel_count": int(r.mask.mask.sum()),
            }
            for r in result.results
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def render_overlay_image(image: ImageArray, result: PipelineResult) -> ImageArray:
    """Draws a translucent mask fill plus a box border per detection, using
    plain numpy only -- no PIL/disk dependency, so callers (e.g. a live
    camera loop) can use this in-memory per frame."""
    import numpy as np

    height, width = validate_image(image)
    if (height, width) != (result.image_height, result.image_width):
        raise ValueError(
            f"image shape {(height, width)} does not match result dims "
            f"{(result.image_height, result.image_width)}"
        )

    overlay = image.copy()
    for i, r in enumerate(result.results):
        color = np.array(_OVERLAY_COLORS[i % len(_OVERLAY_COLORS)], dtype=np.uint8)
        mask = r.mask.mask
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
        x0, y0, x1, y1 = (int(round(v)) for v in r.detection.box.as_xyxy())
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(image.shape[1] - 1, x1)
        y1 = min(image.shape[0] - 1, y1)
        thickness = 2
        overlay[y0 : y0 + thickness, x0:x1] = color
        overlay[max(y0, y1 - thickness) : y1, x0:x1] = color
        overlay[y0:y1, x0 : x0 + thickness] = color
        overlay[y0:y1, max(x0, x1 - thickness) : x1] = color
    return overlay


def save_overlay_image(image: ImageArray, result: PipelineResult, path: str) -> None:
    """Renders the overlay (see render_overlay_image) and saves it to disk.
    PIL is only needed here, deferred-imported since it's an optional extra."""
    overlay = render_overlay_image(image, result)

    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)

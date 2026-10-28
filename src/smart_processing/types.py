"""Portable detection/segmentation contracts (#3).

Pure data/interface definitions: no model downloads, no torch/numpy backend
coupling beyond numpy arrays as the agreed image/mask representation.
Concrete adapters (nanoowl/nanosam/#4/#5) convert to/from these types; they
own normalized-vs-pixel coordinate conversion (see docs/third_party_apis.md)
so every contract defined here is explicit pixel space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# Images are HWC uint8 RGB numpy arrays. Height/width are read from the
# array's shape; no separate Image wrapper to avoid duplicating that state.
ImageArray = np.ndarray


def validate_image(image: ImageArray) -> tuple[int, int]:
    """Validate the HWC uint8 RGB contract; returns (height, width)."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        shape = getattr(image, "shape", None)
        raise ValueError(f"image must be an HWC RGB array, got shape {shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {image.dtype}")
    height, width = image.shape[0], image.shape[1]
    if height <= 0 or width <= 0:
        raise ValueError(f"image must have positive height and width, got {(height, width)}")
    return height, width


@dataclass(frozen=True)
class BoundingBox:
    """Pixel xyxy box, origin top-left, in the coordinate space of one image."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name, value in (("x0", self.x0), ("y0", self.y0), ("x1", self.x1), ("y1", self.y1)):
            if not math.isfinite(value):
                raise ValueError(f"BoundingBox.{name} must be finite, got {value}")
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(
                f"BoundingBox requires x0>=0 and y0>=0, got ({self.x0}, {self.y0})"
            )
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(
                f"BoundingBox requires x0<x1 and y0<y1, got "
                f"({self.x0}, {self.y0}, {self.x1}, {self.y1})"
            )

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Detection:
    box: BoundingBox
    score: float
    label: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass(frozen=True)
class DetectionResult:
    image_width: int
    image_height: int
    detections: list[Detection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width/image_height must be positive")
        for d in self.detections:
            if d.box.x1 > self.image_width or d.box.y1 > self.image_height:
                raise ValueError(
                    f"detection box {d.box.as_xyxy()} exceeds image bounds "
                    f"({self.image_width}x{self.image_height})"
                )


@dataclass(frozen=True)
class SegmentationMask:
    """Boolean mask, shape (height, width), matching the source image's dims."""

    mask: np.ndarray
    score: float

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise ValueError(f"mask dtype must be bool, got {self.mask.dtype}")
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be 2-D (height, width), got ndim={self.mask.ndim}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    @property
    def height(self) -> int:
        return self.mask.shape[0]

    @property
    def width(self) -> int:
        return self.mask.shape[1]


@dataclass(frozen=True)
class BackendConfig:
    """Backend selection kept out of the contracts below; adapters read this,
    the Protocols never reference it, so swapping backends never touches
    the data contracts."""

    device: str = "cpu"
    model_name: str | None = None


class TextDetector(Protocol):
    def detect(
        self, image: ImageArray, prompts: list[str], threshold: float = 0.1
    ) -> DetectionResult: ...


class BoxSegmenter(Protocol):
    def set_image(self, image: ImageArray) -> None: ...

    def segment(self, box: BoundingBox) -> SegmentationMask: ...

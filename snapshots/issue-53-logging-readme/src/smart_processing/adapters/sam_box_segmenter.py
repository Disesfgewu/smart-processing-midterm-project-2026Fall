"""macOS-feasible BoxSegmenter (#5), backed by the SAM backend validated in
#2 (docs/macos_feasibility.md): segment_anything.SamPredictor.

This is not nanosam.utils.predictor.Predictor (the pinned #18 submodule,
TensorRT-only, cannot even import on macOS); it's the documented macOS
substitute. Heavy imports (torch, segment_anything) are deferred into
__init__ so importing this module never requires them.
"""
from __future__ import annotations

from smart_processing.types import BackendConfig, BoundingBox, SegmentationMask, validate_image

DEFAULT_MODEL_TYPE = "vit_b"


def validate_segment_inputs(image_shape, box) -> None:
    """Pure input validation, importable/testable without loading any model."""
    if not isinstance(box, BoundingBox):
        raise ValueError(f"box must be a BoundingBox, got {type(box)!r}")
    if image_shape is None:
        raise RuntimeError("set_image() must be called before segment()")
    height, width = image_shape
    if box.x1 > width or box.y1 > height:
        raise ValueError(f"box {box.as_xyxy()} exceeds image bounds {(width, height)}")


class SamBoxSegmenter:
    def __init__(
        self,
        checkpoint_path: str,
        config: BackendConfig | None = None,
        model_type: str = DEFAULT_MODEL_TYPE,
    ):
        config = config or BackendConfig(device="cpu")
        from segment_anything import SamPredictor, sam_model_registry

        model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        model.to(device=config.device)
        model.eval()
        self._predictor = SamPredictor(model)
        self.device = config.device
        self._image_shape = None  # (height, width), set by set_image()

    def set_image(self, image) -> None:
        height, width = validate_image(image)
        self._predictor.set_image(image)
        self._image_shape = (height, width)

    def segment(self, box: BoundingBox) -> SegmentationMask:
        validate_segment_inputs(self._image_shape, box)

        import numpy as np

        box_array = np.array(box.as_xyxy())
        masks, scores, _ = self._predictor.predict(box=box_array, multimask_output=False)
        mask = masks[0].astype(bool)
        if mask.shape != self._image_shape:
            raise ValueError(
                f"predicted mask shape {mask.shape} does not match the image "
                f"shape {self._image_shape} set by set_image()"
            )
        score = float(min(1.0, max(0.0, scores[0])))
        return SegmentationMask(mask=mask, score=score)

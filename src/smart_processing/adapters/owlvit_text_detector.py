"""macOS-feasible TextDetector (#4), backed by the OWL-ViT backend validated
in #2 (docs/macos_feasibility.md): transformers.OwlViTForObjectDetection.

This is not nanoowl.owl_predictor.OwlPredictor (the pinned #18 submodule,
TensorRT/Jetson path for #13); it's the same underlying backbone, run
directly via transformers, as documented in #2/#18's docs/third_party_apis.md.
Heavy imports (torch, transformers) are deferred into __init__ so importing
this module never requires them.
"""
from __future__ import annotations

from smart_processing.types import BackendConfig, BoundingBox, Detection, DetectionResult

DEFAULT_MODEL_NAME = "google/owlvit-base-patch32"


def validate_detect_inputs(image, prompts: list[str]) -> None:
    """Pure input validation, importable/testable without loading any model."""
    if image is None or getattr(image, "ndim", None) != 3 or image.shape[2] != 3:
        shape = getattr(image, "shape", None)
        raise ValueError(f"image must be an HWC RGB array, got shape {shape}")
    if not prompts:
        raise ValueError("prompts must be a non-empty list of text strings")
    if any(not p or not p.strip() for p in prompts):
        raise ValueError("prompts must not contain empty/blank strings")


class OwlViTTextDetector:
    def __init__(self, config: BackendConfig | None = None):
        config = config or BackendConfig(device="cpu", model_name=DEFAULT_MODEL_NAME)
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self._torch = torch
        self.device = config.device
        self.model_name = config.model_name or DEFAULT_MODEL_NAME
        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model = (
            OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device).eval()
        )

    def detect(self, image, prompts: list[str], threshold: float = 0.1) -> DetectionResult:
        validate_detect_inputs(image, prompts)

        height, width = image.shape[0], image.shape[1]
        inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self._torch.tensor([[height, width]])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        detections: list[Detection] = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            x0, y0, x1, y1 = (float(v) for v in box.tolist())
            # Model output can slightly exceed image bounds; clamp to the
            # contract's "box within image bounds" requirement (#3).
            x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
            y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
            if x1 <= x0 or y1 <= y0:
                continue  # degenerate after clamping; skip rather than fail the whole call
            detections.append(
                Detection(
                    box=BoundingBox(x0, y0, x1, y1),
                    score=float(min(1.0, max(0.0, score.item()))),
                    label=prompts[int(label_idx)],
                )
            )

        return DetectionResult(image_width=width, image_height=height, detections=detections)

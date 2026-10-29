"""Reusable macOS feasibility detection/segmentation logic (#2).

scripts/owlvit_smoke_test.py and scripts/sam_box_segment_smoke_test.py are
thin CLI wrappers over these two functions -- model loading, inference and
post-processing live here, per #2's acceptance: "feasibility code must
live in the package or outside tracked source, not scattered experiment
folders." Heavy imports stay deferred into each function so importing this
module never requires torch/transformers/segment_anything.
"""
import time


def run_owlvit_detection(image_path: str, prompts: list[str], threshold: float = 0.1) -> dict:
    import torch
    from PIL import Image
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    model_name = "google/owlvit-base-patch32"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.time()
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()
    model_load_seconds = time.time() - t0

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompts], images=image, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_seconds = time.time() - t0

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    detections = [
        {"label": prompts[int(label)], "score": float(score), "box_xyxy": [round(v, 1) for v in box.tolist()]}
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"])
    ]
    return {
        "device": device,
        "torch_version": torch.__version__,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "image_path": image_path,
        "image_size": image.size,
        "prompts": prompts,
        "detections": detections,
    }


def run_sam_box_segmentation(image_path: str, checkpoint_path: str, box_xyxy) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from segment_anything import SamPredictor, sam_model_registry

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.time()
    model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    model.to(device=device).eval()
    predictor = SamPredictor(model)
    model_load_seconds = time.time() - t0

    image = np.array(Image.open(image_path).convert("RGB"))
    box_array = np.array(box_xyxy)

    t0 = time.time()
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=box_array, multimask_output=False)
    inference_seconds = time.time() - t0

    mask = masks[0]
    return {
        "device": device,
        "torch_version": torch.__version__,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "image_path": image_path,
        "box_xyxy": list(box_xyxy),
        "mask_shape": mask.shape,
        "mask_dtype": str(mask.dtype),
        "score": float(scores[0]),
        "mask_pixel_count": int(mask.sum()),
        "mask_total_pixels": int(mask.size),
    }

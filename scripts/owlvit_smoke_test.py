"""Real macOS feasibility smoke test for the OWL-ViT backbone NanoOWL wraps.

Runs actual text-conditioned detection (no mocks). Not the TensorRT-optimized
NanoOWL path; documents the torch-only backend as the macOS substitute.
"""
import sys
import time

import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

MODEL_NAME = "google/owlvit-base-patch32"


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/sample_cats.jpg"
    prompts = ["a photo of a cat", "a remote control"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}")

    t0 = time.time()
    processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"model load: {time.time() - t0:.2f}s")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompts], images=image, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"inference: {time.time() - t0:.3f}s")

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=0.1, target_sizes=target_sizes
    )[0]

    print(f"image={image_path} size={image.size} prompts={prompts}")
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        box = [round(v, 1) for v in box.tolist()]
        print(f"  label={prompts[label]!r} score={score:.3f} box_xyxy={box}")


if __name__ == "__main__":
    main()

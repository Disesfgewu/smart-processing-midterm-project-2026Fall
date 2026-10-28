"""Thin CLI wrapper for smart_processing.feasibility.run_owlvit_detection.

Real macOS feasibility smoke test for the OWL-ViT backbone NanoOWL wraps
(no mocks). Not the TensorRT-optimized NanoOWL path; documents the
torch-only backend as the macOS substitute.
"""
import sys

from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image
from smart_processing.feasibility import run_owlvit_detection


def main():
    image_path = ensure_sample_image(sys.argv[1] if len(sys.argv) > 1 else SAMPLE_IMAGE_PATH)
    prompts = ["a photo of a cat", "a remote control"]

    result = run_owlvit_detection(image_path, prompts)

    print(f"device={result['device']} torch={result['torch_version']}")
    print(f"model load: {result['model_load_seconds']:.2f}s")
    print(f"inference: {result['inference_seconds']:.3f}s")
    print(f"image={result['image_path']} size={result['image_size']} prompts={result['prompts']}")
    for d in result["detections"]:
        print(f"  label={d['label']!r} score={d['score']:.3f} box_xyxy={d['box_xyxy']}")


if __name__ == "__main__":
    main()

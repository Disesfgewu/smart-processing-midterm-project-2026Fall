"""Thin CLI wrapper for smart_processing.feasibility.run_sam_box_segmentation.

Real macOS feasibility smoke test for box-prompted segmentation.
nanosam.utils.predictor.Predictor imports tensorrt/torch2trt at module
level (see docs/third_party_apis.md) and cannot run on macOS at all.
This uses the official Segment Anything ViT-B checkpoint (the same
upstream family nanosam/MobileSAM distill from) as the documented macOS
substitute backend, run with plain PyTorch, no TensorRT.
"""
import sys

from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image
from smart_processing.feasibility import run_sam_box_segmentation


def main():
    image_path = ensure_sample_image(sys.argv[1] if len(sys.argv) > 1 else SAMPLE_IMAGE_PATH)
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "tests/data/sam_vit_b_01ec64.pth"
    # Box from the OWL-ViT smoke test's "a photo of a cat" detection.
    box_xyxy = [1.5, 55.3, 315.6, 472.2]

    result = run_sam_box_segmentation(image_path, checkpoint, box_xyxy)

    print(f"device={result['device']} torch={result['torch_version']}")
    print(f"model load: {result['model_load_seconds']:.2f}s")
    print(f"inference: {result['inference_seconds']:.3f}s")
    print(f"image={result['image_path']} box_xyxy={result['box_xyxy']}")
    print(f"mask shape={result['mask_shape']} dtype={result['mask_dtype']} score={result['score']:.3f}")
    pct = 100 * result["mask_pixel_count"] / result["mask_total_pixels"]
    print(f"mask pixel count={result['mask_pixel_count']} of {result['mask_total_pixels']} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

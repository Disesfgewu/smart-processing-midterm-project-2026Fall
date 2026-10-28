"""Real macOS feasibility smoke test for box-prompted segmentation.

nanosam.utils.predictor.Predictor imports tensorrt/torch2trt at module
level (see docs/third_party_apis.md) and cannot run on macOS at all.
This uses the official Segment Anything ViT-B checkpoint (the same
upstream family nanosam/MobileSAM distill from) as the documented macOS
substitute backend, run with plain PyTorch, no TensorRT.
"""
import sys
import time

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image


def main():
    image_path = ensure_sample_image(sys.argv[1] if len(sys.argv) > 1 else SAMPLE_IMAGE_PATH)
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "tests/data/sam_vit_b_01ec64.pth"
    # Box from the OWL-ViT smoke test's "a photo of a cat" detection.
    box_xyxy = np.array([1.5, 55.3, 315.6, 472.2])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}")

    t0 = time.time()
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model.to(device=device).eval()
    predictor = SamPredictor(model)
    print(f"model load: {time.time() - t0:.2f}s")

    image = np.array(Image.open(image_path).convert("RGB"))

    t0 = time.time()
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=box_xyxy, multimask_output=False)
    print(f"inference: {time.time() - t0:.3f}s")

    mask = masks[0]
    print(f"image={image_path} box_xyxy={box_xyxy.tolist()}")
    print(f"mask shape={mask.shape} dtype={mask.dtype} score={scores[0]:.3f}")
    print(f"mask pixel count={int(mask.sum())} of {mask.size} ({100 * mask.mean():.1f}%)")


if __name__ == "__main__":
    main()

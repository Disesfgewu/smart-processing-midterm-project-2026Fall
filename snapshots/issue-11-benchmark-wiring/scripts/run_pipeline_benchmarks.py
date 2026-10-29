"""Thin CLI wrapper for smart_processing.run_benchmarks.run_pipeline_benchmarks.

Runs real OWL-ViT (#4) and SAM (#5) inference, benchmarking detection,
segmentation, and the full pipeline (#6) separately. No mocks: real model
calls, real timing. macOS only -- never claims Jetson/TensorRT numbers.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image
from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
from smart_processing.adapters.sam_box_segmenter import SamBoxSegmenter
from smart_processing.run_benchmarks import export_benchmark_results, run_pipeline_benchmarks
from smart_processing.types import BackendConfig

DEFAULT_OUT = Path(__file__).resolve().parent.parent / ".agent-artifacts" / "benchmark_results.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="path to an image; defaults to the sample image")
    parser.add_argument("--checkpoint", required=True, help="SAM vit_b checkpoint path")
    parser.add_argument("--prompt", default="a photo of a cat")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    image_path = ensure_sample_image(args.image or SAMPLE_IMAGE_PATH)
    image = np.array(Image.open(image_path).convert("RGB"))

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    segmenter = SamBoxSegmenter(args.checkpoint)

    results = run_pipeline_benchmarks(
        detector, segmenter, image, [args.prompt], samples=args.samples
    )

    for name, result in results.items():
        lat = result.latency
        print(
            f"{name}: mean={lat.mean_seconds * 1000:.1f}ms median={lat.median_seconds * 1000:.1f}ms "
            f"fps={lat.fps:.2f} peak_rss_delta={result.peak_rss.delta_bytes / 1024 / 1024:.2f}MB"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_benchmark_results(results, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

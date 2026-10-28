"""Thin CLI wrapper for smart_processing.run_evaluation.run_evaluation.

Runs the real OWL-ViT backend (#4) against #9's annotated evaluation
dataset, scores it against #10's IoU-matching spec, and writes the results
table plus failed examples to a JSON file. No mocks: uses the same
OwlViTTextDetector validated in #2/#4.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
from smart_processing.run_evaluation import export_results, run_evaluation
from smart_processing.types import BackendConfig

DEFAULT_ANNOTATIONS = Path(__file__).resolve().parent.parent / "tests" / "data" / "evaluation" / "annotations.json"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / ".agent-artifacts" / "prompt_evaluation_results.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    result = run_evaluation(args.annotations, detector, threshold=args.threshold)

    overall = result["scores"]["overall"]
    print(f"overall: P={overall['precision']:.3f} R={overall['recall']:.3f} F1={overall['f1']:.3f}")
    print(f"  TP={overall['true_positives']} FP={overall['false_positives']} FN={overall['false_negatives']}")
    for key in ("by_category", "by_prompt_style"):
        print(f"{key}:")
        for name, score in result["scores"][key].items():
            print(f"  {name}: P={score['precision']:.3f} R={score['recall']:.3f} F1={score['f1']:.3f}")
    print(f"{len(result['failed_examples'])} failed example(s) out of {len(result['scores'])} score groups")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_results(result, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

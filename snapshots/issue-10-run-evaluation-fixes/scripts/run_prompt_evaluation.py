"""Thin CLI wrapper for smart_processing.run_evaluation.run_evaluation.

Runs the real OWL-ViT backend (#4) against an annotated evaluation
dataset (schema: docs/evaluation_methodology.md), scores it against #10's
IoU-matching spec, and writes the results table plus failed examples to a
JSON file. No mocks: uses the same OwlViTTextDetector validated in #2/#4.

`--annotations` is required, not defaulted to #9's dataset path: this
branch does not itself contain #9's annotated images (they live on a
separate, not-yet-merged PR -- see README.md's "Prompt evaluation
scorer" row), so a hardcoded default would silently FileNotFoundError
for anyone running this without that unmerged branch checked out. Point
`--annotations` at any annotations.json following the documented schema
plus its sibling image files (PM-Reviwer finding on #43).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
from smart_processing.run_evaluation import export_results, run_evaluation
from smart_processing.types import BackendConfig

DEFAULT_OUT = Path(__file__).resolve().parent.parent / ".agent-artifacts" / "prompt_evaluation_results.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        required=True,
        help="path to an annotations.json following docs/evaluation_methodology.md's schema",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    if not Path(args.annotations).exists():
        sys.exit(
            f"error: {args.annotations!r} does not exist.\n"
            "This branch does not contain #9's annotated dataset images "
            "(see README.md's 'Prompt evaluation scorer' row for why).\n"
            "Point --annotations at any annotations.json following "
            "docs/evaluation_methodology.md's schema, with each entry's "
            "image file present at the matching relative path."
        )

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    result = run_evaluation(args.annotations, detector, threshold=args.threshold)

    print("run_metadata:", result["run_metadata"])
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

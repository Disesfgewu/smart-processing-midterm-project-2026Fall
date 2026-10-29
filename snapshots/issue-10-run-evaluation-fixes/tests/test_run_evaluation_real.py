"""Opt-in real-model end-to-end run of #10's evaluation wiring: real OWL-ViT
inference, a real image, real scoring, a real exported JSON.

Skipped by default (downloads/loads a real ~600MB model); run explicitly
with RUN_FEASIBILITY_SMOKE=1, same convention as #2/#4's smoke tests.

Self-contained dataset (PM-Reviwer finding on #43): this branch does not
contain #9's annotated dataset images (a separate, not-yet-merged PR), so
the exact-head real-inference run below builds its own minimal
annotations.json against the existing real COCO sample image, rather than
silently depending on an external branch. Its ground-truth box is the
real detection box documented in docs/macos_feasibility.md's real OWL-ViT
run (not independently re-verified by visual inspection like #9's dataset
boxes are) -- this is a wiring/integration test proving the pipeline
produces correct, well-formed scores end to end, not new accuracy
evidence for #9/#10's dataset.
"""
import json
import os
from pathlib import Path

import pytest

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run the real evaluation wiring end-to-end"
)

# Real detection box for "a photo of a cat" on the COCO sample image,
# documented in docs/macos_feasibility.md's real OWL-ViT run (image
# 640x480). Used as ground truth for this wiring test.
_DOCUMENTED_CAT_BOX = [1.5, 55.3, 315.6, 472.2]


@pytest.fixture(scope="module")
def sample_image_dir(tmp_path_factory):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image

    image_path = Path(ensure_sample_image(SAMPLE_IMAGE_PATH))
    eval_dir = tmp_path_factory.mktemp("self_contained_eval") / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "sample_cats.jpg").write_bytes(image_path.read_bytes())

    annotations = [
        {
            "image": "evaluation/sample_cats.jpg",
            "image_license": "Flickr source license NOT independently verified (see "
            "docs/macos_feasibility.md's real-detection-run section) -- this is a "
            "wiring/integration test fixture reusing the existing sample image, not "
            "a license-verified entry for #9's accuracy dataset",
            "category": "similar_objects",
            "prompt_style": "basic_noun",
            "prompt": "a photo of a cat",
            "ground_truth_boxes": [{"label": "cat", "box_xyxy": _DOCUMENTED_CAT_BOX}],
        }
    ]
    # annotations.json lives *inside* the "evaluation/" dir, matching #9's
    # real layout (tests/data/evaluation/annotations.json) -- run_evaluation
    # resolves images_dir as annotations_path.parent.parent by default, so
    # entry.image="evaluation/<file>" must be one level below that.
    annotations_path = eval_dir / "annotations.json"
    annotations_path.write_text(json.dumps(annotations))
    return annotations_path


def test_real_run_evaluation_exact_head_self_contained(sample_image_dir, tmp_path):
    from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
    from smart_processing.run_evaluation import export_results, run_evaluation
    from smart_processing.types import BackendConfig

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    result = run_evaluation(str(sample_image_dir), detector, threshold=0.1)

    metadata = result["run_metadata"]
    assert metadata["entry_count"] == 1
    assert metadata["model_name"] == detector.model_name
    assert metadata["device"] == "cpu"
    assert metadata["detection_threshold"] == 0.1

    overall = result["scores"]["overall"]
    # A fresh real detection should match the documented real box: not a
    # fabricated/forced result -- if OWL-ViT's output genuinely changed
    # (different package version, different hardware), this would fail
    # and should be investigated, not papered over.
    assert overall["true_positives"] >= 1

    out_path = tmp_path / "results.json"
    export_results(result, str(out_path))
    exported = json.loads(out_path.read_text())
    assert exported["run_metadata"]["entry_count"] == 1


EXTERNAL_ANNOTATIONS = os.environ.get("EVAL_ANNOTATIONS_PATH")


@pytest.mark.skipif(
    not EXTERNAL_ANNOTATIONS, reason="set EVAL_ANNOTATIONS_PATH to also run against #9's real dataset"
)
def test_real_run_evaluation_against_external_dataset(tmp_path):
    """Optional: set EVAL_ANNOTATIONS_PATH to a real annotations.json (e.g.
    a checkout of #41's branch, or any dataset following
    docs/evaluation_methodology.md's schema) to additionally verify this
    wiring against #9's full dataset once it's available locally."""
    from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
    from smart_processing.run_evaluation import export_results, run_evaluation
    from smart_processing.types import BackendConfig

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    result = run_evaluation(EXTERNAL_ANNOTATIONS, detector, threshold=0.1)

    overall = result["scores"]["overall"]
    assert 0.0 <= overall["precision"] <= 1.0
    assert 0.0 <= overall["recall"] <= 1.0

    out_path = tmp_path / "results.json"
    export_results(result, str(out_path))
    assert out_path.exists() and out_path.stat().st_size > 0

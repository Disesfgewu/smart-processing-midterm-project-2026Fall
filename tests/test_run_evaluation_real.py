"""Opt-in real-model end-to-end run of #10's evaluation wiring against #9's
actual annotated dataset: real OWL-ViT inference, real images, real scoring.

Skipped by default (downloads/loads a real ~600MB model); run explicitly
with RUN_FEASIBILITY_SMOKE=1, same convention as #2/#4's smoke tests.
"""
import os
from pathlib import Path

import pytest

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run the real evaluation dataset end-to-end"
)

ANNOTATIONS_PATH = (
    Path(__file__).resolve().parent / "data" / "evaluation" / "annotations.json"
)


def test_real_run_evaluation_produces_scores_and_json(tmp_path):
    from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
    from smart_processing.run_evaluation import export_results, run_evaluation
    from smart_processing.types import BackendConfig

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    result = run_evaluation(str(ANNOTATIONS_PATH), detector, threshold=0.1)

    overall = result["scores"]["overall"]
    assert 0.0 <= overall["precision"] <= 1.0
    assert 0.0 <= overall["recall"] <= 1.0
    assert set(result["scores"]["by_category"]) == {
        "similar_objects",
        "small_objects",
        "occlusion",
    }
    assert set(result["scores"]["by_prompt_style"]) == {
        "basic_noun",
        "fine_grained",
        "descriptive_phrase",
    }

    out_path = tmp_path / "results.json"
    export_results(result, str(out_path))
    assert out_path.exists() and out_path.stat().st_size > 0

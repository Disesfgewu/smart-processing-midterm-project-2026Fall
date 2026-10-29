"""Opt-in smoke tests for the real macOS feasibility scripts (#2).

Skipped by default: they download/load real multi-hundred-MB models and are
not part of the fast `pytest` suite exercised by CI. Run explicitly with
RUN_FEASIBILITY_SMOKE=1 after placing tests/data/sam_vit_b_01ec64.pth.
"""
import os
import subprocess
import sys

import pytest

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run real model smoke tests")
def test_owlvit_detects_real_objects():
    result = subprocess.run(
        [sys.executable, "scripts/owlvit_smoke_test.py", "tests/data/sample_cats.jpg"],
        capture_output=True, text=True, check=True,
    )
    assert "label=" in result.stdout
    assert "score=" in result.stdout


@pytest.mark.skipif(not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run real model smoke tests")
def test_sam_segments_real_box():
    result = subprocess.run(
        [sys.executable, "scripts/sam_box_segment_smoke_test.py",
         "tests/data/sample_cats.jpg", "tests/data/sam_vit_b_01ec64.pth"],
        capture_output=True, text=True, check=True,
    )
    assert "mask shape=" in result.stdout
    assert "score=" in result.stdout

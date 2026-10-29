"""Fast tests (no model download): input validation only."""
import numpy as np
import pytest

from smart_processing.adapters.owlvit_text_detector import validate_detect_inputs

VALID_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def test_accepts_valid_inputs():
    validate_detect_inputs(VALID_IMAGE, ["a cat"], threshold=0.1)  # must not raise


def test_rejects_empty_prompts():
    with pytest.raises(ValueError):
        validate_detect_inputs(VALID_IMAGE, [], threshold=0.1)


@pytest.mark.parametrize("prompts", [["a cat", ""], ["   "], [""]])
def test_rejects_blank_prompt_strings(prompts):
    with pytest.raises(ValueError):
        validate_detect_inputs(VALID_IMAGE, prompts, threshold=0.1)


def test_rejects_non_string_prompt_elements():
    with pytest.raises(ValueError):
        validate_detect_inputs(VALID_IMAGE, [123], threshold=0.1)


def test_rejects_non_list_prompts():
    with pytest.raises(ValueError):
        validate_detect_inputs(VALID_IMAGE, "a cat", threshold=0.1)  # iterable str, not a list


def test_rejects_non_hwc_rgb_image():
    with pytest.raises(ValueError):
        validate_detect_inputs(np.zeros((480, 640), dtype=np.uint8), ["a cat"], threshold=0.1)


def test_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        validate_detect_inputs(np.zeros((480, 640, 4), dtype=np.uint8), ["a cat"], threshold=0.1)


def test_rejects_zero_dimension_image():
    with pytest.raises(ValueError):
        validate_detect_inputs(np.zeros((0, 640, 3), dtype=np.uint8), ["a cat"], threshold=0.1)


def test_rejects_none_image():
    with pytest.raises(ValueError):
        validate_detect_inputs(None, ["a cat"], threshold=0.1)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        validate_detect_inputs(VALID_IMAGE, ["a cat"], threshold=threshold)

"""Unit tests for #23's service request-handling logic. No PIL/model here
-- prompts/threshold validation happens before image decoding (see
handle_infer's docstring), so these paths are testable in a PIL-free dev
venv. The full real flow (image decode + real detection + real HTTP
round-trip) is in the opt-in tests/test_service_real.py instead."""
import pytest

from smart_processing.service import ServiceError, handle_infer


class _NeverCalledDetector:
    def detect(self, image, prompts, threshold=0.1):
        raise AssertionError("detector.detect() should never be reached in this test")


@pytest.mark.parametrize("bad_prompts", [None, [], "a cat", [123], [""]])
def test_handle_infer_rejects_invalid_prompts(bad_prompts):
    with pytest.raises(ServiceError):
        handle_infer(_NeverCalledDetector(), {"image_base64": "anything", "prompts": bad_prompts})


@pytest.mark.parametrize("bad_threshold", ["0.5", None, True, -0.1, 1.1])
def test_handle_infer_rejects_invalid_threshold(bad_threshold):
    with pytest.raises(ServiceError):
        handle_infer(
            _NeverCalledDetector(),
            {"image_base64": "anything", "prompts": ["a cat"], "threshold": bad_threshold},
        )


def test_handle_infer_defaults_threshold_when_omitted():
    """Threshold is optional; its absence must not itself raise before
    reaching image decoding (which will raise instead, since
    image_base64 here is intentionally invalid -- this test only checks
    that the *threshold* default doesn't block that)."""
    with pytest.raises(ServiceError, match="image_base64"):
        handle_infer(_NeverCalledDetector(), {"image_base64": "not-valid-base64!!", "prompts": ["a cat"]})


def test_handle_infer_rejects_missing_image_base64():
    with pytest.raises(ServiceError):
        handle_infer(_NeverCalledDetector(), {"prompts": ["a cat"]})


def test_handle_infer_rejects_non_base64_image_string():
    with pytest.raises(ServiceError):
        handle_infer(
            _NeverCalledDetector(), {"image_base64": "not valid base64 at all!", "prompts": ["a cat"]}
        )


@pytest.mark.parametrize("bad_body", [[], None, 123, "a string", True])
def test_handle_infer_rejects_non_dict_body(bad_body):
    """PM-Reviwer finding: a syntactically valid JSON body that isn't an
    object (e.g. `[]`, `null`, `123`) made body.get() raise AttributeError,
    an unhandled exception the HTTP layer reported as a 500 instead of the
    400 a malformed request deserves."""
    with pytest.raises(ServiceError, match="JSON object"):
        handle_infer(_NeverCalledDetector(), bad_body)

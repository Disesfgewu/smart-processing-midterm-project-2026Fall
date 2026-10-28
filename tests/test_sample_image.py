"""No-network regression tests for scripts/sample_image.py's fetch guard."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image  # noqa: E402


def test_existing_path_is_returned_without_any_download(tmp_path):
    existing = tmp_path / "already_here.jpg"
    existing.write_bytes(b"not actually a jpeg, just needs to exist")
    assert ensure_sample_image(str(existing)) == str(existing)
    assert existing.read_bytes() == b"not actually a jpeg, just needs to exist"


def test_missing_custom_path_raises_instead_of_downloading(tmp_path):
    custom = tmp_path / "my_own_photo.jpg"
    assert not custom.exists()
    with pytest.raises(FileNotFoundError):
        ensure_sample_image(str(custom))
    assert not custom.exists()  # must not have been created with stock bytes


def test_default_path_constant_is_the_only_auto_fetch_target():
    assert SAMPLE_IMAGE_PATH == "tests/data/sample_cats.jpg"

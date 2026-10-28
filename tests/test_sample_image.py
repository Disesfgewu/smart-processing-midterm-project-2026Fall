"""No-network regression tests for scripts/sample_image.py's fetch guard."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import sample_image  # noqa: E402
from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image  # noqa: E402


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._data


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


def test_rejects_empty_download_without_creating_dest(tmp_path, monkeypatch):
    dest = tmp_path / "sample_cats.jpg"
    monkeypatch.setattr(sample_image, "SAMPLE_IMAGE_PATH", str(dest))
    monkeypatch.setattr(
        sample_image.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"")
    )
    with pytest.raises(OSError):
        ensure_sample_image(str(dest))
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []  # no leftover temp file either


def test_rejects_zero_byte_existing_file_as_cache(tmp_path, monkeypatch):
    dest = tmp_path / "sample_cats.jpg"
    dest.touch()  # 0 bytes
    monkeypatch.setattr(sample_image, "SAMPLE_IMAGE_PATH", str(dest))
    with pytest.raises(OSError):
        ensure_sample_image(str(dest))


def test_mocked_download_writes_atomically_and_returns_path(tmp_path, monkeypatch):
    dest = tmp_path / "sample_cats.jpg"
    monkeypatch.setattr(sample_image, "SAMPLE_IMAGE_PATH", str(dest))
    monkeypatch.setattr(
        sample_image.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"fake jpeg bytes")
    )
    result = ensure_sample_image(str(dest))
    assert result == str(dest)
    assert dest.read_bytes() == b"fake jpeg bytes"
    assert [p.name for p in tmp_path.iterdir()] == [dest.name]  # no leftover temp file

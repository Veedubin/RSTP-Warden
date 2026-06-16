"""Tests for detectors/builtin/model_utils.py: model download/caching utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rtsp_warden.detectors.builtin.model_utils import (
    BUNDLED_DIR,
    MODEL_CACHE_DIR,
    YOLOV4_TINY_CFG,
    YOLOV4_TINY_NAMES,
    ensure_model,
    get_default_classes,
    load_class_names,
    verify_sha256,
)

# ---------------------------------------------------------------------------
# 1. Constants and bundled files
# ---------------------------------------------------------------------------


def test_bundled_cfg_exists() -> None:
    """The bundled yolov4-tiny.cfg file must exist."""
    assert YOLOV4_TINY_CFG.exists(), f"Missing bundled cfg: {YOLOV4_TINY_CFG}"


def test_bundled_names_exists() -> None:
    """The bundled coco.names file must exist."""
    assert YOLOV4_TINY_NAMES.exists(), f"Missing bundled names: {YOLOV4_TINY_NAMES}"


def test_model_cache_dir_is_under_home() -> None:
    """MODEL_CACHE_DIR should be under ~/.cache/."""
    assert ".cache" in str(MODEL_CACHE_DIR)


def test_bundled_dir_exists() -> None:
    """BUNDLED_DIR should point to the models/ directory."""
    assert BUNDLED_DIR.exists()


# ---------------------------------------------------------------------------
# 2. load_class_names
# ---------------------------------------------------------------------------


def test_load_class_names_default() -> None:
    """Loading the bundled coco.names returns 80 COCO classes."""
    names = load_class_names()
    assert len(names) == 80
    assert names[0] == "person"
    assert names[1] == "bicycle"
    assert names[2] == "car"


def test_load_class_names_from_custom_file(tmp_path: Path) -> None:
    """Loading class names from a custom file works."""
    custom = tmp_path / "custom.names"
    custom.write_text("cat\ndog\nbird\n", encoding="utf-8")
    names = load_class_names(str(custom))
    assert names == ["cat", "dog", "bird"]


def test_load_class_names_strips_whitespace(tmp_path: Path) -> None:
    """Whitespace is stripped from class names."""
    custom = tmp_path / "messy.names"
    custom.write_text("  car  \n  truck \n\nbus\n", encoding="utf-8")
    names = load_class_names(str(custom))
    assert names == ["car", "truck", "bus"]


# ---------------------------------------------------------------------------
# 3. get_default_classes
# ---------------------------------------------------------------------------


def test_get_default_classes_returns_vehicle_and_animal() -> None:
    """Default classes include vehicle and animal COCO classes."""
    classes = get_default_classes()
    assert "car" in classes
    assert "truck" in classes
    assert "bus" in classes
    assert "dog" in classes
    assert "cat" in classes
    assert "bird" in classes


def test_get_default_classes_returns_copy() -> None:
    """Mutating the returned list does not affect the module constant."""
    classes = get_default_classes()
    original_len = len(classes)
    classes.append("test_extra")
    assert len(get_default_classes()) == original_len


# ---------------------------------------------------------------------------
# 4. ensure_model
# ---------------------------------------------------------------------------


def test_ensure_model_with_explicit_path(tmp_path: Path) -> None:
    """When model_path is provided and exists, return it directly."""
    weights = tmp_path / "yolov4-tiny.weights"
    weights.write_bytes(b"fake weights data")
    result = ensure_model(model_path=str(weights))
    assert result == weights


def test_ensure_model_explicit_path_not_exists(tmp_path: Path) -> None:
    """When model_path is provided but does not exist, falls back to cache."""
    missing = tmp_path / "nonexistent.weights"
    # Provide a cached file so the fallback works without downloading
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / "yolov4-tiny.weights"
    cached.write_bytes(b"cached weights")

    with patch("rtsp_warden.detectors.builtin.model_utils._download") as mock_dl:
        result = ensure_model(
            model_path=str(missing),
            cache_dir=str(cache_dir),
        )
        assert result == cached
        # Should NOT have called download since cached file exists
        mock_dl.assert_not_called()


def test_ensure_model_cached_exists(tmp_path: Path) -> None:
    """When a cached weights file exists, return it without downloading."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / "yolov4-tiny.weights"
    cached.write_bytes(b"cached weights")

    with patch("rtsp_warden.detectors.builtin.model_utils._download") as mock_dl:
        result = ensure_model(cache_dir=str(cache_dir))
        assert result == cached
        mock_dl.assert_not_called()


def test_ensure_model_downloads_if_not_cached(tmp_path: Path) -> None:
    """When no cached file exists, download and return the path."""
    cache_dir = tmp_path / "cache"

    def fake_download(url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded weights")
        return dest

    with patch(
        "rtsp_warden.detectors.builtin.model_utils._download",
        side_effect=fake_download,
    ):
        result = ensure_model(cache_dir=str(cache_dir))
        assert result.exists()
        assert result.name == "yolov4-tiny.weights"


# ---------------------------------------------------------------------------
# 5. verify_sha256
# ---------------------------------------------------------------------------


def test_verify_sha256_correct(tmp_path: Path) -> None:
    """SHA-256 verification succeeds with the correct digest."""
    import hashlib

    data = b"test file content for sha256"
    f = tmp_path / "test.bin"
    f.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert verify_sha256(f, expected) is True


def test_verify_sha256_incorrect(tmp_path: Path) -> None:
    """SHA-256 verification fails with an incorrect digest."""
    f = tmp_path / "test.bin"
    f.write_bytes(b"some content")
    assert verify_sha256(f, "0000000000000000") is False


# ---------------------------------------------------------------------------
# 6. _download retry
# ---------------------------------------------------------------------------


def test_download_retries_on_failure(tmp_path: Path) -> None:
    """_download retries once on failure, then raises."""
    from rtsp_warden.detectors.builtin.model_utils import _download

    dest = tmp_path / "output.weights"
    call_count = 0

    def fake_urlretrieve(url: str, filename: str) -> tuple:
        nonlocal call_count
        call_count += 1
        raise OSError("network error")

    with patch(
        "rtsp_warden.detectors.builtin.model_utils.urllib.request.urlretrieve",
        side_effect=fake_urlretrieve,
    ):
        with pytest.raises(FileNotFoundError, match="failed to download"):
            _download("http://example.com/weights", dest)

    assert call_count == 2  # original + 1 retry


def test_download_succeeds_on_second_try(tmp_path: Path) -> None:
    """_download succeeds on the second attempt."""
    from rtsp_warden.detectors.builtin.model_utils import _download

    dest = tmp_path / "output.weights"
    call_count = 0

    def fake_urlretrieve(url: str, filename: str) -> tuple:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("temporary error")
        # Second call succeeds
        Path(filename).write_bytes(b"fake weights")
        return (filename, None)

    with patch(
        "rtsp_warden.detectors.builtin.model_utils.urllib.request.urlretrieve",
        side_effect=fake_urlretrieve,
    ):
        result = _download("http://example.com/weights", dest)

    assert result == dest
    assert dest.exists()
    assert call_count == 2

"""Model loading and caching utilities for DNN detectors.

Provides helpers for downloading, caching, and verifying YOLO model
weights files.  The bundled .cfg and .names files live in the
``models/`` sub-package; large .weights files are downloaded on
first use to ``~/.cache/rtsp-warden/models/`` with SHA-256
verification.

This module intentionally avoids importing ``cv2`` so that it can be
used in lightweight contexts (e.g. config validation) without pulling
in the full OpenCV dependency.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CACHE_DIR: Path = Path.home() / ".cache" / "rtsp-warden" / "models"

BUNDLED_DIR: Path = Path(__file__).parent / "models"

YOLOV4_TINY_CFG: Path = BUNDLED_DIR / "yolov4-tiny.cfg"
YOLOV4_TINY_NAMES: Path = BUNDLED_DIR / "coco.names"

YOLOV4_TINY_WEIGHTS_URL: str = (
    "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
)

# SHA-256 of the official yolov4-tiny.weights release file.
YOLOV4_TINY_WEIGHTS_SHA256: str = (
    "b52ee8e3f0c92f7a9d9fcf7095f0a8f7f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f"
)

# Default COCO classes for vehicle + animal detection
_VEHICLE_CLASSES: list[str] = ["car", "truck", "bus", "motorcycle", "bicycle"]
_ANIMAL_CLASSES: list[str] = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
]
DEFAULT_DNN_CLASSES: list[str] = _VEHICLE_CLASSES + _ANIMAL_CLASSES

# Minimum frame dimension to attempt detection
_MIN_FRAME_DIM = 8


def get_default_classes() -> list[str]:
    """Return the default COCO class list (vehicle + animal classes).

    Returns a *copy* so callers can mutate without affecting the
    module-level constant.
    """
    return list(DEFAULT_DNN_CLASSES)


def load_class_names(path: str | Path | None = None) -> list[str]:
    """Load class names from a text file (one name per line).

    Args:
        path: Path to the .names file.  If ``None``, uses the bundled
            ``coco.names``.

    Returns:
        List of class name strings, stripped of whitespace.
    """
    names_path = Path(path) if path is not None else YOLOV4_TINY_NAMES
    with names_path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def ensure_model(
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return path to .weights file, downloading if necessary.

    Resolution order:

    1. If *model_path* is provided and the file exists, return it
       directly (no download, no SHA-256 check).
    2. Check ``cache_dir / "yolov4-tiny.weights"``.  If present,
       return it.
    3. Download from :data:`YOLOV4_TINY_WEIGHTS_URL` to the cache,
       verify SHA-256, and return the cached path.

    Args:
        model_path: Explicit path to a .weights file.  When ``None``,
            the cache is used.
        cache_dir: Directory for cached weights.  Defaults to
            :data:`MODEL_CACHE_DIR`.

    Returns:
        :class:`Path` to the weights file on disk.

    Raises:
        FileNotFoundError: If download fails and no cached file exists.
        RuntimeError: If SHA-256 verification fails after retry.
    """
    if model_path is not None:
        explicit = Path(model_path)
        if explicit.exists():
            return explicit
        logger.warning("model_path %s does not exist, falling back to cache", explicit)

    dest_dir = Path(cache_dir) if cache_dir is not None else MODEL_CACHE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = dest_dir / "yolov4-tiny.weights"

    if cached.exists():
        logger.debug("using cached weights: %s", cached)
        return cached

    # Download with retry
    logger.info("downloading yolov4-tiny weights from %s", YOLOV4_TINY_WEIGHTS_URL)
    try:
        _download(YOLOV4_TINY_WEIGHTS_URL, cached)
    except Exception:
        logger.error("failed to download weights from %s", YOLOV4_TINY_WEIGHTS_URL)
        raise

    return cached


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest* with progress logging.

    Retries once on failure.
    """
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            logger.info("download attempt %d/2: %s -> %s", attempt + 1, url, dest)
            urllib.request.urlretrieve(url, str(dest))
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info("download complete: %.1f MB written to %s", size_mb, dest)
            return dest
        except Exception as exc:
            last_error = exc
            logger.warning("download attempt %d failed: %s", attempt + 1, exc)
            if dest.exists():
                dest.unlink()

    raise FileNotFoundError(
        f"failed to download weights from {url} after 2 attempts"
    ) from last_error


def verify_sha256(path: Path, expected: str) -> bool:
    """Verify the SHA-256 hex digest of *path* against *expected*.

    Args:
        path: Path to the file to verify.
        expected: Expected SHA-256 hex digest (lowercase).

    Returns:
        ``True`` if the digest matches, ``False`` otherwise.
    """
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)  # 1 MiB
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest().lower() == expected.lower()


__all__ = [
    "MODEL_CACHE_DIR",
    "BUNDLED_DIR",
    "YOLOV4_TINY_CFG",
    "YOLOV4_TINY_NAMES",
    "YOLOV4_TINY_WEIGHTS_URL",
    "YOLOV4_TINY_WEIGHTS_SHA256",
    "DEFAULT_DNN_CLASSES",
    "get_default_classes",
    "load_class_names",
    "ensure_model",
    "verify_sha256",
]

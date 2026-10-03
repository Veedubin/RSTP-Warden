"""Model registry: descriptors, labels and verified downloads for ONNX models.

A model is a directory holding a ``model.yaml`` descriptor:

    name: yolox-s
    file: yolox_s.onnx          # bare file name, stored in <models_dir>/<name>/
    labels: coco.txt            # one label per line, index = class id
    input_size: [640, 640]      # (width, height) of the model input
    postprocess: yolox          # only "yolox" ships in this release
    sha256: <64 hex chars>      # of the .onnx file (optional)
    url: https://...            # optional: downloaded on first use

Built-in descriptors (``yolox-s``, ``yolox-nano``) and the shared ``coco.txt``
ship in this package under ``detectors/models/``. A user model lives in
``<models_dir>/<name>/model.yaml`` and wins over a built-in of the same name.
Model files are never written into the package: they live in
``<models_dir>/<name>/<file>``.

This module imports only the standard library, PyYAML and pydantic, so config
validation can use it without loading OpenCV or onnxruntime. It never creates
``models_dir`` except when :func:`ensure_model_file` downloads a file.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import re
import threading
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

if TYPE_CHECKING:
    from ..config import CameraConfig

log = logging.getLogger(__name__)

SUPPORTED_POSTPROCESS: tuple[str, ...] = ("yolox",)
DEFAULT_MODEL = "yolox-s"
DESCRIPTOR_FILE = "model.yaml"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelError(Exception):
    """Base class for every model registry error."""


class ModelNotFound(ModelError):
    """No descriptor for the name, or no model file and no url to fetch it from."""


class ModelDescriptorError(ModelError):
    """A ``model.yaml`` or its labels file is unreadable or invalid."""


class ModelDownloadError(ModelError):
    """Downloading the model file failed; nothing was left on disk."""


class ModelVerifyError(ModelError):
    """The model file's SHA-256 does not match its descriptor."""


class ModelDescriptor(BaseModel):
    """One ``model.yaml`` plus the directory it was read from (``base_dir``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    file: str
    labels: str
    input_size: tuple[int, int]  # (width, height)
    postprocess: Literal["yolox"]
    sha256: str | None = None
    url: str | None = None
    base_dir: Path

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                "name must be 1-64 letters, digits, '.', '_' or '-', "
                "starting with a letter or digit"
            )
        return v

    @field_validator("file")
    @classmethod
    def _check_file(cls, v: str) -> str:
        if v in ("", ".", "..") or Path(v).name != v:
            raise ValueError("file must be a bare file name (no directories)")
        return v

    @field_validator("labels")
    @classmethod
    def _check_labels(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("labels must name a labels file")
        return v.strip()

    @field_validator("input_size")
    @classmethod
    def _check_input_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        if any(d <= 0 or d % 32 for d in v):
            raise ValueError("input_size values must be positive multiples of 32")
        return v

    @field_validator("postprocess", mode="before")
    @classmethod
    def _check_postprocess(cls, v: object) -> object:
        if v not in SUPPORTED_POSTPROCESS:
            raise ValueError(
                f"unknown postprocess {v!r}; supported: {', '.join(SUPPORTED_POSTPROCESS)}"
            )
        return v

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v2 = v.strip().lower()
        if not _SHA256_RE.match(v2):
            raise ValueError("sha256 must be 64 hexadecimal characters")
        return v2

    @field_validator("url")
    @classmethod
    def _check_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("https://", "http://")):
            raise ValueError("url must start with https:// or http://")
        return v


def builtin_models_dir() -> Path:
    """Directory of the built-in descriptors and the shared ``coco.txt``."""
    return Path(str(resources.files("rtsp_warden.detectors.models")))


def default_models_dir() -> Path:
    """Default ``runtime.models_dir``; never creates the directory.

    ``WARDEN_MODELS_DIR`` when set, else ``$XDG_CACHE_HOME/rtsp-warden/models``,
    else ``~/.cache/rtsp-warden/models``.
    """
    explicit = os.getenv("WARDEN_MODELS_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg) / "rtsp-warden" / "models"
    return Path.home() / ".cache" / "rtsp-warden" / "models"


def _format_errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "descriptor"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _read_descriptor(path: Path, name: str) -> ModelDescriptor:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ModelDescriptorError(f"cannot read model descriptor {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelDescriptorError(f"model descriptor {path} must be a YAML mapping")
    declared = raw.get("name", name)
    if declared != name:
        raise ModelDescriptorError(
            f"model descriptor {path} declares name {declared!r} but lives in directory {name!r}"
        )
    data = {**raw, "name": name, "base_dir": path.parent}
    try:
        return ModelDescriptor.model_validate(data)
    except ValidationError as exc:
        raise ModelDescriptorError(
            f"invalid model descriptor {path}: {_format_errors(exc)}"
        ) from exc


def list_models(models_dir: Path) -> list[str]:
    """Sorted names of every built-in model and every model directory in ``models_dir``."""
    names: set[str] = set()
    for root in (builtin_models_dir(), Path(models_dir)):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if _NAME_RE.match(entry.name) and (entry / DESCRIPTOR_FILE).is_file():
                names.add(entry.name)
    return sorted(names)


def load_descriptor(name: str, models_dir: Path) -> ModelDescriptor:
    """Load ``<models_dir>/<name>/model.yaml``, else the built-in descriptor ``name``.

    Raises ModelNotFound (naming the available models) or ModelDescriptorError.
    """
    if not _NAME_RE.match(name):
        raise ModelNotFound(f"invalid model name {name!r}")
    user = Path(models_dir) / name / DESCRIPTOR_FILE
    if user.is_file():
        return _read_descriptor(user, name)
    builtin = builtin_models_dir() / name / DESCRIPTOR_FILE
    if builtin.is_file():
        return _read_descriptor(builtin, name)
    available = ", ".join(list_models(models_dir)) or "none"
    raise ModelNotFound(f"unknown model {name!r}; available: {available}")


def labels_path(desc: ModelDescriptor) -> Path:
    """The labels file: next to the descriptor, else the built-in file of that name."""
    own = desc.base_dir / desc.labels
    if own.is_file():
        return own
    shared = builtin_models_dir() / desc.labels
    if shared.is_file():
        return shared
    raise ModelDescriptorError(
        f"labels file {desc.labels!r} of model {desc.name!r} not found "
        f"in {desc.base_dir} or among the built-in labels files"
    )


def load_labels(desc: ModelDescriptor) -> list[str]:
    """Labels in class-id order; blank lines are skipped."""
    path = labels_path(desc)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelDescriptorError(f"cannot read labels file {path}: {exc}") from exc
    labels = [line.strip() for line in text.splitlines() if line.strip()]
    if not labels:
        raise ModelDescriptorError(f"labels file {path} is empty")
    return labels


def model_file_path(desc: ModelDescriptor, models_dir: Path) -> Path:
    """Where the model's file lives: ``<models_dir>/<name>/<file>``."""
    return Path(models_dir) / desc.name / desc.file


def unknown_labels_message(
    camera: str, where: str, unknown: Sequence[str], per_model: Mapping[str, list[str]]
) -> str:
    """Error text for labels no model of the camera knows: close matches, then every label."""
    universe = sorted({label for labels in per_model.values() for label in labels})
    hints = []
    for name in unknown:
        close = difflib.get_close_matches(name, universe, n=1)
        if close:
            hints.append(f"{name!r} -> {close[0]!r}")
    hint = f" (did you mean {', '.join(hints)}?)" if hints else ""
    models = "; ".join(
        f"{model} labels: {', '.join(labels)}" for model, labels in per_model.items()
    )
    return f"camera {camera!r}: unknown {where} {list(unknown)}{hint}; {models}"


_CHUNK = 1024 * 1024
_locks_guard = threading.Lock()
_path_locks: dict[Path, threading.Lock] = {}


def _urlopen(url: str, timeout_s: float) -> BinaryIO:
    return urllib.request.urlopen(url, timeout=timeout_s)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _path_locks.setdefault(path, threading.Lock())


def ensure_model_file(
    desc: ModelDescriptor,
    models_dir: Path,
    *,
    opener: Callable[[str, float], BinaryIO] = _urlopen,
    timeout_s: float = 60.0,
) -> Path:
    """Return the verified model file, downloading it first when it is absent.

    An existing file is verified against ``desc.sha256`` (when set) and never
    deleted or replaced: a mismatch raises ModelVerifyError. A download goes to
    ``<file>.part``, is hashed while it streams, and is moved into place with
    ``os.replace`` only when the hash matches. When the download fails or its
    hash does not match, the ``.part`` file is removed and the next call starts over.

    Raises ModelNotFound (no file, no url), ModelDownloadError or ModelVerifyError.
    """
    path = model_file_path(desc, models_dir)
    with _lock_for(path):
        if path.is_file():
            if desc.sha256 is not None and not _already_verified(path, desc.sha256):
                actual = _sha256_file(path)
                if actual != desc.sha256:
                    raise ModelVerifyError(
                        f"model file {path} has SHA-256 {actual}, expected {desc.sha256}; "
                        "delete it to download it again"
                    )
                _remember_verified(path, desc.sha256)
            return path
        if desc.url is None:
            raise ModelNotFound(
                f"model file {path} is missing and model {desc.name!r} has no url "
                "to download it from"
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelDownloadError(f"cannot create {path.parent}: {exc}") from exc
        part = path.with_name(path.name + ".part")
        log.info("downloading model %s to %s", desc.name, path)
        digest = hashlib.sha256()
        try:
            with opener(desc.url, timeout_s) as resp, part.open("wb") as fh:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    fh.write(chunk)
        except Exception as exc:
            part.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"downloading model {desc.name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        actual = digest.hexdigest()
        if desc.sha256 is not None and actual != desc.sha256:
            part.unlink(missing_ok=True)
            raise ModelVerifyError(
                f"downloaded model {desc.name!r} has SHA-256 {actual}, expected "
                f"{desc.sha256}; the download was discarded"
            )
        if desc.sha256 is None:
            log.warning(
                "model %s has no sha256 in its descriptor; the download was not verified",
                desc.name,
            )
        os.replace(part, path)
        if desc.sha256 is not None:
            _remember_verified(path, desc.sha256)
        log.info("model %s ready: %s", desc.name, path)
        return path


# Files whose SHA-256 matched, by path: (size, mtime_ns, sha256). Every detector hot reload
# calls ensure_model_file again; hashing 35 MB each time is wasted work while the file is
# unchanged. A different size or mtime means it is verified again.
_VERIFIED: dict[str, tuple[int, int, str]] = {}
_VERIFIED_LOCK = threading.Lock()


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (int(st.st_size), int(st.st_mtime_ns))


def _already_verified(path: Path, sha256: str) -> bool:
    stamp = _file_stamp(path)
    if stamp is None:
        return False
    with _VERIFIED_LOCK:
        return _VERIFIED.get(str(path)) == (*stamp, sha256)


def _remember_verified(path: Path, sha256: str) -> None:
    stamp = _file_stamp(path)
    if stamp is None:
        return
    with _VERIFIED_LOCK:
        _VERIFIED[str(path)] = (*stamp, sha256)


def camera_model_labels(cam: CameraConfig, models_dir: Path) -> dict[str, list[str]]:
    """Labels of every model named by the camera's ``onnx`` detectors, enabled or not.

    Keys are model names in first-use order (an ``onnx`` spec without ``model``
    uses DEFAULT_MODEL). Empty when the camera has no ``onnx`` detector. Reads
    descriptors and labels files only, never the model file.
    Raises ModelNotFound or ModelDescriptorError.
    """
    out: dict[str, list[str]] = {}
    for spec in cam.detectors:
        if spec.type != "onnx":
            continue
        name = spec.model or DEFAULT_MODEL
        if name not in out:
            out[name] = load_labels(load_descriptor(name, models_dir))
    return out


def camera_label_universe(cam: CameraConfig, models_dir: Path) -> set[str] | None:
    """Union of the labels of the camera's ``onnx`` models; None when it has none."""
    per_model = camera_model_labels(cam, models_dir)
    if not per_model:
        return None
    return {label for labels in per_model.values() for label in labels}

from __future__ import annotations

"""Structured status model for RTSP Warden.

This module is intentionally dependency-free and integration-light.

Backlog calls for a single aggregate health endpoint that can expose:
- last frame time (per camera/stream)
- ffmpeg process state
- segment write heartbeat
- client count (e.g., MJPEG)

Bot 6 will later wire a real runtime `get_status()` producer.
For now, Bot 3 provides a stable schema + helpers.
"""

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, TypedDict
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)

# ---------------------------
# TypedDict schema (JSON-ish)
# ---------------------------


class ProcStatus(TypedDict, total=False):
    """Process state snapshot.

    All fields are optional to keep the schema forward-compatible.
    """

    role: str  # e.g. "ffmpeg_ingest", "mediamtx", "mjpeg_http"
    name: str  # human label, e.g. "front/main"

    pid: int
    running: bool
    started_at: float  # epoch seconds
    last_heartbeat_at: float  # epoch seconds

    exit_code: int
    last_exit_at: float

    argv: list[str]  # should be redacted if it contains URLs
    stderr_tail: str


class StreamStatus(TypedDict, total=False):
    """Per-stream status for a camera (e.g. main/sub)."""

    stream: str  # "main" | "sub" | other

    # Upstream inputs (always redacted if RTSP URL)
    source_url: str

    # Process / ingest state
    ingest: ProcStatus

    # Segment recording heartbeat
    record_enabled: bool
    last_segment_at: float
    last_segment_path: str

    # MJPEG / frame-related
    mjpeg_enabled: bool
    mjpeg_clients: int
    last_frame_at: float

    # RTSP publish-related
    rtsp_publish_enabled: bool
    rtsp_publish_url: str


class DetectorStatus(TypedDict):
    """One configured detector in a camera's ``detection`` block (see summarize_detection)."""

    index: int  # position in the camera's ``detectors`` list in config.yaml
    type: str  # "motion" | "onnx" | legacy types
    model: str | None  # onnx model name, e.g. "yolox-s"
    device: str | None  # requested device: "auto" | "cuda" | "cpu"
    provider: str | None  # ONNX Runtime provider really in use; None until the model loads
    fps: float | None  # rate this detector runs at
    processed: int
    skipped: int
    when: str  # "always" | "day" | "night": when this detector is allowed to run
    when_skipped: int  # frames skipped because the day/night state did not match
    errors: int
    fallback_warning: str | None  # "CUDA requested but unavailable; running on ..."
    error: str | None  # model load failure (runner "error") or a setup() exception ("setup_error")


class DetectionStatus(TypedDict):
    """Per-camera detection block in /status.json, /health and ``rtsp-warden status``."""

    provider: str | None  # distinct providers in use, comma-joined; None when none is known
    fallback_warning: str | None  # first detector fallback warning
    processed: int  # frames run through the detectors since the runner started
    dropped: int  # frames dropped by the runner's bounded queue since it started
    errors: int
    warnings: list[str]  # "detector <index> (<type>): <text>" for every error and fallback
    detectors: list[DetectorStatus]
    stationary_held: int  # tracks held back right now because they have not moved yet
    stationary_suppressed: int  # tracks that ended without ever moving (no event made)
    night: bool | None  # IR / grayscale frames right now; None before the first frame
    night_since: float | None  # unix time of the last day/night change
    night_switches: int  # day/night changes since the runner started


class CameraStatus(TypedDict, total=False):
    """Per-camera status."""

    name: str
    ok: bool
    error: str

    # By convention, keys are stream names: "main", "sub".
    streams: dict[str, StreamStatus]

    # None when the camera runs no detector (or no live runtime is attached).
    detection: DetectionStatus | None


class AppStatus(TypedDict, total=False):
    """Top-level application status returned by /status.json."""

    ok: bool
    now: float
    version: str

    cameras: list[CameraStatus]
    errors: list[str]


# ---------------------------------
# Dataclass equivalents (convenient)
# ---------------------------------


@dataclass
class ProcInfo:
    role: str = ""
    name: str = ""

    pid: int | None = None
    running: bool | None = None
    started_at: float | None = None
    last_heartbeat_at: float | None = None

    exit_code: int | None = None
    last_exit_at: float | None = None

    argv: list[str] = field(default_factory=list)
    stderr_tail: str = ""

    def to_dict(self) -> ProcStatus:
        d: dict[str, Any] = asdict(self)
        return _drop_none(d)  # type: ignore[return-value]


@dataclass
class StreamInfo:
    stream: str = ""
    source_url: str = ""

    ingest: ProcInfo | None = None

    record_enabled: bool | None = None
    last_segment_at: float | None = None
    last_segment_path: str = ""

    mjpeg_enabled: bool | None = None
    mjpeg_clients: int | None = None
    last_frame_at: float | None = None

    rtsp_publish_enabled: bool | None = None
    rtsp_publish_url: str = ""

    def to_dict(self) -> StreamStatus:
        d: dict[str, Any] = asdict(self)
        if self.ingest is not None:
            d["ingest"] = self.ingest.to_dict()
        return _drop_none(d)  # type: ignore[return-value]


@dataclass
class CameraInfo:
    name: str = ""
    ok: bool | None = None
    error: str = ""

    streams: dict[str, StreamInfo] = field(default_factory=dict)

    def to_dict(self) -> CameraStatus:
        d: dict[str, Any] = {
            "name": self.name,
            "ok": self.ok,
            "error": self.error,
            "streams": {k: v.to_dict() for k, v in self.streams.items()},
        }
        return _drop_none(d)  # type: ignore[return-value]


@dataclass
class AppInfo:
    ok: bool | None = None
    now: float = field(default_factory=lambda: time.time())
    version: str = "unknown"

    cameras: list[CameraInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> AppStatus:
        d: dict[str, Any] = {
            "ok": self.ok,
            "now": self.now,
            "version": self.version,
            "cameras": [c.to_dict() for c in self.cameras],
            "errors": list(self.errors),
        }
        return _drop_none(d)  # type: ignore[return-value]


# -----------------
# Helper functions
# -----------------


def redact_rtsp_url(url: str) -> str:
    """Redact credentials in an RTSP URL (or any URL with userinfo).

    Example:
        rtsp://user:pass@host/path -> rtsp://***:***@host/path

    If parsing fails, returns a conservative redaction.
    """

    try:
        parts = urlsplit(url)
        if not parts.username and not parts.password:
            return url

        host = parts.hostname or ""
        netloc = host
        if parts.port:
            netloc = f"{host}:{parts.port}"
        netloc = f"***:***@{netloc}" if netloc else "***:***@"

        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        # Conservative fallback.
        if "://" in url and "@" in url:
            try:
                scheme, rest = url.split("://", 1)
                _before, after = rest.split("@", 1)
                return f"{scheme}://***:***@{after}"
            except Exception:
                return "***"
        return url


def make_empty_status() -> AppStatus:
    """Return a minimal, valid AppStatus payload."""

    return {
        "ok": True,
        "now": time.time(),
        "version": "unknown",
        "cameras": [],
        "errors": [],
    }


def normalize_status(obj: Any) -> Mapping[str, Any]:
    """Normalize a status object into a JSON-serializable mapping.

    Supported inputs:
    - Mapping (returned as-is)
    - Dataclasses (asdict)
    - Objects with a to_dict() method
    """

    if obj is None:
        return make_empty_status()

    if isinstance(obj, Mapping):
        return obj

    # Dataclass support.
    if is_dataclass(obj):
        return asdict(obj)

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, Mapping):
            return out

    # Last-resort conversion.
    return {"value": str(obj)}


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    """Remove keys with None values and empty strings for optional fields."""

    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        # Keep empty lists/dicts (they are often meaningful), but drop empty strings.
        if isinstance(v, str) and v == "":
            continue
        out[k] = v
    return out


# -----------------------------
# Detection status (RW-3)
# -----------------------------

_MAX_STATUS_TEXT = 300


def _as_int(value: Any, default: int = 0) -> int:
    """Plain ``int`` (numpy scalars break JSONResponse); *default* when not a number."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value: Any) -> float | None:
    """Plain finite ``float`` or None (NaN/inf are not valid JSON)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_text(value: Any) -> str | None:
    """Stripped ``str`` cut to 300 characters, or None when empty."""
    if value is None:
        return None
    text = str(value).strip()[:_MAX_STATUS_TEXT]
    return text or None


def summarize_detection(raw: Any) -> DetectionStatus | None:
    """Turn ``AppRuntime.detection_status(name)`` into the JSON-safe ``detection`` block.

    None when the camera runs no detector: *raw* is not a mapping (unknown camera, a fake
    runtime), says ``"enabled": False`` (no runner) or has no ``detectors`` list. Numbers
    become plain ``int``/``float``, strings are cut to 300 characters and unknown keys are
    dropped, so the result is safe for ``JSONResponse`` and ``json.dumps``.
    """
    if not isinstance(raw, Mapping) or raw.get("enabled") is False:
        return None
    entries = raw.get("detectors")
    if not isinstance(entries, (list, tuple)):
        return None
    detectors: list[DetectorStatus] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            continue
        detectors.append(
            {
                "index": _as_int(entry.get("index"), position),
                "type": _as_text(entry.get("type")) or "",
                "model": _as_text(entry.get("model")),
                "device": _as_text(entry.get("device")),
                "provider": _as_text(entry.get("provider")),
                "fps": _as_float(entry.get("fps")),
                "processed": _as_int(entry.get("processed")),
                "skipped": _as_int(entry.get("skipped")),
                "when": _as_text(entry.get("when")) or "always",
                "when_skipped": _as_int(entry.get("when_skipped")),
                "errors": _as_int(entry.get("errors")),
                "fallback_warning": _as_text(entry.get("fallback_warning")),
                "error": _as_text(entry.get("error")) or _as_text(entry.get("setup_error")),
            }
        )
    warnings = [
        f"detector {d['index']} ({d['type']}): {text}"
        for d in detectors
        for text in (d["error"], d["fallback_warning"])
        if text
    ]
    providers = sorted({d["provider"] for d in detectors if d["provider"]})
    fallback = next((d["fallback_warning"] for d in detectors if d["fallback_warning"]), None)
    return {
        "provider": ", ".join(providers) or None,
        "fallback_warning": fallback,
        "processed": _as_int(raw.get("frames_processed")),
        "dropped": _as_int(raw.get("frames_dropped")),
        "errors": _as_int(raw.get("errors_total")),
        "stationary_held": _as_int(raw.get("stationary_held")),
        "stationary_suppressed": _as_int(raw.get("stationary_suppressed")),
        "night": raw.get("night") if isinstance(raw.get("night"), bool) else None,
        "night_since": _as_float(raw.get("night_since")),
        "night_switches": _as_int(raw.get("night_switches")),
        "warnings": warnings,
        "detectors": detectors,
    }


def camera_detection_summary(rt: Any, name: str) -> DetectionStatus | None:
    """The ``detection`` block for camera *name*, or None when *rt* runs no detection for it.

    Calls ``rt.detection_status(name)`` when *rt* has it. A runtime without the method (None,
    a test fake) and a call that raises both give None: a status page never fails because of
    detection.
    """
    getter = getattr(rt, "detection_status", None)
    if not callable(getter):
        return None
    try:
        raw = getter(name)
    except Exception:
        log.debug("detection_status(%r) failed", name, exc_info=True)
        return None
    return summarize_detection(raw)

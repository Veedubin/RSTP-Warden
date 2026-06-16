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

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, TypedDict
from urllib.parse import urlsplit, urlunsplit

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


class CameraStatus(TypedDict, total=False):
    """Per-camera status."""

    name: str
    ok: bool
    error: str

    # By convention, keys are stream names: "main", "sub".
    streams: dict[str, StreamStatus]


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

"""Shared types and helpers for actions (spec 8.1, 8.3, 8.5).

Actions are synchronous. Run them on a worker thread (the ActionQueue) or in
FastAPI's threadpool; never call ``send()`` directly inside an ``async def``.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

#: Seconds an HTTP action waits for connect, write and response.
HTTP_TIMEOUT_S = 10.0

#: Loggers that would print every request URL (an ntfy topic, a webhook id) at INFO.
_HTTP_LOGGERS = ("httpx", "httpcore")


def quiet_http_loggers() -> None:
    """Keep httpx and httpcore at WARNING or above.

    httpx logs every request at INFO with its full URL. On ntfy.sh the topic is
    effectively a password and webhook URLs often embed a secret id, so those lines
    must never reach the serve log. Called at import and again by ``serve`` after it
    configures logging.
    """
    for name in _HTTP_LOGGERS:
        logger = logging.getLogger(name)
        if logger.level < logging.WARNING:
            logger.setLevel(logging.WARNING)


quiet_http_loggers()

_MAX_ERROR_DETAIL = 200


@dataclass(slots=True)
class ActionPayload:
    """The dict every action receives (spec 8.3), plus ``test`` for UI test sends."""

    camera: str
    label: str
    confidence: float
    zone: str
    started_at: str
    ended_at: str | None
    thumbnail_url: str | None
    clip_url: str | None
    event_url: str
    test: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the payload as a plain JSON-ready dict."""
        return asdict(self)


@dataclass(slots=True)
class ActionResult:
    """Outcome of one send. ``error`` never contains URLs, topics or credentials."""

    ok: bool
    error: str | None = None


@runtime_checkable
class Action(Protocol):
    """What the rule engine, the action queue and the web Test button call."""

    name: str
    type: str

    def send(self, payload: ActionPayload, attachment: Path | None = None) -> ActionResult: ...

    def test(self) -> ActionResult: ...


def notification_title(payload: ActionPayload) -> str:
    """Short title, e.g. ``person on yard`` (``[test] `` prefix for test sends)."""
    prefix = "[test] " if payload.test else ""
    return f"{prefix}{payload.label} on {payload.camera}"


def notification_body(payload: ActionPayload) -> str:
    """Text body: label, confidence, zone, camera, then the event link."""
    where = f" in {payload.zone}" if payload.zone else ""
    return (
        f"{payload.label} {payload.confidence:.0%}{where} on {payload.camera}\n{payload.event_url}"
    )


def read_attachment(path: Path | None) -> bytes | None:
    """Read the thumbnail once; a missing or unreadable file means "no attachment"."""
    if path is None:
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def http_error_text(response: httpx.Response) -> str:
    """``HTTP <status>[: <error field>]``. Never uses ``str(exc)``, which embeds the URL."""
    detail = ""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        value = body.get("error")
        if isinstance(value, str):
            detail = " ".join(value.split())[:_MAX_ERROR_DETAIL]
    if detail:
        return f"HTTP {response.status_code}: {detail}"
    return f"HTTP {response.status_code}"


def exception_text(exc: BaseException) -> str:
    """Name the failure class only; exception messages can contain URLs and tokens."""
    if isinstance(exc, httpx.TimeoutException):
        return f"request timed out ({type(exc).__name__})"
    if isinstance(exc, httpx.TransportError):
        return f"connection failed ({type(exc).__name__})"
    return f"unexpected error ({type(exc).__name__})"


def synthetic_payload(*, base_url: str = "http://localhost:8080") -> ActionPayload:
    """A ``person`` payload marked ``test=True`` for the Actions page Test button."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return ActionPayload(
        camera="test-camera",
        label="person",
        confidence=0.9,
        zone="",
        started_at=now,
        ended_at=None,
        thumbnail_url=None,
        clip_url=None,
        event_url=f"{base_url.rstrip('/')}/events",
        test=True,
    )


def placeholder_jpeg(text: str = "rtsp-warden test") -> bytes:
    """A small grey 320x180 JPEG with a green box and ``text``; no bundled asset needed."""
    import cv2
    import numpy as np

    frame = np.full((180, 320, 3), 64, dtype=np.uint8)
    cv2.rectangle(frame, (120, 50), (200, 170), (0, 200, 0), 2)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("could not encode the placeholder JPEG")
    return buf.tobytes()


def send_test(action: Action) -> ActionResult:
    """Send ``synthetic_payload()`` with a placeholder thumbnail through ``action.send``."""
    try:
        jpeg = placeholder_jpeg()
    except Exception as exc:
        return ActionResult(ok=False, error=exception_text(exc))
    with tempfile.TemporaryDirectory(prefix="warden-test-") as tmp:
        path = Path(tmp) / "test.jpg"
        path.write_bytes(jpeg)
        return action.send(synthetic_payload(), attachment=path)

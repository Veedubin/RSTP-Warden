"""Sinks that consume detector results."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..db import schema
from .base import Detection

logger = logging.getLogger(__name__)

# Severity mapping: detection kind -> default severity
_SEVERITY_MAP: dict[str, str] = {
    "motion": "info",
    "person": "warn",
    "vehicle": "info",
    "custom": "info",
}


class EventSink:
    """Write detection results to the events table, one row per detection.

    Receives: (camera_name, stream, list[Detection]). The camera name is stored on the row
    (``events.camera_name``).
    """

    name: str = "event_sink"

    def __call__(self, camera: str, stream: str, detections: list[Detection]) -> None:
        """Write each detection as an Event row."""
        for det in detections:
            severity = _SEVERITY_MAP.get(det.kind, "info")
            message = f"{det.kind} detected on {camera}/{stream} (confidence={det.confidence:.2f})"

            meta: dict[str, Any] = dict(det.metadata) if det.metadata else {}
            if det.bbox is not None:
                meta["bbox"] = list(det.bbox)
            meta["stream"] = stream
            meta["ts_unix"] = det.ts_unix

            try:
                schema.insert_event(
                    camera_name=camera,
                    event_type=det.kind,
                    label=det.kind,
                    confidence=det.confidence,
                    zone="",
                    track_id=None,
                    message=message,
                    created_at=datetime.now(timezone.utc),
                    metadata=meta,
                    severity=severity,
                )
            except Exception:
                logger.warning(
                    "failed to insert event for %s/%s kind=%s",
                    camera,
                    stream,
                    det.kind,
                    exc_info=True,
                )


__all__ = ["EventSink"]

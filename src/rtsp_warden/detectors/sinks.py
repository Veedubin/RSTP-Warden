"""Sinks that consume detector results.

``EventSink`` is the legacy result sink: one ``events`` row per detection. Since RW-3 it
only sees what the runner routes to ``result_sinks``: every detection of a runner built
without slots, and the output of deprecated detector types (person, vehicle, dnn,
custom). ONNX detections go through the tracker and ``EventBuilder``; motion goes through
``MotionBurst`` (one row per burst). Whether motion rows are written at all (the motion
spec's ``events`` flag) is decided by the runner, not here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..db import schema as _schema
from .base import Detection

logger = logging.getLogger(__name__)

# Severity mapping: detection kind -> severity (the column default is "info")
_SEVERITY_MAP: dict[str, str] = {
    "motion": "info",
    "person": "warn",
    "vehicle": "info",
    "custom": "info",
}


def _created_at(ts_unix: float) -> datetime:
    """Frame time as aware UTC; the wall clock when a detector left ts_unix at 0."""
    if ts_unix > 0:
        return datetime.fromtimestamp(ts_unix, tz=timezone.utc)
    return datetime.now(timezone.utc)


class EventSink:
    """Write detection results to the events table, one row per detection.

    Receives: (camera_name, stream, list[Detection]). Rows carry ``camera_name``,
    ``label`` (= detection kind), ``confidence`` and ``created_at`` (= frame time).
    """

    name: str = "event_sink"

    def __init__(self, db: Any = None) -> None:
        self._db = db if db is not None else _schema

    def __call__(self, camera: str, stream: str, detections: list[Detection]) -> None:
        """Write each detection as an Event row."""
        if not detections:
            return

        for det in detections:
            kind = str(det.kind)
            message = f"{kind} detected on {camera}/{stream} (confidence={det.confidence:.2f})"

            meta: dict[str, Any] = dict(det.metadata) if det.metadata else {}
            if det.bbox is not None:
                meta["bbox"] = list(det.bbox)
            meta["stream"] = stream
            meta["ts_unix"] = det.ts_unix

            try:
                event_id = self._db.insert_event(
                    camera_name=camera,
                    event_type=kind[:32],
                    label=kind[:64],
                    confidence=float(det.confidence),
                    zone="",
                    track_id=None,
                    message=message[:512],
                    created_at=_created_at(det.ts_unix),
                    metadata=meta,
                )
                severity = _SEVERITY_MAP.get(kind, "info")
                if severity != "info":
                    self._db.update_event(event_id, severity=severity)
            except Exception:
                logger.warning(
                    "failed to insert event for %s/%s kind=%s",
                    camera,
                    stream,
                    kind,
                    exc_info=True,
                )


__all__ = ["EventSink"]

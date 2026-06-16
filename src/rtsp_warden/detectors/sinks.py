"""Sinks that consume detector results."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from ..db.engine import get_session
from ..db.models import Camera, Event
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
    """Write detection results to the events table.

    Receives: (camera_name, stream, list[Detection])
    Looks up camera_id from camera_name (cached for performance).
    Inserts one Event row per detection.
    """

    name: str = "event_sink"

    def __init__(self) -> None:
        self._camera_id_cache: dict[str, int | None] = {}
        self._cache_lock: threading.Lock = threading.Lock()

    def __call__(self, camera: str, stream: str, detections: list[Detection]) -> None:
        """Write each detection as an Event row."""
        if not detections:
            return

        camera_id = self._resolve_camera_id(camera)

        for det in detections:
            severity = _SEVERITY_MAP.get(det.kind, "info")
            message = f"{det.kind} detected on {camera}/{stream} (confidence={det.confidence:.2f})"

            meta: dict[str, Any] = dict(det.metadata) if det.metadata else {}
            if det.bbox is not None:
                meta["bbox"] = list(det.bbox)
            meta["stream"] = stream
            meta["ts_unix"] = det.ts_unix

            try:
                self._insert_event(
                    camera_id=camera_id,
                    event_type=det.kind,
                    severity=severity,
                    message=message,
                    metadata=meta,
                )
            except Exception:
                logger.warning(
                    "failed to insert event for %s/%s kind=%s",
                    camera,
                    stream,
                    det.kind,
                    exc_info=True,
                )

    def _resolve_camera_id(self, camera_name: str) -> int | None:
        """Look up camera_id by name, with caching."""
        with self._cache_lock:
            cached = self._camera_id_cache.get(camera_name)
            if cached is not None or camera_name in self._camera_id_cache:
                return cached

        # Cache miss: query DB
        try:
            cam_id = self._query_camera_id(camera_name)
        except Exception:
            logger.debug("failed to resolve camera_id for %s", camera_name, exc_info=True)
            cam_id = None

        with self._cache_lock:
            self._camera_id_cache[camera_name] = cam_id
        return cam_id

    def _query_camera_id(self, camera_name: str) -> int | None:
        """Query the cameras table for the id matching camera_name."""
        with get_session() as session:
            cam = session.query(Camera).filter(Camera.name == camera_name).first()
            return cam.id if cam is not None else None

    @staticmethod
    def _insert_event(
        camera_id: int | None,
        event_type: str,
        severity: str,
        message: str,
        metadata: dict[str, Any],
    ) -> None:
        """Insert a single Event row."""
        with get_session() as session:
            event = Event(
                camera_id=camera_id,
                event_type=event_type,
                severity=severity,
                message=message,
                metadata_json=json.dumps(metadata, default=str),
            )
            session.add(event)
            session.commit()


__all__ = ["EventSink"]

"""Turn tracker updates and motion bursts into ``events`` rows and thumbnail files.

One ``EventBuilder`` per camera. The camera's ``DetectorRunner`` owns it and calls it
from its single worker thread; ``close_all`` also runs from ``DetectorRunner.teardown``
after the worker has been joined. Rebuilding a camera's detectors builds a new runner
and a new builder, so builder state resets together with the tracker (spec 7.1).

Object events (``event_type="object"``), from ``TrackerUpdate``:

* ``opened``: insert the row (label, best confidence, zone, track id, ``created_at`` =
  first seen), write ``<camera>/thumbnails/<event_id>.jpg`` with the best box drawn,
  store that relative path, then call ``on_open``.
* ``improved``: mark the event dirty. A dirty event is rewritten (confidence, zone,
  thumbnail) at most once per ``update_min_interval_s`` of frame time, and once more
  when it closes, so the final best frame is never lost to the throttle.
* ``closed``: set ``ended_at`` (= last seen), then call ``on_close``.

Motion events (``event_type="motion"``, ``label="motion"``): one row per burst (see
``MotionBurst``), confidence 1.0, empty zone, no thumbnail.

Times are frame timestamps (``ts_unix``), never the wall clock. Datetimes handed to the
database helpers and to callbacks are timezone-aware UTC. Every database, file and
callback call is wrapped, so a locked database or a full disk logs a warning and never
stops the worker loop.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ..db import schema as _schema
from .grid_mask import GridMask, zone_for_point

if TYPE_CHECKING:
    from .tracking import Track, TrackerUpdate

log = logging.getLogger(__name__)

OBJECT_EVENT_TYPE = "object"
MOTION_EVENT_TYPE = "motion"
MOTION_LABEL = "motion"

_BOX_COLOR_BGR = (0, 255, 0)
_JPEG_QUALITY = 85


@dataclass(slots=True)
class LiveBox:
    """One box of the latest tracked frame, in tap-frame pixels (x, y, w, h)."""

    label: str
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(slots=True)
class LiveBoxes:
    """Immutable-by-convention snapshot the runner publishes after each tracked frame."""

    boxes: tuple[LiveBox, ...]
    frame_w: int
    frame_h: int
    ts_unix: float


@dataclass(slots=True)
class EventInfo:
    """What callbacks (rule engine, clip scheduler) learn about an event."""

    id: int
    camera: str
    label: str
    confidence: float
    zone: str
    started_at: datetime
    ended_at: datetime | None
    thumbnail_path: str | None
    clip_path: str | None
    track_id: int | None
    event_type: str


@dataclass(slots=True)
class _OpenEvent:
    info: EventInfo
    track: Any  # tracking.Track; kept to read last_seen and best_* on flush and close
    last_write_ts: float
    dirty: bool = False


def _utc(ts_unix: float) -> datetime:
    return datetime.fromtimestamp(float(ts_unix), tz=timezone.utc)


def _bbox(value: Sequence[float]) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in value)
    return (x, y, w, h)


class EventBuilder:
    """Per-camera event lifecycle: rows, thumbnails and open/close callbacks."""

    def __init__(
        self,
        *,
        camera: str,
        output_dir: Path,
        area_masks: Sequence[tuple[str, GridMask]],
        on_open: Callable[[EventInfo], None] | None = None,
        on_close: Callable[[EventInfo], None] | None = None,
        update_min_interval_s: float = 1.0,
        db: Any = _schema,
    ) -> None:
        self.camera = camera
        self.output_dir = Path(output_dir)
        self.area_masks: list[tuple[str, GridMask]] = list(area_masks)
        self.on_open = on_open
        self.on_close = on_close
        self.update_min_interval_s = float(update_min_interval_s)
        self._db = db
        self._lock = threading.Lock()
        self._closed = False
        self._open: dict[int, _OpenEvent] = {}
        self._motion: EventInfo | None = None
        self._frame_w = 0
        self._frame_h = 0

    # -- public API ----------------------------------------------------------

    @staticmethod
    def thumbnail_rel_path(camera: str, event_id: int) -> str:
        """Thumbnail path relative to ``record.output_dir`` (spec 9.2, ruling R20)."""
        return f"{camera}/thumbnails/{event_id}.jpg"

    @property
    def open_event_ids(self) -> list[int]:
        """Ids of the object and motion events that are still open (for status and tests)."""
        with self._lock:
            ids = [state.info.id for state in self._open.values()]
            if self._motion is not None:
                ids.append(self._motion.id)
            return ids

    def on_tracks(self, update: TrackerUpdate, frame_shape: tuple[int, int]) -> None:
        """Apply one tracker update. ``frame_shape`` is ``(height, width)`` of the tap frame."""
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        with self._lock:
            if self._closed:
                return
            self._frame_w, self._frame_h = frame_w, frame_h
            just_opened: set[int] = set()
            for track in update.opened:
                if track.id not in self._open and self._open_track(track):
                    just_opened.add(track.id)
            for track in update.improved:
                state = self._open.get(track.id)
                if state is not None and track.id not in just_opened:
                    state.dirty = True
            closing = {track.id for track in update.closed}
            for state in list(self._open.values()):
                track = state.track
                if not state.dirty or track.id in closing:
                    continue
                if float(track.last_seen) - state.last_write_ts >= self.update_min_interval_s:
                    self._flush(state)
            for track in update.closed:
                state = self._open.pop(track.id, None)
                if state is not None:
                    self._close_track(state)

    def on_motion(self, burst_opened: bool, burst_closed: bool, ts_unix: float) -> None:
        """Open or close the camera's motion event.

        ``ts_unix`` is the burst's first motion frame when opening and its last motion
        frame when closing; the runner calls this once per transition.
        """
        with self._lock:
            if self._closed:
                return
            if burst_closed and self._motion is not None:
                self._close_motion(ts_unix)
            if burst_opened and self._motion is None:
                self._open_motion(ts_unix)

    def close_all(self, ts_unix: float) -> None:
        """Close every open event (runner teardown); later calls do nothing."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for state in list(self._open.values()):
                self._close_track(state)
            self._open.clear()
            if self._motion is not None:
                self._close_motion(ts_unix)

    def write_thumbnail(
        self,
        rel_path: str,
        frame_bgr: np.ndarray,
        bbox: Sequence[float] | None,
    ) -> None:
        """Draw ``bbox`` on a copy of the frame and write it atomically as JPEG."""
        path = self.output_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = frame_bgr.copy()
        if bbox is not None:
            x, y, w, h = _bbox(bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), _BOX_COLOR_BGR, 2)
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for {rel_path}")
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # -- object events -------------------------------------------------------

    def _zone(self, bbox: Sequence[float]) -> str:
        if not self.area_masks or self._frame_w <= 0 or self._frame_h <= 0:
            return ""
        x, y, w, h = _bbox(bbox)
        return zone_for_point(
            self.area_masks, x + w / 2.0, y + h / 2.0, self._frame_w, self._frame_h
        )

    def _open_track(self, track: Track) -> bool:
        zone = self._zone(track.best_bbox)
        started = _utc(track.first_seen)
        confidence = float(track.best_confidence)
        try:
            event_id = int(
                self._db.insert_event(
                    camera_name=self.camera,
                    event_type=OBJECT_EVENT_TYPE,
                    label=track.label,
                    confidence=confidence,
                    zone=zone,
                    track_id=int(track.id),
                    message=f"{track.label} detected on {self.camera} "
                    f"(confidence={confidence:.2f})",
                    created_at=started,
                    metadata={
                        "bbox": list(_bbox(track.best_bbox)),
                        "frame_size": [self._frame_w, self._frame_h],
                    },
                )
            )
        except Exception:
            log.warning(
                "failed to insert event for %s label=%s", self.camera, track.label, exc_info=True
            )
            return False
        thumbnail = self._write_track_thumbnail(event_id, track)
        if thumbnail is not None:
            try:
                self._db.update_event(event_id, thumbnail_path=thumbnail)
            except Exception:
                log.warning("failed to store thumbnail path for event %s", event_id, exc_info=True)
                thumbnail = None
        track.event_id = event_id
        track.zone = zone
        info = EventInfo(
            id=event_id,
            camera=self.camera,
            label=track.label,
            confidence=confidence,
            zone=zone,
            started_at=started,
            ended_at=None,
            thumbnail_path=thumbnail,
            clip_path=None,
            track_id=int(track.id),
            event_type=OBJECT_EVENT_TYPE,
        )
        self._open[track.id] = _OpenEvent(
            info=info, track=track, last_write_ts=float(track.last_seen)
        )
        self._notify(self.on_open, info)
        return True

    def _write_track_thumbnail(self, event_id: int, track: Track) -> str | None:
        if track.best_frame is None:
            return None
        rel_path = self.thumbnail_rel_path(self.camera, event_id)
        try:
            self.write_thumbnail(rel_path, track.best_frame, track.best_bbox)
        except Exception:
            log.warning("failed to write thumbnail %s", rel_path, exc_info=True)
            return None
        return rel_path

    def _refreshed_fields(self, state: _OpenEvent) -> dict[str, Any]:
        track = state.track
        fields: dict[str, Any] = {
            "confidence": float(track.best_confidence),
            "zone": self._zone(track.best_bbox),
        }
        thumbnail = self._write_track_thumbnail(state.info.id, track)
        if thumbnail is not None and state.info.thumbnail_path is None:
            fields["thumbnail_path"] = thumbnail
        return fields

    def _apply(self, state: _OpenEvent, fields: dict[str, Any]) -> None:
        state.info = replace(
            state.info,
            confidence=fields["confidence"],
            zone=fields["zone"],
            thumbnail_path=fields.get("thumbnail_path", state.info.thumbnail_path),
        )
        state.track.zone = fields["zone"]
        state.last_write_ts = float(state.track.last_seen)
        state.dirty = False

    def _flush(self, state: _OpenEvent) -> None:
        fields = self._refreshed_fields(state)
        try:
            self._db.update_event(state.info.id, **fields)
        except Exception:
            log.warning("failed to update event %s", state.info.id, exc_info=True)
        self._apply(state, fields)

    def _close_track(self, state: _OpenEvent) -> None:
        fields: dict[str, Any] = {}
        if state.dirty:
            fields = self._refreshed_fields(state)
            self._apply(state, fields)
        ended = _utc(state.track.last_seen)
        try:
            self._db.close_event(state.info.id, ended, **fields)
        except Exception:
            log.warning("failed to close event %s", state.info.id, exc_info=True)
        self._notify(self.on_close, replace(state.info, ended_at=ended))

    # -- motion events -------------------------------------------------------

    def _open_motion(self, ts_unix: float) -> None:
        started = _utc(ts_unix)
        try:
            event_id = int(
                self._db.insert_event(
                    camera_name=self.camera,
                    event_type=MOTION_EVENT_TYPE,
                    label=MOTION_LABEL,
                    confidence=1.0,
                    zone="",
                    track_id=None,
                    message=f"motion detected on {self.camera}",
                    created_at=started,
                    metadata={},
                )
            )
        except Exception:
            log.warning("failed to insert motion event for %s", self.camera, exc_info=True)
            return
        self._motion = EventInfo(
            id=event_id,
            camera=self.camera,
            label=MOTION_LABEL,
            confidence=1.0,
            zone="",
            started_at=started,
            ended_at=None,
            thumbnail_path=None,
            clip_path=None,
            track_id=None,
            event_type=MOTION_EVENT_TYPE,
        )
        self._notify(self.on_open, self._motion)

    def _close_motion(self, ts_unix: float) -> None:
        info = self._motion
        if info is None:
            return
        self._motion = None
        ended = _utc(ts_unix)
        try:
            self._db.close_event(info.id, ended)
        except Exception:
            log.warning("failed to close motion event %s", info.id, exc_info=True)
        self._notify(self.on_close, replace(info, ended_at=ended))

    # -- callbacks -----------------------------------------------------------

    def _notify(self, callback: Callable[[EventInfo], None] | None, info: EventInfo) -> None:
        if callback is None:
            return
        try:
            callback(info)
        except Exception:
            log.warning(
                "event callback failed for %s event %s", self.camera, info.id, exc_info=True
            )


class MotionBurst:
    """Debounce per-frame motion into bursts: one motion event per burst (ruling R11).

    A burst opens on the ``min_frames``-th consecutive update with motion and closes on
    the first update at least ``grace_seconds`` after the last update with motion. The
    runner calls ``update`` only on frames where a motion detector with events on
    actually ran, so frames skipped for per-detector fps never break a run. Time is the
    frame's ``ts_unix``.
    """

    def __init__(self, *, min_frames: int, grace_seconds: float) -> None:
        if min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        if grace_seconds <= 0:
            raise ValueError("grace_seconds must be > 0")
        self.min_frames = int(min_frames)
        self.grace_seconds = float(grace_seconds)
        self.start_ts: float | None = None  # first motion frame of the open burst
        self.last_motion_ts: float | None = None
        self.closed_end_ts: float | None = None  # last motion of the burst the latest update closed
        self._open = False
        self._run = 0
        self._run_start: float | None = None

    @property
    def is_open(self) -> bool:
        return self._open

    def update(self, has_motion: bool, ts_unix: float) -> tuple[bool, bool]:
        """Feed one frame; return ``(opened, closed)``. Both can be true on one frame."""
        opened = False
        closed = False
        self.closed_end_ts = None
        if (
            self._open
            and self.last_motion_ts is not None
            and ts_unix - self.last_motion_ts >= self.grace_seconds
        ):
            closed = True
            self.closed_end_ts = self.last_motion_ts
            self._open = False
            self.start_ts = None
            self._run = 0
            self._run_start = None
        if not has_motion:
            if not self._open:
                self._run = 0
                self._run_start = None
            return opened, closed
        self.last_motion_ts = ts_unix
        if self._open:
            return opened, closed
        if self._run == 0:
            self._run_start = ts_unix
        self._run += 1
        if self._run >= self.min_frames:
            self._open = True
            opened = True
            self.start_ts = self._run_start
            self._run = 0
            self._run_start = None
        return opened, closed

    def close(self) -> float | None:
        """Force-close (runner teardown). Return the last motion ts of an open burst, else None."""
        self._run = 0
        self._run_start = None
        self.closed_end_ts = None
        if not self._open:
            return None
        self._open = False
        self.start_ts = None
        return self.last_motion_ts


__all__ = [
    "MOTION_EVENT_TYPE",
    "MOTION_LABEL",
    "OBJECT_EVENT_TYPE",
    "EventBuilder",
    "EventInfo",
    "LiveBox",
    "LiveBoxes",
    "MotionBurst",
]

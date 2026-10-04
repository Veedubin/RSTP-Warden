"""DetectorRunner -- FrameConsumer that runs one camera's detectors on tapped frames.

Per frame, on a worker thread:

    decode JPEG -> apply privacy masks -> for each detector that is due at this ts:
        process() -> route by its DetectorSlot:
            tracked (onnx)         -> ROI + ignore zones -> Tracker -> EventBuilder.on_tracks
            motion, events on      -> ROI + ignore zones -> MotionBurst -> EventBuilder.on_motion
            motion, events off     -> dropped (the detector still runs and keeps learning)
            other slots / no slots -> ROI + ignore zones -> result_sinks(camera, stream, dets)

Without ``slots`` (legacy construction) every detector runs on every frame and all
detections go to ``result_sinks``, exactly as before. A tracked slot without a
``tracker``, or a motion slot with events on but no ``motion_burst``, also falls back to
``result_sinks``. A runner that holds a tracker, an event builder or a motion burst must
have at most one worker, because all three depend on frame order.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .base import Detection, Detector
from .daylight import DayNight, allows
from .event_builder import LiveBox, LiveBoxes
from .grid_mask import GridMask
from .roi import ROI, Mask, apply_masks, filter_by_roi

if TYPE_CHECKING:
    from .event_builder import EventBuilder, MotionBurst
    from .registry import DetectorSlot
    from .tracking import Tracker

logger = logging.getLogger(__name__)

_ERROR_TEXT_MAX = 200


@dataclass(slots=True)
class _FrameJob:
    """Internal job enqueued for the worker thread."""

    camera: str
    stream: str
    jpeg_bytes: bytes
    ts_unix: float


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_ERROR_TEXT_MAX]


def _plain(value: Any) -> str | None:
    return None if value is None else str(value)


@dataclass
class DetectorRunner:
    """Decode JPEG -> cv2 frame -> apply masks -> run detectors -> route results.

    Implements the FrameConsumer protocol so it can be wired into a
    FrameTapDispatcher.
    """

    name: str = "detector_runner"
    detectors: Sequence[Detector] = field(default_factory=tuple)
    result_sinks: list[Callable[[str, str, list[Detection]], None]] = field(default_factory=list)
    queue_maxsize: int = 32
    worker_count: int = 2
    masks: list[Mask] = field(default_factory=list)
    roi: ROI | None = None
    grid_masks: list[GridMask] = field(default_factory=list)
    swallow_exceptions: bool = True
    camera: str | None = None  # when set, on_frame ignores frames of other cameras
    stream: str | None = None  # when set, on_frame ignores frames of other streams
    slots: Sequence[DetectorSlot] = field(default_factory=tuple)
    tracker: Tracker | None = None
    event_builder: EventBuilder | None = None
    motion_burst: MotionBurst | None = None
    tap_fps: float = 5.0
    # Day / night (IR) state of this camera, updated from every decoded frame (RW-5).
    daynight: DayNight = field(default_factory=DayNight)

    def __post_init__(self) -> None:
        if self.slots and not self.detectors:
            self.detectors = tuple(slot.detector for slot in self.slots)
        if self.slots and (
            len(self.slots) != len(self.detectors)
            or any(self.slots[i].detector is not self.detectors[i] for i in range(len(self.slots)))
        ):
            raise ValueError("slots must be parallel to detectors (same length and order)")
        stateful = (
            self.tracker is not None
            or self.event_builder is not None
            or self.motion_burst is not None
        )
        if stateful and self.worker_count > 1:
            raise ValueError(
                "a runner with a tracker, event builder or motion burst needs worker_count <= 1"
            )
        self._queue: queue.Queue[_FrameJob] = queue.Queue(maxsize=self.queue_maxsize)
        self._stop_event: threading.Event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._frames_processed: int = 0
        self._frames_dropped: int = 0
        self._detections_total: int = 0
        self._errors_total: int = 0
        count = len(self.detectors)
        self._next_due: list[float | None] = [None] * count
        self._slot_processed: list[int] = [0] * count
        self._slot_skipped: list[int] = [0] * count
        self._slot_errors: list[int] = [0] * count
        self._slot_when_skipped: list[int] = [0] * count  # paused by `when` (RW-5)
        self._setup_errors: list[str | None] = [None] * count
        self._due_tolerance: float = 0.25 / self.tap_fps if self.tap_fps > 0 else 0.0
        self._last_ts: float = 0.0
        self._live: LiveBoxes | None = None

    def setup(self) -> None:
        """Call detector.setup() on each detector and start worker threads.

        A detector whose setup raises is skipped on every frame; its error text is
        kept for ``status()`` so the UI can show why it is not running.
        """
        for i, det in enumerate(self.detectors):
            try:
                det.setup()
            except Exception as exc:
                if not self.swallow_exceptions:
                    raise
                logger.warning("detector %s setup failed", det.name, exc_info=True)
                self._errors_total += 1
                self._setup_errors[i] = _error_text(exc)

        for i in range(self.worker_count):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"detector_worker_{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def teardown(self) -> None:
        """Stop workers, close open events, then call detector.teardown() on each detector."""
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers.clear()

        self._close_open_events()

        for det in self.detectors:
            try:
                det.teardown()
            except Exception:
                if not self.swallow_exceptions:
                    raise
                logger.warning("detector %s teardown failed", det.name, exc_info=True)

    def on_frame(self, camera: str, stream: str, jpeg_bytes: bytes, ts_unix: float) -> None:
        """Enqueue a frame for processing. Returns quickly; work happens in worker threads.

        Frames of another camera or stream (when ``camera`` / ``stream`` are set) are
        ignored. On queue overflow, drops the oldest frame to keep the pipeline moving.
        """
        if self.camera is not None and camera != self.camera:
            return
        if self.stream is not None and stream != self.stream:
            return
        job = _FrameJob(camera=camera, stream=stream, jpeg_bytes=jpeg_bytes, ts_unix=ts_unix)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            # Drop oldest: get one, then put the new one
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                pass
            self._frames_dropped += 1
            logger.debug("detector queue full, dropped oldest frame for %s/%s", camera, stream)

    def live_boxes(self) -> LiveBoxes | None:
        """Boxes of the tracks matched on the latest tracked frame, or None before any."""
        return self._live

    def status(self) -> dict[str, Any]:
        """Return a status dict for /status.json and the UI (plain JSON types only)."""
        return {
            "name": self.name,
            "frames_processed": int(self._frames_processed),
            "frames_dropped": int(self._frames_dropped),
            "detections_total": int(self._detections_total),
            "errors_total": int(self._errors_total),
            "queue_size": int(self._queue.qsize()),
            "worker_count": len(self._workers),
            "detector_count": len(self.detectors),
            "detectors": [self._slot_status(i, slot) for i, slot in enumerate(self.slots)],
            # Stationary suppression (RW-4): tracks held because they never moved.
            "stationary_held": int(getattr(self.event_builder, "held_count", 0) or 0),
            "stationary_suppressed": int(getattr(self.event_builder, "suppressed_total", 0) or 0),
            # Day / night flag (RW-5): None until the first frame was decoded.
            "night": self.daynight.night,
            "night_since": self.daynight.since_ts,
            "night_switches": int(self.daynight.switches),
        }

    def _slot_status(self, i: int, slot: DetectorSlot) -> dict[str, Any]:
        det = slot.detector
        model = slot.spec.model
        descriptor = getattr(det, "descriptor", None)
        if descriptor is not None:
            model = getattr(descriptor, "name", model)
        return {
            "index": int(slot.index),
            "type": str(slot.spec.type),
            "model": _plain(model),
            "device": str(slot.spec.device),
            "provider": _plain(getattr(det, "provider", None)),
            "fallback_warning": _plain(getattr(det, "fallback_warning", None)),
            "fps": float(slot.fps),
            "processed": int(self._slot_processed[i]),
            "skipped": int(self._slot_skipped[i]),
            "when": str(slot.when),
            "when_skipped": int(self._slot_when_skipped[i]),
            "errors": int(self._slot_errors[i]),
            "setup_error": self._setup_errors[i],
        }

    def _worker_loop(self) -> None:
        """Worker thread: pull jobs from queue, decode, detect, route results."""
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_job(job)
            except Exception:
                if not self.swallow_exceptions:
                    raise
                logger.debug("detector job failed for %s/%s", job.camera, job.stream, exc_info=True)
                self._errors_total += 1

    def _due(self, i: int, fps: float, ts_unix: float) -> bool:
        """Per-detector rate limit keyed by position, driven by frame time (spec 6)."""
        if fps <= 0 or fps >= self.tap_fps - 1e-9:
            return True
        period = 1.0 / fps
        next_due = self._next_due[i]
        if next_due is None:
            self._next_due[i] = ts_unix + period
            return True
        if ts_unix + self._due_tolerance < next_due:
            return False
        # On schedule: advance from the schedule (keeps the average rate under jitter).
        # More than a period late (a pause in frames): restart the schedule from now.
        self._next_due[i] = next_due + period if ts_unix - next_due < period else ts_unix + period
        return True

    def _filter(self, detections: list[Detection], frame_w: int, frame_h: int) -> list[Detection]:
        detections = filter_by_roi(detections, self.roi)
        for gm in self.grid_masks:
            detections = gm.filter_detections(detections, frame_w, frame_h)
        return detections

    def _process_job(self, job: _FrameJob) -> None:
        """Decode JPEG, apply masks, run the detectors that are due, route their results."""
        buf = np.frombuffer(job.jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            logger.debug("failed to decode JPEG for %s/%s", job.camera, job.stream)
            return

        # Before the masks: masked pixels are black and would pull the measure to "night".
        night = self.daynight.update(frame, job.ts_unix)
        if self.event_builder is not None:
            self.event_builder.night = night
        frame = apply_masks(frame, self.masks)
        frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        self._last_ts = job.ts_unix

        untracked: list[Detection] = []
        tracked: list[Detection] = []
        motion: list[Detection] = []
        tracked_ran = False
        motion_ran = False
        for i, det in enumerate(self.detectors):
            if self._setup_errors[i] is not None:
                continue
            slot = self.slots[i] if self.slots else None
            if slot is not None and not allows(slot.when, night):
                self._slot_when_skipped[i] += 1
                continue
            if slot is not None and not self._due(i, slot.fps, job.ts_unix):
                self._slot_skipped[i] += 1
                continue
            try:
                results = det.process(frame, job.ts_unix) or []
            except Exception:
                if not self.swallow_exceptions:
                    raise
                logger.warning(
                    "detector %s raised on %s/%s", det.name, job.camera, job.stream, exc_info=True
                )
                self._errors_total += 1
                self._slot_errors[i] += 1
                continue
            self._slot_processed[i] += 1
            if slot is None:
                untracked.extend(results)
            elif slot.tracked and self.tracker is not None:
                tracked_ran = True
                tracked.extend(results)
            elif slot.spec.type == "motion" and not slot.motion_events:
                continue
            elif slot.spec.type == "motion" and self.motion_burst is not None:
                motion_ran = True
                motion.extend(results)
            else:
                untracked.extend(results)

        untracked = self._filter(untracked, frame_w, frame_h)
        tracked = self._filter(tracked, frame_w, frame_h)
        motion = self._filter(motion, frame_w, frame_h)

        self._frames_processed += 1
        self._detections_total += len(untracked) + len(tracked) + len(motion)

        for sink in self.result_sinks:
            try:
                sink(job.camera, job.stream, untracked)
            except Exception:
                if not self.swallow_exceptions:
                    raise
                logger.warning(
                    "result_sink failed for %s/%s", job.camera, job.stream, exc_info=True
                )

        if motion_ran:
            self._motion_stage(bool(motion), job.ts_unix)
        if tracked_ran and self.tracker is not None:
            self._tracked_stage(tracked, frame, frame_w, frame_h, job.ts_unix)

    def _motion_stage(self, has_motion: bool, ts_unix: float) -> None:
        burst = self.motion_burst
        if burst is None:
            return
        opened, closed = burst.update(has_motion, ts_unix)
        builder = self.event_builder
        if builder is None:
            return
        try:
            if closed and burst.closed_end_ts is not None:
                builder.on_motion(False, True, burst.closed_end_ts)
            if opened and burst.start_ts is not None:
                builder.on_motion(True, False, burst.start_ts)
        except Exception:
            if not self.swallow_exceptions:
                raise
            logger.warning("motion events failed for %s", self.name, exc_info=True)
            self._errors_total += 1

    def _tracked_stage(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        frame_w: int,
        frame_h: int,
        ts_unix: float,
    ) -> None:
        tracker = self.tracker
        if tracker is None:
            return
        boxed = [d for d in detections if d.bbox is not None]
        try:
            update = tracker.update(boxed, frame, ts_unix)
            if self.event_builder is not None:
                self.event_builder.on_tracks(update, (frame_h, frame_w))
        except Exception:
            if not self.swallow_exceptions:
                raise
            logger.warning("tracking failed for %s", self.name, exc_info=True)
            self._errors_total += 1
        boxes = tuple(
            LiveBox(
                label=str(label),
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                confidence=float(confidence),
            )
            for label, bbox, confidence in tracker.active_boxes()
        )
        self._live = LiveBoxes(boxes=boxes, frame_w=frame_w, frame_h=frame_h, ts_unix=ts_unix)

    def _close_open_events(self) -> None:
        """Close the open motion burst and every open event (after the worker joined)."""
        builder = self.event_builder
        try:
            if self.motion_burst is not None:
                last_motion = self.motion_burst.close()
                if last_motion is not None and builder is not None:
                    builder.on_motion(False, True, last_motion)
            if builder is not None:
                builder.close_all(self._last_ts)
        except Exception:
            if not self.swallow_exceptions:
                raise
            logger.warning("closing open events failed for %s", self.name, exc_info=True)
        if self.tracker is not None:
            self.tracker.reset()
        self._live = None


__all__ = ["DetectorRunner"]

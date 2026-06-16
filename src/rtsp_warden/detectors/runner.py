"""DetectorRunner -- FrameConsumer that fans out frames to a per-camera worker pool."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .base import Detection, Detector
from .grid_mask import GridMask
from .roi import ROI, Mask, apply_masks, filter_by_roi

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _FrameJob:
    """Internal job enqueued for the worker thread."""

    camera: str
    stream: str
    jpeg_bytes: bytes
    ts_unix: float


@dataclass
class DetectorRunner:
    """Decode JPEG -> cv2 frame -> apply masks -> run detectors -> yield results.

    Results are passed to optional result_sinks (list of callables).

    Implements the FrameConsumer protocol so it can be wired into the
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

    def __post_init__(self) -> None:
        self._queue: queue.Queue[_FrameJob] = queue.Queue(maxsize=self.queue_maxsize)
        self._stop_event: threading.Event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._frames_processed: int = 0
        self._frames_dropped: int = 0
        self._detections_total: int = 0
        self._errors_total: int = 0

    def setup(self) -> None:
        """Call detector.setup() on each detector and start worker threads."""
        for det in self.detectors:
            try:
                det.setup()
            except Exception:
                if not self.swallow_exceptions:
                    raise
                logger.warning("detector %s setup failed", det.name, exc_info=True)
                self._errors_total += 1

        for i in range(self.worker_count):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"detector_worker_{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def teardown(self) -> None:
        """Stop workers and call detector.teardown() on each detector."""
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers.clear()

        for det in self.detectors:
            try:
                det.teardown()
            except Exception:
                if not self.swallow_exceptions:
                    raise
                logger.warning("detector %s teardown failed", det.name, exc_info=True)

    def on_frame(self, camera: str, stream: str, jpeg_bytes: bytes, ts_unix: float) -> None:
        """Enqueue a frame for processing. Returns quickly; work happens in worker threads.

        On queue overflow, drops the oldest frame to keep the pipeline moving.
        """
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

    def status(self) -> dict[str, Any]:
        """Return a status dict for /status.json and dashboard."""
        return {
            "name": self.name,
            "frames_processed": self._frames_processed,
            "frames_dropped": self._frames_dropped,
            "detections_total": self._detections_total,
            "errors_total": self._errors_total,
            "queue_size": self._queue.qsize(),
            "worker_count": len(self._workers),
            "detector_count": len(self.detectors),
        }

    def _worker_loop(self) -> None:
        """Worker thread: pull jobs from queue, decode, detect, push to sinks."""
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

    def _process_job(self, job: _FrameJob) -> None:
        """Decode JPEG, apply masks, run all detectors, push results to sinks."""
        # Decode JPEG bytes to BGR frame
        buf = np.frombuffer(job.jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            logger.debug("failed to decode JPEG for %s/%s", job.camera, job.stream)
            return

        # Apply privacy masks
        frame = apply_masks(frame, self.masks)

        # Run each detector and collect detections
        all_detections: list[Detection] = []
        for det in self.detectors:
            try:
                results = det.process(frame, job.ts_unix)
                if results:
                    all_detections.extend(results)
            except Exception:
                if not self.swallow_exceptions:
                    raise
                logger.warning(
                    "detector %s raised on %s/%s", det.name, job.camera, job.stream, exc_info=True
                )
                self._errors_total += 1

        # Filter detections by ROI (discard those outside)
        all_detections = filter_by_roi(all_detections, self.roi)

        # Filter detections by grid masks (discard those in blocked cells)
        for gm in self.grid_masks:
            all_detections = gm.filter_detections(all_detections)

        self._frames_processed += 1
        self._detections_total += len(all_detections)

        # Push to result sinks
        for sink in self.result_sinks:
            try:
                sink(job.camera, job.stream, all_detections)
            except Exception:
                if not self.swallow_exceptions:
                    raise
                logger.warning(
                    "result_sink failed for %s/%s", job.camera, job.stream, exc_info=True
                )


__all__ = ["DetectorRunner"]

"""Background delivery of rule actions, and delayed clip jobs (spec 8.3, 8.4; ruling R8).

``ActionQueue``: a bounded queue drained by one daemon thread. Each job calls one action
and writes one ``action_runs`` row (status ``ok`` or ``failed``). When the queue is full
the oldest job is dropped and counted in ``dropped``. No retries.

``ClipScheduler``: its own daemon thread. It holds clip jobs until their ``not_before``
time (the end of the clip window plus a margin, so the newest segment already holds the
whole window), builds the clip and stores its path on the event row. It is separate from
the ActionQueue so a slow ffmpeg never delays a notification.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..clips import clip_rel_path
from ..db import schema
from .base import Action, ActionPayload, ActionResult

log = logging.getLogger(__name__)

ACTION_QUEUE_MAXSIZE = 256  # spec 8.3
CLIP_DELAY_MARGIN_S = 2.0  # R8: not_before = ended_at + clips.post_seconds + this margin
_ERROR_MAX_CHARS = 500


@dataclass(slots=True)
class ActionJob:
    """One action to run for one event. ``attachment`` is an absolute file path or None."""

    event_id: int
    action_name: str
    payload: ActionPayload
    attachment: Path | None


class ActionQueue:
    """Bounded, drop-oldest queue of ``ActionJob`` drained by one daemon worker thread."""

    def __init__(
        self,
        actions: Mapping[str, Action],
        *,
        maxsize: int = ACTION_QUEUE_MAXSIZE,
        db: Any = schema,
    ) -> None:
        self._actions: dict[str, Action] = dict(actions)
        self._db = db
        self._queue: queue.Queue[ActionJob | None] = queue.Queue(maxsize=maxsize)
        self._put_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def start(self) -> None:
        """Start the worker thread (no-op when it is already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="action-queue", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the worker after the job in hand; queued jobs are discarded and logged."""
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            try:
                self._queue.put_nowait(None)  # wake the worker now
            except queue.Full:
                pass  # it sees the stop flag within its 1 s poll
            thread.join(timeout=timeout_s)
        discarded = 0
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                discarded += 1
        if discarded:
            log.warning("[actions] stopped with %d queued action(s) not sent", discarded)

    def enqueue(self, job: ActionJob) -> None:
        """Queue *job*. When the queue is full the oldest job is dropped and counted."""
        oldest: ActionJob | None = None
        with self._put_lock:
            try:
                self._queue.put_nowait(job)
                return
            except queue.Full:
                pass
            try:
                oldest = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                pass
            self.dropped += 1
        if oldest is not None:
            log.warning(
                "[actions] queue full; dropped %s for event %d", oldest.action_name, oldest.event_id
            )

    def run_one(self, job: ActionJob) -> ActionResult:
        """Run *job* now on the calling thread and write its action_runs row. Never raises."""
        action = self._actions.get(job.action_name)
        if action is None:
            result = ActionResult(ok=False, error="unknown action")
        else:
            attachment = job.attachment
            if attachment is not None and not os.path.isfile(attachment):
                attachment = None
            try:
                result = action.send(job.payload, attachment)
            except Exception as exc:
                # Actions report failures in ActionResult; this is the safety net. str(exc)
                # can hold the URL, topic or token, so only the type is kept (ruling R17).
                result = ActionResult(ok=False, error=f"{type(exc).__name__} while sending")
        status = "ok" if result.ok else "failed"
        error = result.error[:_ERROR_MAX_CHARS] if result.error else None
        try:
            self._db.insert_action_run(
                event_id=job.event_id,
                action_name=job.action_name,
                status=status,
                error=error,
            )
        except Exception:
            log.warning(
                "[actions] could not record the %s run for event %d",
                job.action_name,
                job.event_id,
                exc_info=True,
            )
        if result.ok:
            log.info("[actions] %s sent for event %d", job.action_name, job.event_id)
        else:
            log.warning(
                "[actions] %s failed for event %d: %s", job.action_name, job.event_id, error
            )
        return result

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:
                break
            self.run_one(job)


@dataclass(slots=True)
class ClipJob:
    """Clip to cut for a closed event, not before ``not_before`` (unix seconds)."""

    event_id: int
    camera: str
    started_at: datetime
    ended_at: datetime
    not_before: float


class ClipScheduler:
    """Runs clip jobs on one daemon thread once their ``not_before`` time has passed."""

    _MAX_WAIT_S = 0.5  # the worker re-reads the clock at least this often

    def __init__(
        self,
        *,
        builder: Callable[[ClipJob], Path | None],
        clock: Callable[[], float] = time.time,
        db: Any = schema,
    ) -> None:
        self._builder = builder
        self._clock = clock
        self._db = db
        self._heap: list[tuple[float, int, ClipJob]] = []
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the worker thread (no-op when it is already running)."""
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(target=self._loop, name="clip-scheduler", daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the worker after the job in hand. Pending jobs are dropped and logged."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=timeout_s)
        with self._cond:
            pending = len(self._heap)
            self._heap.clear()
        if pending:
            log.warning("[clips] stopped with %d clip job(s) not built", pending)

    def schedule(self, job: ClipJob) -> None:
        """Hold *job* until ``job.not_before``. Safe from any thread."""
        with self._cond:
            heapq.heappush(self._heap, (job.not_before, next(self._seq), job))
            self._cond.notify_all()

    def run_due(self, now: float) -> int:
        """Run, on the calling thread, every job with ``not_before <= now``, earliest first.

        Returns the number of jobs run, whether or not each produced a clip.
        """
        ran = 0
        while True:
            with self._cond:
                if not self._heap or self._heap[0][0] > now:
                    return ran
                _, _, job = heapq.heappop(self._heap)
            self._run(job)
            ran += 1

    def _run(self, job: ClipJob) -> None:
        try:
            path = self._builder(job)
        except Exception:
            log.warning("[clips] clip job for event %d failed", job.event_id, exc_info=True)
            return
        if path is None:
            return
        try:
            self._db.update_event(job.event_id, clip_path=clip_rel_path(job.camera, path))
        except Exception:
            log.warning(
                "[clips] could not store the clip path for event %d", job.event_id, exc_info=True
            )

    def _loop(self) -> None:
        while True:
            with self._cond:
                if self._stopping:
                    return
                delay = self._MAX_WAIT_S
                if self._heap:
                    delay = min(delay, self._heap[0][0] - self._clock())
                if delay > 0:
                    self._cond.wait(timeout=delay)
                    continue
                _, _, job = heapq.heappop(self._heap)
            self._run(job)

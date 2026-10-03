"""IoU tracker for tracked (onnx) detections: one Tracker per camera, tracks kept per label.

Spec 7.1 and 7.2. Pure: no I/O, no clock, no threads, no OpenCV. Time comes from the
``ts_unix`` of the frame passed to ``update``.

Contract (the runner, EventBuilder and live preview rely on every line):

- ``update(detections, frame_bgr, ts_unix)`` is called once per frame on which a tracked
  detector ran, in frame order, from one thread (the runner's single worker). Frames where the
  tracked detector was skipped for its ``fps`` are not passed in: a skipped frame is not a miss.
- Detections are grouped by ``Detection.kind`` (the label). Within a label they are matched to
  live tracks greedily, highest IoU first, when IoU >= ``iou_threshold``. Detections without a
  bbox, or with a width or height <= 0, are ignored.
- A new track is tentative until it has been matched on ``min_frames`` consecutive updates (the
  update that creates it counts as one). The update on which it gets there lists it in
  ``opened``. A tentative track that misses one update is discarded silently: it is never listed
  in ``opened`` or ``closed``.
- An opened track that misses stays alive while ``ts_unix - last_seen < grace_seconds``. The
  first update with ``ts_unix - last_seen >= grace_seconds`` lists it in ``closed`` (with
  ``open = False``) BEFORE matching, so a detection after the grace starts a new track.
- Every track listed in ``opened`` is later listed in ``closed`` exactly once, by ``update`` or by
  ``close_all``; ``reset`` discards tracks without listing them.
- ``improved`` lists tracks opened on an earlier update whose ``best_confidence`` rose (strictly)
  on this update. A track is never in ``opened`` and ``improved`` on the same update.
- ``best_frame`` is a copy of ``frame_bgr`` taken when ``best_confidence`` rose; each track owns
  its own copy, so callers may draw on it after copying it again (never draw on it in place).
- ``ts_unix`` lower than the previous update's is clamped to it (wall clock stepped back), so
  ``first_seen <= last_seen`` always holds.
- Bboxes are xywh ints in pixels of the frame given to ``update``; confidences are plain floats.
- ``first_bbox`` is the box the track was created with and never changes; ``bbox`` follows the
  object.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .base import Detection

Box = tuple[int, int, int, int]


@dataclass(slots=True, eq=False)
class Track:
    """One object followed across frames. Equality is identity (``eq=False``).

    ``frames`` counts the updates the track was matched on, including the one that created it.
    ``event_id`` and ``zone`` belong to the EventBuilder; the tracker never reads or writes them.
    """

    id: int
    label: str
    first_seen: float
    last_seen: float
    frames: int
    bbox: Box
    best_confidence: float
    best_bbox: Box
    best_frame: np.ndarray | None = field(repr=False)
    best_ts: float
    open: bool = True
    event_id: int | None = None
    zone: str = ""
    # Where the track started; never updated. The EventBuilder compares ``bbox`` against it to
    # tell an object that moved from one that only flickers in place (stationary suppression).
    first_bbox: Box | None = None


@dataclass
class TrackerUpdate:
    """What one ``Tracker.update`` call changed. Every list is ordered by track id.

    - ``opened``: tracks that reached ``min_frames`` on this update.
    - ``improved``: tracks opened earlier whose ``best_confidence`` rose on this update.
    - ``closed``: opened tracks whose grace ran out (``open`` is now False).
    - ``active``: every track still alive after this update, tentative ones and opened ones
      inside their grace period included.
    """

    opened: list[Track] = field(default_factory=list)
    improved: list[Track] = field(default_factory=list)
    closed: list[Track] = field(default_factory=list)
    active: list[Track] = field(default_factory=list)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two xywh boxes; 0.0 when either box has no area."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    inter_w = min(ax + aw, bx + bw) - max(ax, bx)
    inter_h = min(ay + ah, by + bh) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = float(inter_w) * float(inter_h)
    union = float(aw) * float(ah) + float(bw) * float(bh) - inter
    return inter / union


def _as_box(bbox: Sequence[int] | None) -> Box | None:
    """Normalise a Detection.bbox to a tuple of plain ints, or None when unusable."""
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = (int(v) for v in bbox)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _copy_frame(frame_bgr: np.ndarray | None) -> np.ndarray | None:
    return None if frame_bgr is None else frame_bgr.copy()


class Tracker:
    """Greedy IoU tracker, one per camera. Not thread-safe except ``active_boxes``."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        grace_seconds: float = 3.0,
        min_frames: int = 2,
        id_start: int = 1,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if not grace_seconds > 0.0:
            raise ValueError(f"grace_seconds must be > 0, got {grace_seconds}")
        if min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {min_frames}")
        self.iou_threshold = float(iou_threshold)
        self.grace_seconds = float(grace_seconds)
        self.min_frames = int(min_frames)
        self._next_id = int(id_start)
        self._tracks: list[Track] = []
        # (label, bbox, confidence) of the tracks seen on the latest update; rebound, never mutated.
        self._latest: tuple[tuple[str, Box, float], ...] = ()
        self._last_ts: float | None = None

    def update(
        self,
        detections: Sequence[Detection],
        frame_bgr: np.ndarray | None,
        ts_unix: float,
    ) -> TrackerUpdate:
        """Feed one tracked frame; see the module docstring for the full contract."""
        ts = float(ts_unix)
        if self._last_ts is not None and ts < self._last_ts:
            ts = self._last_ts
        self._last_ts = ts
        result = TrackerUpdate()

        # 1. Grace expiry runs before matching, so a late detection starts a new track.
        alive: list[Track] = []
        for track in self._tracks:
            if ts - track.last_seen >= self.grace_seconds:
                self._end(track, result.closed)
            else:
                alive.append(track)

        # 2. Usable detections grouped by label, in input order.
        by_label: dict[str, list[tuple[Box, float]]] = {}
        for det in detections:
            box = _as_box(det.bbox)
            if box is None:
                continue
            by_label.setdefault(str(det.kind), []).append((box, float(det.confidence)))

        # 3. Greedy matching per label: every pair at or above the threshold, highest IoU first
        #    (ties: lower track id, then earlier detection).
        matched: list[tuple[Track, Box, float]] = []
        created: list[tuple[Track, float]] = []
        for label, dets in by_label.items():
            candidates = [t for t in alive if t.label == label]
            pairs: list[tuple[float, int, int, int]] = []
            for ti, track in enumerate(candidates):
                for di, (box, _conf) in enumerate(dets):
                    score = iou(track.bbox, box)
                    if score >= self.iou_threshold:
                        pairs.append((score, track.id, di, ti))
            pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            for _score, _track_id, di, ti in pairs:
                if ti in used_tracks or di in used_dets:
                    continue
                used_tracks.add(ti)
                used_dets.add(di)
                matched.append((candidates[ti], dets[di][0], dets[di][1]))
            for di, (box, conf) in enumerate(dets):
                if di in used_dets:
                    continue
                track = Track(
                    id=self._next_id,
                    label=label,
                    first_seen=ts,
                    last_seen=ts,
                    frames=1,
                    bbox=box,
                    best_confidence=conf,
                    best_bbox=box,
                    best_frame=_copy_frame(frame_bgr),
                    best_ts=ts,
                    first_bbox=box,
                )
                self._next_id += 1
                created.append((track, conf))
                if self.min_frames <= 1:
                    result.opened.append(track)

        # 4. Apply the matches.
        matched_ids: set[int] = set()
        for track, box, conf in matched:
            matched_ids.add(track.id)
            was_open = track.frames >= self.min_frames
            track.frames += 1
            track.last_seen = ts
            track.bbox = box
            if conf > track.best_confidence:
                track.best_confidence = conf
                track.best_bbox = box
                track.best_frame = _copy_frame(frame_bgr)
                track.best_ts = ts
                if was_open:
                    result.improved.append(track)
            if not was_open and track.frames >= self.min_frames:
                result.opened.append(track)

        # 5. Unmatched tracks: tentative ones are discarded, opened ones wait out their grace.
        kept: list[Track] = []
        for track in alive:
            if track.id in matched_ids or track.frames >= self.min_frames:
                kept.append(track)
            else:
                track.open = False
        kept.extend(track for track, _conf in created)
        kept.sort(key=lambda t: t.id)
        self._tracks = kept

        seen = [(track, conf) for track, _box, conf in matched] + created
        seen.sort(key=lambda item: item[0].id)
        self._latest = tuple((track.label, track.bbox, conf) for track, conf in seen)

        result.opened.sort(key=lambda t: t.id)
        result.improved.sort(key=lambda t: t.id)
        result.closed.sort(key=lambda t: t.id)
        result.active = list(kept)
        return result

    def _end(self, track: Track, closed: list[Track]) -> None:
        """Mark ``track`` ended; list it in ``closed`` only if it had been opened."""
        track.open = False
        if track.frames >= self.min_frames:
            closed.append(track)

    def close_all(self, ts_unix: float) -> list[Track]:
        """End every live track (runner teardown). Returns the opened ones, ordered by id.

        Each returned track keeps ``last_seen`` as its end time; ``ts_unix`` (the caller's
        clock at teardown) only advances the tracker's clock so later updates are clamped to it.
        Tentative tracks are discarded without being returned.
        """
        ts = float(ts_unix)
        if self._last_ts is None or ts > self._last_ts:
            self._last_ts = ts
        closed: list[Track] = []
        for track in self._tracks:
            self._end(track, closed)
        self._tracks = []
        self._latest = ()
        return closed

    def active_boxes(self) -> list[tuple[str, Box, float]]:
        """``(label, bbox, confidence)`` of the tracks matched or created on the latest update.

        Tracks inside their grace period are left out (the object is no longer where the box
        was). ``confidence`` is that update's detection confidence, not the best one. Safe to
        call from another thread: it reads one tuple that ``update`` rebinds.
        """
        return list(self._latest)

    def reset(self) -> None:
        """Drop every track without listing it anywhere; ids keep counting up.

        Call ``close_all`` first when the open events must be closed.
        """
        for track in self._tracks:
            track.open = False
        self._tracks = []
        self._latest = ()
        self._last_ts = None

"""Tests for DetectorRunner slots: per-detector fps, routing, tracking, motion bursts, status."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from rtsp_warden.config import CameraConfig
from rtsp_warden.detectors.base import Detection
from rtsp_warden.detectors.event_builder import EventBuilder, EventInfo, LiveBox, MotionBurst
from rtsp_warden.detectors.grid_mask import GridMask
from rtsp_warden.detectors.registry import (
    DetectorSlot,
    DetectorSpec,
    build_detectors_for_camera,
)
from rtsp_warden.detectors.runner import DetectorRunner, _FrameJob
from rtsp_warden.detectors.tracking import Tracker, TrackerUpdate

W, H = 320, 180
PERSON_BOX = (100, 50, 40, 60)


def _make_jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((H, W, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


JPEG = _make_jpeg()


def _job(ts: float) -> _FrameJob:
    return _FrameJob(camera="yard", stream="main", jpeg_bytes=JPEG, ts_unix=ts)


def _ticks(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step)) + 1
    return [round(start + step * i, 6) for i in range(count)]


def _utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


Script = Callable[[float], list[Detection]]


class ScriptedDetector:
    """Returns script(ts) on every call and records the timestamps it ran at."""

    name = "scripted"

    def __init__(
        self,
        script: Script,
        *,
        kind: str = "onnx",
        provider: str | None = None,
        fallback_warning: str | None = None,
    ) -> None:
        self.script = script
        self.kind = kind
        self.provider = provider
        self.fallback_warning = fallback_warning
        self.calls: list[float] = []

    def setup(self) -> None:
        pass

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        self.calls.append(ts_unix)
        return self.script(ts_unix)

    def teardown(self) -> None:
        pass


class MissingModelDetector:
    """An onnx detector whose model is absent and cannot be downloaded (offline)."""

    name = "onnx"
    kind = "onnx"

    def __init__(self) -> None:
        self.calls = 0

    def setup(self) -> None:
        raise FileNotFoundError("model file missing: yolox_s.onnx (download failed: offline)")

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        self.calls += 1
        return []

    def teardown(self) -> None:
        pass


class RecordingTracker:
    """Stands in for Tracker; records (ts, number of detections) per update."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, int]] = []

    def update(
        self, detections: list[Detection], frame_bgr: np.ndarray, ts_unix: float
    ) -> TrackerUpdate:
        self.updates.append((ts_unix, len(detections)))
        return TrackerUpdate(opened=[], improved=[], closed=[], active=[])

    def active_boxes(self) -> list[tuple[str, tuple[int, int, int, int], float]]:
        return []

    def reset(self) -> None:
        pass


class FakeDb:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, Any]]] = []
        self._next_id = 0

    def insert_event(self, **fields: Any) -> int:
        self._next_id += 1
        self.calls.append(("insert", self._next_id, fields))
        return self._next_id

    def update_event(self, event_id: int, **fields: Any) -> None:
        self.calls.append(("update", event_id, fields))

    def close_event(self, event_id: int, ended_at: datetime, **fields: Any) -> None:
        self.calls.append(("close", event_id, {"ended_at": ended_at, **fields}))

    def of(self, kind: str) -> list[tuple[str, int, dict[str, Any]]]:
        return [call for call in self.calls if call[0] == kind]


def _person(
    bbox: tuple[int, int, int, int] = PERSON_BOX, conf: float = 0.9, until: float = 1e18
) -> Script:
    def script(ts: float) -> list[Detection]:
        if ts > until:
            return []
        return [Detection(kind="person", confidence=conf, bbox=bbox, ts_unix=ts)]

    return script


def _motion(until: float = 1e18) -> Script:
    def script(ts: float) -> list[Detection]:
        if ts > until:
            return []
        return [Detection(kind="motion", confidence=0.7, bbox=(10, 10, 50, 50), ts_unix=ts)]

    return script


def _slot(
    det: Any,
    *,
    index: int = 0,
    type_: str = "onnx",
    fps: float = 5.0,
    tracked: bool | None = None,
    motion_events: bool = False,
) -> DetectorSlot:
    return DetectorSlot(
        index=index,
        spec=DetectorSpec(type=type_, fps=fps),
        detector=det,
        fps=fps,
        tracked=(type_ == "onnx") if tracked is None else tracked,
        motion_events=motion_events,
        input_width=None,
    )


def _builder(tmp_path: Path, db: FakeDb, **kwargs: Any) -> EventBuilder:
    return EventBuilder(camera="yard", output_dir=tmp_path, area_masks=[], db=db, **kwargs)


def _run(runner: DetectorRunner, timestamps: list[float]) -> None:
    for ts in timestamps:
        runner._process_job(_job(ts))


# ---------------------------------------------------------------------------
# Per-detector fps
# ---------------------------------------------------------------------------


def test_fps_skipping_feeds_the_tracker_only_frames_the_detector_ran_on() -> None:
    det = ScriptedDetector(_person())
    tracker = RecordingTracker()
    runner = DetectorRunner(
        slots=[_slot(det, fps=2.0)], tracker=tracker, worker_count=0, tap_fps=5.0
    )

    _run(runner, _ticks(0.0, 2.0, 0.2))

    assert det.calls == [0.0, 0.6, 1.0, 1.6, 2.0]
    assert tracker.updates == [(0.0, 1), (0.6, 1), (1.0, 1), (1.6, 1), (2.0, 1)]
    row = runner.status()["detectors"][0]
    assert (row["processed"], row["skipped"]) == (5, 6)


def test_fps_schedule_restarts_after_a_pause_without_a_catch_up_burst() -> None:
    det = ScriptedDetector(lambda ts: [])
    runner = DetectorRunner(slots=[_slot(det, fps=2.0)], worker_count=0, tap_fps=5.0)

    _run(runner, [0.0, 10.0, 10.2, 10.4, 10.6])

    assert det.calls == [0.0, 10.0, 10.6]


def test_interval_seconds_one_runs_the_detector_at_one_fps(tmp_path: Path) -> None:
    """(review focus) A legacy interval_seconds: 1.0 loads and runs at 1 fps on a 5 fps tap."""
    spec = DetectorSpec.model_validate({"type": "motion", "interval_seconds": 1.0})
    cam = CameraConfig(name="yard", main_url="rtsp://u:p@h/m", detect_fps=5.0, detectors=[spec])
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=tmp_path)
    assert bundle.slots[0].fps == 1.0
    runner = DetectorRunner(slots=bundle.slots, worker_count=0, tap_fps=cam.detect_fps)
    runner.setup()
    try:
        _run(runner, _ticks(100.0, 101.8, 0.2))
    finally:
        runner.teardown()

    row = runner.status()["detectors"][0]
    assert (row["fps"], row["processed"], row["skipped"]) == (1.0, 2, 8)


# ---------------------------------------------------------------------------
# Tracked path: one event per object visit
# ---------------------------------------------------------------------------


def test_one_event_row_per_visit_through_runner_tracker_and_builder(tmp_path: Path) -> None:
    db = FakeDb()
    opened: list[EventInfo] = []
    closed: list[EventInfo] = []
    builder = _builder(tmp_path, db, on_open=opened.append, on_close=closed.append)
    det = ScriptedDetector(_person(until=1.0))
    runner = DetectorRunner(
        name="detector_yard",
        camera="yard",
        stream="main",
        slots=[_slot(det)],
        tracker=Tracker(grace_seconds=3.0, min_frames=2),
        event_builder=builder,
        worker_count=0,
    )

    _run(runner, _ticks(0.0, 6.0, 0.2))

    inserts = db.of("insert")
    assert len(inserts) == 1
    event_id = inserts[0][1]
    assert inserts[0][2]["label"] == "person"
    assert inserts[0][2]["created_at"] == _utc(0.0)
    assert db.of("close") == [("close", event_id, {"ended_at": _utc(1.0)})]
    thumbnail = tmp_path / "yard" / "thumbnails" / f"{event_id}.jpg"
    image = cv2.imread(str(thumbnail))
    assert image is not None
    assert image.shape == (H, W, 3)
    assert [info.id for info in opened] == [event_id]
    assert [info.ended_at for info in closed] == [_utc(1.0)]


def test_two_overlapping_people_become_two_events(tmp_path: Path) -> None:
    """(review focus) Overlapping boxes (IoU 0.33) are two tracks, so two events."""

    def two_people(ts: float) -> list[Detection]:
        return [
            Detection(kind="person", confidence=0.9, bbox=(100, 40, 40, 100), ts_unix=ts),
            Detection(kind="person", confidence=0.8, bbox=(120, 40, 40, 100), ts_unix=ts),
        ]

    db = FakeDb()
    runner = DetectorRunner(
        slots=[_slot(ScriptedDetector(two_people))],
        tracker=Tracker(grace_seconds=3.0, min_frames=2),
        event_builder=_builder(tmp_path, db),
        worker_count=0,
    )

    _run(runner, _ticks(0.0, 2.0, 0.2))

    inserts = db.of("insert")
    assert len(inserts) == 2
    assert len({call[2]["track_id"] for call in inserts}) == 2


def test_person_standing_still_for_ten_minutes_is_one_event(tmp_path: Path) -> None:
    """(review focus) A stationary person seen every second for 600 s is one event."""
    db = FakeDb()
    runner = DetectorRunner(
        slots=[_slot(ScriptedDetector(_person()), fps=1.0)],
        tracker=Tracker(grace_seconds=3.0, min_frames=2),
        event_builder=_builder(tmp_path, db),
        worker_count=0,
        tap_fps=1.0,
    )

    _run(runner, _ticks(0.0, 600.0, 1.0))

    assert len(db.of("insert")) == 1
    assert db.of("close") == []
    runner.teardown()
    assert len(db.of("close")) == 1
    assert db.of("close")[0][2]["ended_at"] == _utc(600.0)


def test_live_boxes_reflect_the_latest_tracked_frame() -> None:
    det = ScriptedDetector(_person(until=0.4))
    runner = DetectorRunner(
        slots=[_slot(det)], tracker=Tracker(grace_seconds=3.0, min_frames=2), worker_count=0
    )
    assert runner.live_boxes() is None

    _run(runner, [0.0, 0.2])
    live = runner.live_boxes()
    assert live is not None
    assert live.boxes == (LiveBox(label="person", bbox=PERSON_BOX, confidence=0.9),)
    assert (live.frame_w, live.frame_h, live.ts_unix) == (W, H, 0.2)

    _run(runner, [0.4, 0.6])  # 0.6: the tracked detector ran and saw nobody
    live = runner.live_boxes()
    assert live is not None
    assert live.boxes == ()
    assert live.ts_unix == 0.6


def test_teardown_closes_open_events_and_is_idempotent(tmp_path: Path) -> None:
    db = FakeDb()
    closed: list[EventInfo] = []
    runner = DetectorRunner(
        slots=[_slot(ScriptedDetector(_person()))],
        tracker=Tracker(grace_seconds=3.0, min_frames=2),
        event_builder=_builder(tmp_path, db, on_close=closed.append),
        worker_count=1,
    )
    runner.setup()
    _run(runner, [0.0, 0.2, 0.4])
    assert len(db.of("insert")) == 1
    assert db.of("close") == []

    runner.teardown()

    assert [call[2]["ended_at"] for call in db.of("close")] == [_utc(0.4)]
    assert [info.ended_at for info in closed] == [_utc(0.4)]
    assert runner.live_boxes() is None
    runner.teardown()
    assert len(db.of("close")) == 1


# ---------------------------------------------------------------------------
# Motion path
# ---------------------------------------------------------------------------


def test_motion_rows_are_suppressed_when_motion_events_is_false(tmp_path: Path) -> None:
    db = FakeDb()
    seen: list[list[Detection]] = []
    det = ScriptedDetector(_motion(), kind="motion")
    runner = DetectorRunner(
        slots=[_slot(det, type_="motion", motion_events=False)],
        result_sinks=[lambda cam, stream, dets: seen.append(list(dets))],
        motion_burst=MotionBurst(min_frames=2, grace_seconds=3.0),
        event_builder=_builder(tmp_path, db),
        worker_count=0,
    )

    _run(runner, _ticks(0.0, 2.0, 0.2))

    assert db.calls == []
    assert seen == [[]] * 11
    assert runner.status()["detectors"][0]["processed"] == 11  # it still runs and learns


def test_motion_burst_writes_one_row_per_burst(tmp_path: Path) -> None:
    db = FakeDb()
    seen: list[list[Detection]] = []
    det = ScriptedDetector(_motion(until=1.0), kind="motion")
    runner = DetectorRunner(
        slots=[_slot(det, type_="motion", motion_events=True)],
        result_sinks=[lambda cam, stream, dets: seen.append(list(dets))],
        motion_burst=MotionBurst(min_frames=2, grace_seconds=3.0),
        event_builder=_builder(tmp_path, db),
        worker_count=0,
    )

    _run(runner, _ticks(0.0, 6.0, 0.2))

    inserts = db.of("insert")
    assert len(inserts) == 1
    assert inserts[0][2]["event_type"] == "motion"
    assert inserts[0][2]["created_at"] == _utc(0.0)
    assert db.of("close") == [("close", inserts[0][1], {"ended_at": _utc(1.0)})]
    assert all(dets == [] for dets in seen)  # motion never reaches result_sinks


def test_open_motion_burst_is_closed_on_teardown(tmp_path: Path) -> None:
    db = FakeDb()
    runner = DetectorRunner(
        slots=[
            _slot(ScriptedDetector(_motion(), kind="motion"), type_="motion", motion_events=True)
        ],
        motion_burst=MotionBurst(min_frames=2, grace_seconds=3.0),
        event_builder=_builder(tmp_path, db),
        worker_count=0,
    )
    _run(runner, [0.0, 0.2, 0.4])

    runner.teardown()

    assert [call[2]["ended_at"] for call in db.of("close")] == [_utc(0.4)]


def test_motion_with_events_but_no_burst_falls_back_to_result_sinks() -> None:
    seen: list[list[Detection]] = []
    det = ScriptedDetector(_motion(), kind="motion")
    runner = DetectorRunner(
        slots=[_slot(det, type_="motion", motion_events=True)],
        result_sinks=[lambda cam, stream, dets: seen.append(list(dets))],
        worker_count=0,
    )

    runner._process_job(_job(0.0))

    assert [d.kind for d in seen[0]] == ["motion"]


# ---------------------------------------------------------------------------
# Routing, filters and construction
# ---------------------------------------------------------------------------


def test_legacy_slot_goes_to_result_sinks_not_the_tracker() -> None:
    seen: list[list[Detection]] = []
    tracker = RecordingTracker()
    runner = DetectorRunner(
        slots=[_slot(ScriptedDetector(_person(), kind="person"), type_="person", tracked=False)],
        tracker=tracker,
        result_sinks=[lambda cam, stream, dets: seen.append(list(dets))],
        worker_count=0,
    )

    runner._process_job(_job(0.0))

    assert [d.bbox for d in seen[0]] == [PERSON_BOX]
    assert tracker.updates == []


def test_ignore_zones_filter_in_decoded_frame_pixels() -> None:
    def two_boxes(ts: float) -> list[Detection]:
        return [
            Detection(kind="person", confidence=0.9, bbox=(40, 20, 40, 40), ts_unix=ts),
            Detection(kind="person", confidence=0.9, bbox=(220, 110, 40, 40), ts_unix=ts),
        ]

    seen: list[list[Detection]] = []
    runner = DetectorRunner(
        slots=[_slot(ScriptedDetector(two_boxes), type_="person", tracked=False)],
        grid_masks=[GridMask(grid_cols=2, grid_rows=2, blocked_cells={(0, 0)})],
        result_sinks=[lambda cam, stream, dets: seen.append(list(dets))],
        worker_count=0,
    )

    runner._process_job(_job(0.0))

    assert [d.bbox for d in seen[0]] == [(220, 110, 40, 40)]


def test_on_frame_ignores_other_cameras_and_streams() -> None:
    runner = DetectorRunner(camera="yard", stream="main", worker_count=0, queue_maxsize=4)
    runner.on_frame("other", "main", JPEG, 1.0)
    runner.on_frame("yard", "sub", JPEG, 1.0)
    assert runner._queue.qsize() == 0
    runner.on_frame("yard", "main", JPEG, 1.0)
    assert runner._queue.qsize() == 1


def test_stateful_runner_rejects_more_than_one_worker() -> None:
    with pytest.raises(ValueError, match="worker_count"):
        DetectorRunner(tracker=RecordingTracker(), worker_count=2)


def test_slots_must_be_parallel_to_detectors() -> None:
    first = ScriptedDetector(lambda ts: [])
    second = ScriptedDetector(lambda ts: [])
    with pytest.raises(ValueError, match="parallel"):
        DetectorRunner(detectors=(first,), slots=[_slot(second)])
    runner = DetectorRunner(slots=[_slot(first), _slot(second, index=1)], worker_count=0)
    assert list(runner.detectors) == [first, second]


# ---------------------------------------------------------------------------
# Status and setup failures
# ---------------------------------------------------------------------------


def test_status_detector_rows_are_plain_json() -> None:
    det = ScriptedDetector(
        lambda ts: [],
        provider="CPUExecutionProvider",
        fallback_warning="device cuda requested but unavailable; using CPU",
    )
    spec = DetectorSpec(type="onnx", model="yolox-nano", device="cuda", fps=2.0)
    slot = DetectorSlot(
        index=3,
        spec=spec,
        detector=det,
        fps=2.0,
        tracked=True,
        motion_events=False,
        input_width=416,
    )
    runner = DetectorRunner(slots=[slot], tracker=RecordingTracker(), worker_count=0)

    _run(runner, _ticks(0.0, 1.0, 0.2))

    status = runner.status()
    json.dumps(status)
    assert status["detectors"] == [
        {
            "index": 3,
            "type": "onnx",
            "model": "yolox-nano",
            "device": "cuda",
            "provider": "CPUExecutionProvider",
            "fallback_warning": "device cuda requested but unavailable; using CPU",
            "fps": 2.0,
            "processed": 3,
            "skipped": 3,
            "errors": 0,
            "setup_error": None,
        }
    ]
    for key in ("frames_processed", "frames_dropped", "detections_total", "errors_total"):
        assert type(status[key]) is int


def test_detector_whose_setup_fails_is_skipped_and_reported() -> None:
    """(review focus) An offline camera without model weights keeps its other detectors."""
    broken = MissingModelDetector()
    motion = ScriptedDetector(_motion(), kind="motion")
    seen: list[list[Detection]] = []
    runner = DetectorRunner(
        slots=[
            _slot(broken, index=0),
            _slot(motion, index=1, type_="motion", motion_events=True),
        ],
        result_sinks=[lambda cam, stream, dets: seen.append(list(dets))],
        tracker=Tracker(grace_seconds=3.0, min_frames=2),
        worker_count=0,
    )

    runner.setup()  # must not raise
    _run(runner, _ticks(0.0, 1.0, 0.2))
    runner.teardown()

    assert broken.calls == 0
    status = runner.status()
    assert status["errors_total"] == 1
    assert status["detectors"][0]["setup_error"] == (
        "FileNotFoundError: model file missing: yolox_s.onnx (download failed: offline)"
    )
    assert status["detectors"][0]["processed"] == 0
    assert status["detectors"][1]["processed"] == 6
    assert [len(dets) for dets in seen] == [1] * 6


# ---------------------------------------------------------------------------
# Registry: bundle.slots
# ---------------------------------------------------------------------------


def test_bundle_slots_are_parallel_and_keep_spec_indexes(tmp_path: Path) -> None:
    cam = CameraConfig(
        name="yard",
        main_url="rtsp://u:p@h/m",
        detect_fps=5.0,
        detectors=[
            DetectorSpec(type="custom"),  # no import_path: the build fails and is skipped
            DetectorSpec(type="motion", enabled=False),
            DetectorSpec(type="motion", fps=2.0),
            DetectorSpec(type="onnx", fps=1.0),
        ],
    )

    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=tmp_path)

    assert [slot.index for slot in bundle.slots] == [2, 3]
    assert all(
        slot.detector is det for slot, det in zip(bundle.slots, bundle.detectors, strict=True)
    )
    motion, onnx = bundle.slots
    assert (motion.spec.type, motion.fps, motion.tracked, motion.motion_events) == (
        "motion",
        2.0,
        False,
        False,  # an enabled onnx spec turns motion events off unless events: true
    )
    assert motion.input_width is None
    assert (onnx.spec.type, onnx.fps, onnx.tracked, onnx.motion_events) == (
        "onnx",
        1.0,
        True,
        False,
    )
    assert onnx.input_width == 640


def test_motion_events_flag_resolves_per_camera(tmp_path: Path) -> None:
    alone = CameraConfig(
        name="a", main_url="rtsp://u:p@h/m", detectors=[DetectorSpec(type="motion")]
    )
    forced = CameraConfig(
        name="b",
        main_url="rtsp://u:p@h/m",
        detectors=[DetectorSpec(type="motion", events=True), DetectorSpec(type="onnx")],
    )

    alone_slots = build_detectors_for_camera(alone, alone.detectors, models_dir=tmp_path).slots
    forced_slots = build_detectors_for_camera(forced, forced.detectors, models_dir=tmp_path).slots

    assert alone_slots[0].motion_events is True
    assert forced_slots[0].motion_events is True
    assert build_detectors_for_camera(alone, [], models_dir=tmp_path).slots == []


# ---------------------------------------------------------------------------
# Stationary suppression through the runner (RW-4)
# ---------------------------------------------------------------------------


def _walk_in_then_stand(ts: float) -> list[Detection]:
    """A person walks in from the left for 2 s, then stands at PERSON_BOX."""
    if ts < 2.0:
        x = 20 + int(40 * ts)
        return [Detection(kind="person", confidence=0.9, bbox=(x, 50, 40, 60))]
    return [Detection(kind="person", confidence=0.9, bbox=PERSON_BOX)]


def test_person_who_never_moves_makes_no_event_with_stationary_suppression(
    tmp_path: Path,
) -> None:
    db = FakeDb()
    runner = DetectorRunner(
        slots=[_slot(ScriptedDetector(_person()), fps=1.0)],
        tracker=Tracker(grace_seconds=3.0, min_frames=2),
        event_builder=_builder(tmp_path, db, stationary_iou=0.6),
        worker_count=0,
        tap_fps=1.0,
    )

    _run(runner, _ticks(0.0, 60.0, 1.0))

    assert db.of("insert") == []
    assert runner.status()["stationary_held"] == 1
    assert runner.status()["stationary_suppressed"] == 0
    runner.teardown()
    assert db.calls == []
    assert runner.status()["stationary_held"] == 0
    assert runner.status()["stationary_suppressed"] == 1


def test_person_who_walks_in_and_then_stands_still_is_one_event(tmp_path: Path) -> None:
    db = FakeDb()
    runner = DetectorRunner(
        slots=[_slot(ScriptedDetector(_walk_in_then_stand), fps=1.0)],
        tracker=Tracker(grace_seconds=3.0, min_frames=2),
        event_builder=_builder(tmp_path, db, stationary_iou=0.6),
        worker_count=0,
        tap_fps=1.0,
    )

    _run(runner, _ticks(0.0, 600.0, 1.0))

    inserts = db.of("insert")
    assert len(inserts) == 1
    assert inserts[0][2]["created_at"] == _utc(1.0)  # the frame on which it had moved
    assert runner.status()["stationary_held"] == 0
    runner.teardown()
    assert len(db.of("close")) == 1

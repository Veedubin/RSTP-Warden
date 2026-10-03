"""End-to-end detection path through the real runtime wiring (RW-3 Task 12, spec 12).

Synthetic JPEG tap frames go into each camera's own FrameTapDispatcher, through the
DetectorRunner (drained on the test thread, so time is the frames' ts_unix), a stub ONNX
detector that reports one person walking past, the Tracker, the EventBuilder (events row
and thumbnail), the RuleEngine and the real ActionQueue worker to a recording fake action.
Closing the track queues the clip on the ClipScheduler, whose ffmpeg step is faked. No
camera, ffmpeg process, network or model weights are involved.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import rtsp_warden.app as app_mod
from rtsp_warden.db import schema as db_schema
from tests.helpers_runtime import (
    ONNX,
    PERSON_RULE,
    STEP,
    T0,
    RecordingAction,
    camera,
    feed,
    install_stub_onnx,
    make_runtime,
    wait_for,
)


@pytest.fixture
def wired(
    tmp_path: Path, clean_db: None, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SimpleNamespace]:
    """Two built cameras (front has the rule, back has none) and a running action queue."""
    install_stub_onnx(monkeypatch)
    hook = RecordingAction("hook")
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: {"hook": hook})
    clip_calls: list[dict[str, Any]] = []

    def fake_build_event_clip(**kwargs: Any) -> Path:
        clip_calls.append(kwargs)
        out = kwargs["camera_root"] / "clips" / f"{kwargs['event_id']}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00\x00\x00\x18ftypisom")
        return out

    monkeypatch.setattr(app_mod, "build_event_clip", fake_build_event_clip)
    front = camera(
        "front",
        tmp_path,
        detect_fps=5,
        track_grace_seconds=3.0,
        min_track_frames=2,
        # The stub person drifts 4 px per frame; this test is about the pipeline, not about
        # stationary suppression (test_the_default_holds_a_slow_object_until_it_has_moved).
        stationary_iou=0,
        detectors=[ONNX],
        rules=[PERSON_RULE],
    )
    rt = make_runtime(tmp_path, [front, camera("back", tmp_path, detectors=[ONNX])])
    rt.build()
    assert rt.action_queue is not None
    rt.action_queue.start()
    try:
        yield SimpleNamespace(rt=rt, hook=hook, clip_calls=clip_calls)
    finally:
        rt.stop_all()


def test_one_person_walking_past_is_one_event_one_thumbnail_one_action_run_and_one_clip(
    wired: SimpleNamespace, tmp_path: Path
) -> None:
    rt, hook = wired.rt, wired.hook

    # Frames 0 and 1: the person is matched twice (min_track_frames 2), so the event opens.
    for step in range(2):
        feed(rt, "front", step)

    rows = db_schema.list_events(camera_name="front")
    assert len(rows) == 1
    event = rows[0]
    assert (event.camera_name, event.label) == ("front", "person")
    assert event.ended_at is None
    assert event.thumbnail_path == f"front/thumbnails/{event.id}.jpg"
    thumbnail = tmp_path / "rec" / event.thumbnail_path
    assert thumbnail.read_bytes()[:2] == b"\xff\xd8"

    # Rules fire on open: one run of the matched action, thumbnail attached, absolute links.
    runs = wait_for(lambda: db_schema.list_action_runs(event.id))
    assert [(run.action_name, run.status) for run in runs] == [("hook", "ok")]
    assert len(hook.sent) == 1
    payload, attachment = hook.sent[0]
    assert (payload.camera, payload.label) == ("front", "person")
    assert payload.event_url == f"http://warden.test:8080/events/{event.id}"
    assert payload.thumbnail_url == f"http://warden.test:8080/events/{event.id}/thumbnail.jpg"
    assert (payload.ended_at, payload.clip_url, payload.test) == (None, None, False)
    assert attachment == thumbnail

    # Frames 2-29: the person walks on, is gone from frame 5, and the track closes after
    # track_grace_seconds without a match. Still one event and one action run.
    for step in range(2, 30):
        feed(rt, "front", step)

    assert [row.id for row in db_schema.list_events(camera_name="front")] == [event.id]
    closed = db_schema.get_event(event.id)
    assert closed is not None and closed.ended_at is not None
    assert len(db_schema.list_action_runs(event.id)) == 1
    assert len(hook.sent) == 1

    # The clip job may run 10 s (clips.post_seconds) + 2 s after the end, and runs once.
    assert rt.clip_scheduler is not None
    ended_at = db_schema.as_utc(closed.ended_at)
    assert ended_at is not None
    not_before = ended_at.timestamp() + 10.0 + 2.0
    assert rt.clip_scheduler.run_due(not_before - 1.0) == 0
    assert wired.clip_calls == []
    assert rt.clip_scheduler.run_due(not_before + 1.0) == 1
    assert [call["event_id"] for call in wired.clip_calls] == [event.id]
    assert wired.clip_calls[0]["camera_root"] == tmp_path / "rec" / "front"
    clipped = db_schema.get_event(event.id)
    assert clipped is not None and clipped.clip_path == f"front/clips/{event.id}.mp4"

    # The other camera saw none of front's frames.
    back = rt.find_runner("back")
    assert back is not None and back.status()["frames_processed"] == 0
    assert db_schema.list_events(camera_name="back") == []


def test_the_default_holds_a_slow_object_until_it_has_moved(
    clean_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With stationary_iou at its default (0.6) the 4 px/frame stub person is held on
    frame 1 (IoU 0.88 with its first box) and opens on frame 4 (IoU 0.58), with that
    frame's time as the event start."""
    install_stub_onnx(monkeypatch)
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: {})
    rt = make_runtime(tmp_path, [camera("front", tmp_path, detectors=[ONNX])], actions=())
    rt.build()
    try:
        for step in range(4):
            feed(rt, "front", step)
        assert db_schema.list_events(camera_name="front") == []
        assert rt.detection_status("front")["stationary_held"] == 1

        feed(rt, "front", 4)

        rows = db_schema.list_events(camera_name="front")
        assert len(rows) == 1
        assert db_schema.as_utc(rows[0].created_at) == datetime.fromtimestamp(
            T0 + 4 * STEP, tz=timezone.utc
        )
        assert rt.detection_status("front")["stationary_held"] == 0
    finally:
        rt.stop_all()

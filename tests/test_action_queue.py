"""ActionQueue (spec 8.3) and ClipScheduler (spec 8.4, ruling R8)."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rtsp_warden.actions import queue as queue_mod
from rtsp_warden.actions.base import ActionPayload, ActionResult
from rtsp_warden.actions.queue import (
    ACTION_QUEUE_MAXSIZE,
    CLIP_DELAY_MARGIN_S,
    ActionJob,
    ActionQueue,
    ClipJob,
    ClipScheduler,
)

T0 = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
T0_NAIVE = T0.replace(tzinfo=None)  # the events table stores naive UTC


def _payload(event_id: int = 1) -> ActionPayload:
    return ActionPayload(
        camera="yard",
        label="person",
        confidence=0.9,
        zone="",
        started_at=T0.isoformat(),
        ended_at=None,
        thumbnail_url=f"http://localhost:8080/events/{event_id}/thumbnail.jpg",
        clip_url=None,
        event_url=f"http://localhost:8080/events/{event_id}",
    )


def _job(
    event_id: int = 1, action_name: str = "phone", attachment: Path | None = None
) -> ActionJob:
    return ActionJob(
        event_id=event_id,
        action_name=action_name,
        payload=_payload(event_id),
        attachment=attachment,
    )


class _FakeDb:
    """Stands in for rtsp_warden.db.schema: records the calls the queue makes."""

    def __init__(self) -> None:
        self.runs: list[tuple[int, str, str, str | None]] = []
        self.updates: list[tuple[int, dict[str, object]]] = []

    def insert_action_run(
        self, *, event_id: int, action_name: str, status: str, error: str | None
    ) -> int:
        self.runs.append((event_id, action_name, status, error))
        return len(self.runs)

    def update_event(self, event_id: int, **fields: object) -> None:
        self.updates.append((event_id, fields))


class _RecordingAction:
    type = "webhook"

    def __init__(self, name: str = "phone", result: ActionResult | None = None) -> None:
        self.name = name
        self.result = result or ActionResult(ok=True)
        self.calls: list[tuple[ActionPayload, Path | None]] = []

    def send(self, payload: ActionPayload, attachment: Path | None = None) -> ActionResult:
        self.calls.append((payload, attachment))
        return self.result

    def test(self) -> ActionResult:
        return ActionResult(ok=True)


class _RaisingAction(_RecordingAction):
    def send(self, payload: ActionPayload, attachment: Path | None = None) -> ActionResult:
        raise RuntimeError("POST https://ntfy.example/secret-topic failed")


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def _warn(msg: str, *args: object, **_kw: object) -> None:
        seen.append(msg % args if args else msg)

    monkeypatch.setattr(queue_mod.log, "warning", _warn)
    return seen


# ---------------------------------------------------------------------------
# ActionQueue.run_one
# ---------------------------------------------------------------------------


def test_defaults_match_spec() -> None:
    assert ACTION_QUEUE_MAXSIZE == 256
    assert CLIP_DELAY_MARGIN_S == 2.0


def test_run_one_ok_writes_ok_row() -> None:
    db = _FakeDb()
    action = _RecordingAction()
    q = ActionQueue({"phone": action}, db=db)

    result = q.run_one(_job(event_id=7))

    assert result == ActionResult(ok=True)
    assert db.runs == [(7, "phone", "ok", None)]
    assert action.calls[0][0].event_url == "http://localhost:8080/events/7"


def test_run_one_passes_existing_attachment(tmp_path: Path) -> None:
    thumb = tmp_path / "7.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xd9")
    action = _RecordingAction()
    q = ActionQueue({"phone": action}, db=_FakeDb())

    q.run_one(_job(attachment=thumb))

    assert action.calls[0][1] == thumb


def test_run_one_missing_attachment_becomes_none(tmp_path: Path) -> None:
    action = _RecordingAction()
    q = ActionQueue({"phone": action}, db=_FakeDb())

    q.run_one(_job(attachment=tmp_path / "expired.jpg"))

    assert action.calls[0][1] is None


def test_run_one_failed_result_writes_failed_row() -> None:
    db = _FakeDb()
    action = _RecordingAction(result=ActionResult(ok=False, error="HTTP 403: forbidden"))
    q = ActionQueue({"phone": action}, db=db)

    result = q.run_one(_job(event_id=3))

    assert result.ok is False
    assert db.runs == [(3, "phone", "failed", "HTTP 403: forbidden")]


def test_run_one_exception_keeps_only_the_type() -> None:
    db = _FakeDb()
    q = ActionQueue({"phone": _RaisingAction()}, db=db)

    result = q.run_one(_job(event_id=4))

    assert result == ActionResult(ok=False, error="RuntimeError while sending")
    assert db.runs == [(4, "phone", "failed", "RuntimeError while sending")]
    assert "ntfy.example" not in db.runs[0][3]


def test_run_one_unknown_action_writes_failed_row() -> None:
    db = _FakeDb()
    q = ActionQueue({}, db=db)

    result = q.run_one(_job(event_id=5, action_name="gone"))

    assert result == ActionResult(ok=False, error="unknown action")
    assert db.runs == [(5, "gone", "failed", "unknown action")]


def test_run_one_survives_a_database_error(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _capture_warnings(monkeypatch)

    class _BrokenDb(_FakeDb):
        def insert_action_run(self, **_kw: object) -> int:
            raise RuntimeError("database is locked")

    action = _RecordingAction()
    q = ActionQueue({"phone": action}, db=_BrokenDb())

    assert q.run_one(_job(event_id=6)).ok is True
    assert len(action.calls) == 1
    assert warnings == ["[actions] could not record the phone run for event 6"]


# ---------------------------------------------------------------------------
# ActionQueue: drop-oldest and the worker thread
# ---------------------------------------------------------------------------


def test_enqueue_drops_oldest_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _capture_warnings(monkeypatch)
    db = _FakeDb()
    q = ActionQueue({"phone": _RecordingAction()}, maxsize=2, db=db)

    for event_id in (1, 2, 3):
        q.enqueue(_job(event_id=event_id))

    assert q.dropped == 1
    assert warnings == ["[actions] queue full; dropped phone for event 1"]
    q.start()
    try:
        assert _wait_for(lambda: len(db.runs) == 2)
    finally:
        q.stop()
    assert [run[0] for run in db.runs] == [2, 3]


def test_worker_runs_jobs_in_order_and_writes_ok_and_failed_rows() -> None:
    db = _FakeDb()
    good = _RecordingAction("phone")
    bad = _RecordingAction("ha", result=ActionResult(ok=False, error="HTTP 500"))
    q = ActionQueue({"phone": good, "ha": bad}, db=db)
    q.start()
    try:
        q.enqueue(_job(event_id=1, action_name="phone"))
        q.enqueue(_job(event_id=1, action_name="ha"))
        assert _wait_for(lambda: len(db.runs) == 2)
    finally:
        q.stop()
    assert db.runs == [(1, "phone", "ok", None), (1, "ha", "failed", "HTTP 500")]


def test_stop_is_prompt_and_discards_queued_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _capture_warnings(monkeypatch)
    db = _FakeDb()
    q = ActionQueue({"phone": _RecordingAction()}, db=db)
    q.start()
    began = time.monotonic()
    q.stop()  # the stop wakes the idle worker instead of waiting for its 1 s poll
    assert time.monotonic() - began < 0.5
    q.enqueue(_job(event_id=9))
    q.stop()  # no worker: the queued job is discarded and logged
    assert db.runs == []
    assert warnings == ["[actions] stopped with 1 queued action(s) not sent"]


def test_worker_records_runs_in_the_database(clean_db: None) -> None:
    from rtsp_warden.db import schema

    event_id = schema.insert_event(
        camera_name="yard",
        event_type="object",
        label="person",
        confidence=0.9,
        zone="",
        track_id=1,
        message="person on yard",
        created_at=T0_NAIVE,
    )
    q = ActionQueue({"phone": _RecordingAction()})  # default db = rtsp_warden.db.schema
    q.start()
    try:
        q.enqueue(_job(event_id=event_id))
        assert _wait_for(lambda: len(schema.list_action_runs(event_id)) == 1)
    finally:
        q.stop()
    run = schema.list_action_runs(event_id)[0]
    assert (run.action_name, run.status, run.error) == ("phone", "ok", None)


# ---------------------------------------------------------------------------
# ClipScheduler
# ---------------------------------------------------------------------------


def _clip_job(event_id: int, not_before: float, camera: str = "yard") -> ClipJob:
    return ClipJob(
        event_id=event_id,
        camera=camera,
        started_at=T0,
        ended_at=T0,
        not_before=not_before,
    )


def test_run_due_runs_only_due_jobs_earliest_first(tmp_path: Path) -> None:
    built: list[int] = []

    def builder(job: ClipJob) -> Path:
        built.append(job.event_id)
        return tmp_path / "yard" / "clips" / f"{job.event_id}.mp4"

    sched = ClipScheduler(builder=builder, db=_FakeDb())
    sched.schedule(_clip_job(2, not_before=1010.0))
    sched.schedule(_clip_job(1, not_before=1000.0))

    assert sched.run_due(999.9) == 0
    assert built == []
    assert sched.run_due(1000.0) == 1
    assert built == [1]
    assert sched.run_due(5000.0) == 1
    assert built == [1, 2]
    assert sched.run_due(5000.0) == 0


def test_success_stores_clip_path_relative_to_output_dir(tmp_path: Path) -> None:
    db = _FakeDb()
    sched = ClipScheduler(builder=lambda job: tmp_path / "yard" / "clips" / "7.ts", db=db)
    sched.schedule(_clip_job(7, not_before=0.0))

    assert sched.run_due(1.0) == 1
    assert db.updates == [(7, {"clip_path": "yard/clips/7.ts"})]


def test_failed_builds_leave_clip_path_unset_and_do_not_stop_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings = _capture_warnings(monkeypatch)
    db = _FakeDb()

    def builder(job: ClipJob) -> Path | None:
        if job.event_id == 1:
            raise RuntimeError("disk full")
        return None

    sched = ClipScheduler(builder=builder, db=db)
    sched.schedule(_clip_job(1, not_before=0.0))
    sched.schedule(_clip_job(2, not_before=0.0))

    assert sched.run_due(1.0) == 2
    assert db.updates == []
    assert warnings == ["[clips] clip job for event 1 failed"]


def test_worker_waits_for_not_before() -> None:
    """A clip built before the window has been recorded would be silently short (R8)."""
    clock = {"now": 1000.0}
    built: list[int] = []

    def builder(job: ClipJob) -> None:
        built.append(job.event_id)

    sched = ClipScheduler(builder=builder, clock=lambda: clock["now"], db=_FakeDb())
    sched.start()
    try:
        sched.schedule(_clip_job(1, not_before=1012.0))
        time.sleep(0.3)
        assert built == []
        clock["now"] = 1012.0
        assert _wait_for(lambda: built == [1])
    finally:
        sched.stop()


def test_stop_drops_pending_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _capture_warnings(monkeypatch)
    sched = ClipScheduler(builder=lambda job: None, clock=lambda: 0.0, db=_FakeDb())
    sched.start()
    sched.schedule(_clip_job(1, not_before=60.0))
    sched.stop()

    assert sched.run_due(1e9) == 0
    assert warnings == ["[clips] stopped with 1 clip job(s) not built"]


def test_clip_path_is_stored_in_the_database(clean_db: None, tmp_path: Path) -> None:
    from rtsp_warden.db import schema

    event_id = schema.insert_event(
        camera_name="yard",
        event_type="object",
        label="person",
        confidence=0.9,
        zone="",
        track_id=1,
        message="person on yard",
        created_at=T0_NAIVE,
    )
    sched = ClipScheduler(builder=lambda job: tmp_path / "yard" / "clips" / f"{event_id}.mp4")
    sched.schedule(_clip_job(event_id, not_before=0.0))

    assert sched.run_due(1.0) == 1
    assert schema.get_event(event_id).clip_path == f"yard/clips/{event_id}.mp4"

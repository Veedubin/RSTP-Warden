"""Runtime wiring for detection and automation (RW-3 Task 12).

Covers the payload link base (runtime.public_url, cli.serve), events -> rules -> action
queue and clip scheduler, the per-camera runner wiring, restart-on-tap-change, the
detection status, the doctor's onnx device check, and review focus 1 (a model that cannot
load must not stop recording or a hot reload). Nothing spawns ffmpeg or touches the network:
process start/stop are recorded stubs and actions are fakes.
"""

from __future__ import annotations

import io
import json
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from rich.console import Console
from typer.testing import CliRunner

import rtsp_warden.app as app_mod
import rtsp_warden.cli as cli_mod
from rtsp_warden.actions.queue import ActionJob, ActionQueue, ClipJob, ClipScheduler
from rtsp_warden.actions.rules import RuleEngine
from rtsp_warden.app import AppRuntime
from rtsp_warden.config import AppConfig, DetectorSpec, RuntimeConfig
from rtsp_warden.db import schema as db_schema
from rtsp_warden.detectors.event_builder import EventBuilder, EventInfo, MotionBurst
from rtsp_warden.detectors.tracking import Tracker
from rtsp_warden.web.config import WebSettings
from tests.helpers_runtime import (
    HOOK,
    MOTION,
    ONNX,
    PERSON_RULE,
    PHONE,
    RecordingAction,
    StubOnnxDetector,
    camera,
    install_stub_onnx,
    jpeg_frame,
    make_runtime,
    start_runtime,
    stub_processes,
)

STARTED = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


class _FakeQueue:
    """ActionQueue stand-in that keeps the jobs instead of running them."""

    def __init__(self) -> None:
        self.jobs: list[ActionJob] = []

    def enqueue(self, job: ActionJob) -> None:
        self.jobs.append(job)


class _FakeScheduler:
    """ClipScheduler stand-in that keeps the scheduled jobs."""

    def __init__(self) -> None:
        self.jobs: list[ClipJob] = []

    def schedule(self, job: ClipJob) -> None:
        self.jobs.append(job)


@pytest.fixture
def cleanup() -> Iterator[Callable[[AppRuntime], None]]:
    """Register runtimes whose runner worker threads must be stopped after the test."""
    made: list[AppRuntime] = []
    yield made.append
    for rt in made:
        rt.stop_all()


def _stored_event(
    tmp_path: Path, camera_name: str, *, label: str = "person", event_type: str = "person"
) -> EventInfo:
    """Insert an open event row with a thumbnail file on disk; return its EventInfo."""
    event_id = db_schema.insert_event(
        camera_name=camera_name,
        event_type=event_type,
        label=label,
        confidence=0.9,
        zone="",
        track_id=1,
        message=f"{label} on {camera_name}",
        created_at=STARTED,
    )
    rel = f"{camera_name}/thumbnails/{event_id}.jpg"
    thumb = tmp_path / "rec" / rel
    thumb.parent.mkdir(parents=True, exist_ok=True)
    thumb.write_bytes(jpeg_frame(0))
    db_schema.update_event(event_id, thumbnail_path=rel)
    return EventInfo(
        id=event_id,
        camera=camera_name,
        label=label,
        confidence=0.9,
        zone="",
        started_at=STARTED,
        ended_at=None,
        thumbnail_path=rel,
        clip_path=None,
        track_id=1,
        event_type=event_type,
    )


def _spy_restarts(rt: AppRuntime, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every request_restart_camera call; the real request is still queued."""
    requested: list[str] = []
    real = rt.request_restart_camera

    def spy(name: str) -> Any:
        requested.append(name)
        return real(name)

    monkeypatch.setattr(rt, "request_restart_camera", spy)
    return requested


# ---------------------------------------------------------------------------
# Link base for action payloads (R19)
# ---------------------------------------------------------------------------


def test_runtime_public_url_is_optional_and_normalised() -> None:
    assert RuntimeConfig().public_url is None
    assert RuntimeConfig(public_url="  ").public_url is None
    url = RuntimeConfig(public_url="https://cams.example.org/warden/").public_url
    assert url == "https://cams.example.org/warden"


def test_runtime_public_url_must_be_http() -> None:
    with pytest.raises(ValidationError, match="public_url must start with http:// or https://"):
        RuntimeConfig(public_url="cams.example.org")


@pytest.mark.parametrize(
    ("configured", "host", "port", "expected"),
    [
        (None, "0.0.0.0", 8080, "http://localhost:8080"),
        (None, "::", 9000, "http://localhost:9000"),
        (None, "192.168.1.5", 8080, "http://192.168.1.5:8080"),
        (None, "fd00::5", 8080, "http://[fd00::5]:8080"),
        ("https://cams.example.org/warden", "0.0.0.0", 8080, "https://cams.example.org/warden"),
    ],
)
def test_public_url_comes_from_config_else_the_bind_address(
    configured: str | None, host: str, port: int, expected: str
) -> None:
    cfg = AppConfig(cameras=[], runtime=RuntimeConfig(public_url=configured))
    assert cli_mod._public_url(cfg, WebSettings(host=host, port=port)) == expected


@pytest.mark.parametrize(
    ("runtime_yaml", "expected"),
    [
        ("", "http://localhost:8181"),
        ("runtime:\n  public_url: https://cams.example.org/\n", "https://cams.example.org"),
    ],
)
def test_serve_hands_the_public_url_to_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_yaml: str, expected: str
) -> None:
    config = textwrap.dedent(
        """\
        cameras:
          - name: cam
            main_url: rtsp://u:p@h/m
            record:
              enabled: false
            proxy:
              enabled: false
        """
    )
    (tmp_path / "config.yaml").write_text(config + runtime_yaml, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    made: list[Any] = []

    class _FakeRuntime:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.cameras: list[Any] = []
            made.append(self)

        def build(self) -> None: ...

        def start(self) -> None: ...

        def run_forever(self) -> None: ...

        def stop_all(self) -> None: ...

    monkeypatch.setattr(cli_mod, "AppRuntime", _FakeRuntime)
    monkeypatch.setattr(cli_mod, "_require_binaries", lambda cfg: None)
    monkeypatch.setattr(cli_mod, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr("rtsp_warden.db.bootstrap.bootstrap_database", lambda **kwargs: None)

    result = CliRunner().invoke(
        cli_mod.app,
        ["serve", "-c", "config.yaml", "--no-web", "--web-host", "0.0.0.0", "--web-port", "8181"],
    )

    assert result.exit_code == 0, result.output
    assert [rt.kwargs["public_url"] for rt in made] == [expected]


def test_payload_for_builds_absolute_links_from_public_url(tmp_path: Path) -> None:
    rt = make_runtime(
        tmp_path, [camera("front", tmp_path)], actions=(), public_url="https://cams.example.org/w/"
    )
    assert rt.public_url == "https://cams.example.org/w/"
    started = datetime(2026, 10, 2, 12, 0)  # naive, as read back from the database (UTC)
    info = EventInfo(
        id=7,
        camera="front",
        label="person",
        confidence=0.87,
        zone="driveway",
        started_at=started,
        ended_at=None,
        thumbnail_path="front/thumbnails/7.jpg",
        clip_path=None,
        track_id=3,
        event_type="person",
    )

    payload = rt.payload_for(info)

    assert (payload.camera, payload.label, payload.confidence, payload.zone) == (
        "front",
        "person",
        0.87,
        "driveway",
    )
    assert payload.started_at == "2026-10-02T12:00:00+00:00"
    assert payload.ended_at is None
    assert payload.event_url == "https://cams.example.org/w/events/7"
    assert payload.thumbnail_url == "https://cams.example.org/w/events/7/thumbnail.jpg"
    assert payload.clip_url is None
    assert payload.test is False

    done = replace(
        info,
        ended_at=datetime(2026, 10, 2, 12, 0, 9, tzinfo=timezone.utc),
        thumbnail_path=None,
        clip_path="front/clips/7.mp4",
        event_type="test",
    )
    payload = rt.payload_for(done)

    assert payload.ended_at == "2026-10-02T12:00:09+00:00"
    assert payload.thumbnail_url is None
    assert payload.clip_url == "https://cams.example.org/w/events/7/clip"
    assert payload.test is True


# ---------------------------------------------------------------------------
# Events -> rules -> action queue and clip scheduler (spec 8.2-8.4, R8, R13, R20)
# ---------------------------------------------------------------------------


def test_build_creates_the_actions_queue_scheduler_and_camera_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = RecordingAction("hook")
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: {"hook": hook})
    rt = make_runtime(tmp_path, [camera("front", tmp_path, rules=[PERSON_RULE])])

    rt.build()

    assert rt.actions == {"hook": hook}
    assert isinstance(rt.action_queue, ActionQueue)
    assert isinstance(rt.clip_scheduler, ClipScheduler)
    assert isinstance(rt.cameras[0].rule_engine, RuleEngine)


def test_dispatch_event_queues_each_matched_action_once_with_the_thumbnail(
    tmp_path: Path, clean_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions = {"hook": RecordingAction("hook"), "phone": RecordingAction("phone")}
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: actions)
    everything = {"name": "everything", "labels": [], "actions": ["hook", "phone"]}
    cam = camera("front", tmp_path, detectors=[ONNX], rules=[PERSON_RULE, everything])
    rt = make_runtime(tmp_path, [cam], actions=(HOOK, PHONE))
    rt.build()
    jobs = _FakeQueue()
    rt.action_queue = jobs  # type: ignore[assignment]
    info = _stored_event(tmp_path, "front")

    decision = rt.dispatch_event(info)

    assert decision is not None
    assert [m.rule.name for m in decision.matched] == ["person-any-time", "everything"]
    assert [(j.event_id, j.action_name) for j in jobs.jobs] == [
        (info.id, "hook"),
        (info.id, "phone"),
    ]
    assert jobs.jobs[0].attachment == tmp_path / "rec" / f"front/thumbnails/{info.id}.jpg"
    assert jobs.jobs[0].payload == rt.payload_for(info)


def test_a_rule_in_cooldown_is_noted_as_suppressed_by_on_the_event(
    tmp_path: Path, clean_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: {"hook": RecordingAction("hook")})
    rt = make_runtime(tmp_path, [camera("front", tmp_path, detectors=[ONNX], rules=[PERSON_RULE])])
    rt.build()
    cam_rt = rt.cameras[0]
    cam_rt.rule_engine = RuleEngine("front", cam_rt.camera.rules, now=lambda: 1000.0)
    jobs = _FakeQueue()
    rt.action_queue = jobs  # type: ignore[assignment]
    first = _stored_event(tmp_path, "front")
    second = _stored_event(tmp_path, "front")

    rt.dispatch_event(first)
    decision = rt.dispatch_event(second)

    assert decision is not None
    assert decision.matched == []
    assert decision.suppressed == ["person-any-time"]
    assert [j.event_id for j in jobs.jobs] == [first.id]
    second_row = db_schema.get_event(second.id)
    assert second_row is not None
    assert json.loads(second_row.metadata_json)["suppressed_by"] == ["person-any-time"]
    first_row = db_schema.get_event(first.id)
    assert first_row is not None
    assert "suppressed_by" not in json.loads(first_row.metadata_json or "{}")


def test_test_events_bypass_cooldown_and_never_queue_a_clip(
    tmp_path: Path, clean_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: {"hook": RecordingAction("hook")})
    rt = make_runtime(tmp_path, [camera("front", tmp_path, detectors=[ONNX], rules=[PERSON_RULE])])
    rt.build()
    cam_rt = rt.cameras[0]
    cam_rt.rule_engine = RuleEngine("front", cam_rt.camera.rules, now=lambda: 1000.0)
    jobs = _FakeQueue()
    clips = _FakeScheduler()
    rt.action_queue = jobs  # type: ignore[assignment]
    rt.clip_scheduler = clips  # type: ignore[assignment]
    rt.dispatch_event(_stored_event(tmp_path, "front"))  # starts the rule's cooldown
    test_event = _stored_event(tmp_path, "front", event_type="test")

    decision = rt.dispatch_event(test_event, bypass_cooldown=True, allow_clip=False)

    assert decision is not None
    assert [m.rule.name for m in decision.matched] == ["person-any-time"]
    assert [j.event_id for j in jobs.jobs][-1] == test_event.id
    assert jobs.jobs[-1].payload.test is True
    rt._on_event_close(replace(test_event, ended_at=STARTED + timedelta(seconds=5)))
    assert clips.jobs == []


def test_closing_a_clip_event_schedules_its_clip_after_post_seconds(
    tmp_path: Path, clean_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: {"hook": RecordingAction("hook")})
    rt = make_runtime(tmp_path, [camera("front", tmp_path, detectors=[ONNX], rules=[PERSON_RULE])])
    rt.build()
    clips = _FakeScheduler()
    rt.action_queue = _FakeQueue()  # type: ignore[assignment]
    rt.clip_scheduler = clips  # type: ignore[assignment]
    info = _stored_event(tmp_path, "front")
    ended = STARTED + timedelta(seconds=8)

    rt.dispatch_event(info)
    rt._on_event_close(replace(info, ended_at=ended))
    rt._on_event_close(replace(info, ended_at=ended))  # a second close queues nothing more

    assert len(clips.jobs) == 1
    job = clips.jobs[0]
    assert (job.event_id, job.camera) == (info.id, "front")
    assert (job.started_at, job.ended_at) == (STARTED, ended)
    assert job.not_before == ended.timestamp() + 10.0 + 2.0  # clips.post_seconds + settle


@pytest.mark.parametrize(("record", "label"), [(True, "car"), (False, "person")])
def test_no_clip_without_a_matching_clip_rule_or_a_recording_camera(
    tmp_path: Path, clean_db: None, monkeypatch: pytest.MonkeyPatch, record: bool, label: str
) -> None:
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: {"hook": RecordingAction("hook")})
    recording = {"enabled": record, "output_dir": str(tmp_path / "rec")}
    cam = camera("front", tmp_path, detectors=[ONNX], rules=[PERSON_RULE], record=recording)
    rt = make_runtime(tmp_path, [cam])
    rt.build()
    clips = _FakeScheduler()
    rt.action_queue = _FakeQueue()  # type: ignore[assignment]
    rt.clip_scheduler = clips  # type: ignore[assignment]
    info = _stored_event(tmp_path, "front", label=label)

    rt.dispatch_event(info)
    rt._on_event_close(replace(info, ended_at=STARTED + timedelta(seconds=8)))

    assert clips.jobs == []


def test_build_clip_cuts_from_the_camera_root_and_returns_a_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_build_event_clip(**kwargs: Any) -> Path:
        seen.append(kwargs)
        return kwargs["camera_root"] / "clips" / f"{kwargs['event_id']}.mp4"

    monkeypatch.setattr(app_mod, "build_event_clip", fake_build_event_clip)
    rt = make_runtime(tmp_path, [camera("front", tmp_path)], actions=())
    ended = STARTED + timedelta(seconds=8)
    job = ClipJob(event_id=7, camera="front", started_at=STARTED, ended_at=ended, not_before=0.0)

    assert rt._build_clip(job) == Path("front/clips/7.mp4")
    assert rt._build_clip(replace(job, camera="gone")) is None

    assert len(seen) == 1
    kwargs = seen[0]
    assert kwargs["camera_root"] == tmp_path / "rec" / "front"
    assert (kwargs["stream"], kwargs["chunk_seconds"]) == ("main", 300)
    assert (kwargs["started_at"], kwargs["ended_at"], kwargs["event_id"]) == (STARTED, ended, 7)
    assert (kwargs["pre_seconds"], kwargs["post_seconds"], kwargs["max_duration"]) == (
        10.0,
        10.0,
        120.0,
    )
    assert kwargs["ffmpeg_path"] == "ffmpeg"


# ---------------------------------------------------------------------------
# Runner wiring, worker lifecycle, restart on tap change, detection status (R18)
# ---------------------------------------------------------------------------


def test_build_gives_each_camera_a_tracking_runner_on_its_own_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    install_stub_onnx(monkeypatch)
    cam = camera("front", tmp_path, detectors=[ONNX], track_grace_seconds=4.5, min_track_frames=3)
    rt = make_runtime(tmp_path, [cam], actions=())
    cleanup(rt)

    rt.build()

    cam_rt = rt.cameras[0]
    runner = rt.find_runner("front")
    assert runner is not None
    assert rt.find_runner("nope") is None
    consumers = list(cam_rt.dispatcher.consumers)
    assert len(consumers) == 1 and consumers[0] is runner
    assert (runner.camera, runner.worker_count, runner.queue_maxsize) == ("front", 1, 8)
    assert [slot.index for slot in runner.slots] == [0]
    assert isinstance(runner.tracker, Tracker)
    assert (runner.tracker.grace_seconds, runner.tracker.min_frames) == (4.5, 3)
    assert isinstance(runner.event_builder, EventBuilder)
    assert isinstance(runner.motion_burst, MotionBurst)
    assert runner.tap_fps == 5.0
    ing = cam_rt.recorder.main
    assert ing is not None and ing.frame_tap_enabled
    assert (ing.frame_tap_fps, ing.frame_tap_scale_width) == (5.0, 640)  # yolox-s input width


def test_start_starts_and_stop_all_stops_the_action_and_clip_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = stub_processes(monkeypatch)
    rt = make_runtime(tmp_path, [camera("front", tmp_path)], actions=())
    rt.build()

    start_runtime(rt, monkeypatch)

    assert calls.index(("queue.start", "")) < calls.index(("rec.start", "front"))
    assert calls.index(("clips.start", "")) < calls.index(("rec.start", "front"))
    calls.clear()

    rt.stop_all()

    assert calls == [("rec.stop", "front"), ("queue.stop", ""), ("clips.stop", "")]


def test_rebuild_with_an_unchanged_tap_requests_no_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    cams = [camera("front", tmp_path, detectors=[MOTION]), camera("idle", tmp_path)]
    rt = make_runtime(tmp_path, cams, actions=())
    cleanup(rt)
    rt.build()
    requested = _spy_restarts(rt, monkeypatch)

    rt.rebuild_camera_detectors("front")
    rt.cameras[1].camera.detect_fps = 2.0  # nothing consumes idle's frames: no restart
    rt.rebuild_camera_detectors("idle")

    assert requested == []
    front = rt.detection_status("front")
    assert front is not None and front["restart_pending"] is False


def test_a_new_detect_fps_restarts_the_ingest_once_with_the_new_tap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    calls = stub_processes(monkeypatch)
    rt = make_runtime(tmp_path, [camera("front", tmp_path, detectors=[MOTION])], actions=())
    cleanup(rt)
    rt.build()
    requested = _spy_restarts(rt, monkeypatch)

    rt.cameras[0].camera.detect_fps = 2.0
    rt.rebuild_camera_detectors("front")
    rt.rebuild_camera_detectors("front")  # a second save before the supervisor's next tick

    assert requested == ["front"]
    status = rt.detection_status("front")
    assert status is not None and status["restart_pending"] is True

    rt._drain_requests()  # what run_forever does at the top of its next tick

    ing = rt.cameras[0].recorder.main
    assert ing is not None and (ing.frame_tap_fps, ing.frame_tap_scale_width) == (2.0, 320)
    assert requested == ["front"]  # the restart's own rebuild found the tap right
    status = rt.detection_status("front")
    assert status is not None and status["restart_pending"] is False
    assert ("rec.stop", "front") in calls and ("rec.start", "front") in calls


def test_enabling_an_onnx_detector_widens_the_tap_to_its_model_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    install_stub_onnx(monkeypatch)
    stub_processes(monkeypatch)
    cam = camera("front", tmp_path, detectors=[MOTION, {**ONNX, "enabled": False}])
    rt = make_runtime(tmp_path, [cam], actions=())
    cleanup(rt)
    rt.build()
    cam_cfg = rt.cameras[0].camera
    assert rt.tap_settings_for(cam_cfg) == (5.0, 320)
    requested = _spy_restarts(rt, monkeypatch)

    cam_cfg.detectors[1].enabled = True
    rt.rebuild_camera_detectors("front")

    assert rt.tap_settings_for(cam_cfg) == (5.0, 640)
    assert requested == ["front"]
    rt._drain_requests()
    ing = rt.cameras[0].recorder.main
    assert ing is not None and (ing.frame_tap_fps, ing.frame_tap_scale_width) == (5.0, 640)
    assert requested == ["front"]


def test_the_first_detector_on_a_camera_without_ingest_gets_it_a_tap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    stub_processes(monkeypatch)
    cam = camera("front", tmp_path, record={"enabled": False})
    rt = make_runtime(tmp_path, [cam], actions=())
    cleanup(rt)
    rt.build()
    assert rt.cameras[0].recorder.main is None  # records, proxies and detects nothing
    requested = _spy_restarts(rt, monkeypatch)

    rt.cameras[0].camera.detectors.append(DetectorSpec(type="motion"))
    rt.rebuild_camera_detectors("front")

    assert requested == ["front"]
    rt._drain_requests()
    ing = rt.cameras[0].recorder.main
    assert ing is not None and ing.frame_tap_enabled
    assert (ing.frame_tap_fps, ing.frame_tap_scale_width) == (5.0, 320)


def test_detection_status_is_json_safe_and_reports_the_tap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    install_stub_onnx(monkeypatch)
    cams = [camera("front", tmp_path, detectors=[ONNX]), camera("idle", tmp_path)]
    rt = make_runtime(tmp_path, cams, actions=())
    cleanup(rt)
    rt.build()

    assert rt.detection_status("nope") is None
    front = rt.detection_status("front")
    idle = rt.detection_status("idle")

    assert front is not None and idle is not None
    json.dumps(front)
    json.dumps(idle)
    assert (front["camera"], front["enabled"], front["restart_pending"]) == ("front", True, False)
    assert (front["tap_fps"], front["tap_width"]) == (5.0, 640)
    assert front["frames_dropped"] == 0
    assert front["detectors"][0]["error"] is None
    assert idle == {
        "camera": "idle",
        "enabled": False,
        "tap_fps": 5.0,
        "tap_width": 320,
        "restart_pending": False,
    }


def test_a_model_that_cannot_load_keeps_recording_and_hot_reload_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    """Review focus 1: an onnx model that is absent and cannot be downloaded (an offline
    home lab) must not stop serve, the camera's recording or a hot reload, and its error
    must reach the detection status the Detection panel shows."""
    install_stub_onnx(monkeypatch)
    monkeypatch.setattr(StubOnnxDetector, "fail_with", "yolox-s: download failed (offline)")
    calls = stub_processes(monkeypatch)
    rt = make_runtime(tmp_path, [camera("front", tmp_path, detectors=[ONNX])], actions=())
    cleanup(rt)
    rt.build()

    start_runtime(rt, monkeypatch)

    assert ("rec.start", "front") in calls
    status = rt.detection_status("front")
    assert status is not None and status["enabled"] is True
    assert status["detectors"][0]["error"] == "yolox-s: download failed (offline)"
    json.dumps(status)

    rt.rebuild_camera_detectors("front")  # hot reload, e.g. a sensitivity save

    status = rt.detection_status("front")
    assert status is not None and status["detectors"][0]["error"] is not None
    assert status["restart_pending"] is False
    assert len(StubOnnxDetector.instances) == 2  # the rebuild built a new detector


def test_a_lifecycle_rebuild_never_requests_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    """_apply_add / _apply_restart rebuild the detectors inside a drain on the main thread:
    that rebuild must never enqueue another lifecycle request, even when the tap looks off."""
    stub_processes(monkeypatch)
    rt = make_runtime(tmp_path, [camera("front", tmp_path, detectors=[MOTION])], actions=())
    cleanup(rt)
    rt.build()
    monkeypatch.setattr(rt, "_tap_restart_needed", lambda cam_rt: True)
    requested = _spy_restarts(rt, monkeypatch)

    restart = rt.request_restart_camera("front")
    add = rt.request_add_camera(
        AppConfig.model_validate(
            {"cameras": [camera("side", tmp_path, detectors=[MOTION])]}
        ).cameras[0]
    )
    rt._drain_requests()

    assert restart.result(timeout=0) is None and add.result(timeout=0) is None
    assert requested == ["front"]  # only the test's own request
    assert rt._requests.qsize() == 0
    assert rt._tap_restarts == {}


def test_a_new_detector_fps_hot_reloads_without_an_ingest_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: Callable[[AppRuntime], None]
) -> None:
    """A per-detector fps change applies through rebuild_camera_detectors alone (spec 6):
    the tap rate is the camera's detect_fps, so no ingest restart is requested."""
    stub_processes(monkeypatch)
    rt = make_runtime(tmp_path, [camera("front", tmp_path, detectors=[MOTION])], actions=())
    cleanup(rt)
    rt.build()
    requested = _spy_restarts(rt, monkeypatch)

    rt.cameras[0].camera.detectors[0].fps = 1.0
    rt.rebuild_camera_detectors("front")

    assert requested == []
    runner = rt.find_runner("front")
    assert runner is not None and runner.slots[0].fps == 1.0


# ---------------------------------------------------------------------------
# doctor: CUDA requested without a CUDA-capable onnxruntime (spec 10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "providers", "expected"),
    [
        (
            "cuda",
            ["CPUExecutionProvider"],
            "WARN (cuda requested, but this onnxruntime has no CUDA provider: "
            "runs on CPU; install the gpu extra)",
        ),
        ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], "OK (cuda)"),
        ("auto", ["CPUExecutionProvider"], "OK (auto: cpu)"),
        ("cpu", [], "FAIL (onnxruntime is not installed)"),
    ],
)
def test_doctor_checks_each_onnx_detector_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    providers: list[str],
    expected: str,
) -> None:
    config = textwrap.dedent(
        f"""\
        cameras:
          - name: cam
            main_url: rtsp://u:p@h/m
            record:
              enabled: false
            proxy:
              enabled: false
            detectors:
              - type: motion
              - type: onnx
                model: yolox-s
                device: {device}
        runtime:
          models_dir: {tmp_path / "models"}
        """
    )
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_onnx_providers", lambda: list(providers))
    out = io.StringIO()
    monkeypatch.setattr(cli_mod, "console", Console(file=out, width=240))

    result = CliRunner().invoke(cli_mod.app, ["doctor", "-c", "config.yaml"])

    assert result.exit_code == 0, result.output
    rows = [line for line in out.getvalue().splitlines() if "onnx cam[" in line]
    assert len(rows) == 1  # one row per enabled onnx detector, labelled with its index
    assert "onnx cam[1]" in rows[0]
    assert expected in rows[0]

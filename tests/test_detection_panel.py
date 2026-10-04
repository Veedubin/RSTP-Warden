"""Detection panel on the camera page (RW-3 Task 15).

Covers GET/POST /cameras/{name}/detection, the index-keyed detector toggle
POST /cameras/{name}/detectors/{index}/enabled, the "Fire test event" route
POST /cameras/{name}/rules/test, and the helpers in web/services/detection.py
they stand on. Offline: no ffmpeg, no camera, no network, no model file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

import rtsp_warden.app as app_mod
from rtsp_warden.actions.rules import RuleEngine
from rtsp_warden.auth import hash_password
from rtsp_warden.config import load_config
from rtsp_warden.db import schema
from rtsp_warden.db.schema import create_user
from rtsp_warden.detectors.event_builder import EventInfo
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.services import detection as detection_service
from tests.helpers_runtime import HOOK, PERSON_RULE, RecordingAction, camera, make_runtime, wait_for

FALLBACK = "device cuda requested but unavailable; using CPU"
MODEL_ERROR = "model yolox-nano unavailable: download failed (offline)"

CONFIG_TEMPLATE = """\
cameras:
  - name: yard
    main_url: rtsp://${{T15_USER}}:${{T15_PASS}}@cam.local/main
    record:
      output_dir: {output_dir}
    detectors:
      - type: motion
        min_area: 500
      - type: motion
        note: an unknown key that must survive
      - type: onnx
        model: yolox-s
        device: cuda
        fps: 2
      - type: onnx
        model: yolox-nano
        device: cpu
    rules:
      - name: person-any-time
        labels: [person]
        clip: true
        actions: [phone]
      - name: cars-only
        labels: [car]
        actions: [phone]
  - name: porch
    main_url: rtsp://u:p@porch.local/main
    record:
      output_dir: {output_dir}
runtime:
  models_dir: {models_dir}
actions:
  - name: phone
    type: webhook
    url: http://ha.local/hook
"""


def _live_status(name: str) -> dict[str, Any] | None:
    """What AppRuntime.detection_status("yard") returns (shape of Task 9's runner status)."""
    if name != "yard":
        return None
    return {
        "frames_processed": 120,
        "frames_dropped": 7,
        "stationary_held": 1,
        "stationary_suppressed": 4,
        "night": True,
        "night_since": 1759536000.0,
        "night_switches": 1,
        "restart_pending": False,
        "detectors": [
            {
                "index": 0,
                "type": "motion",
                "provider": None,
                "fallback_warning": None,
                "fps": 5.0,
                "processed": 100,
                "skipped": 0,
                "errors": 0,
                "setup_error": None,
            },
            {
                "index": 2,
                "type": "onnx",
                "provider": "CPUExecutionProvider",
                "fallback_warning": FALLBACK,
                "fps": 2.0,
                "processed": 40,
                "skipped": 60,
                "errors": 0,
                "setup_error": None,
                "error": None,
            },
            {
                "index": 3,
                "type": "onnx",
                "provider": None,
                "fallback_warning": None,
                "fps": 5.0,
                "processed": 0,
                "skipped": 0,
                "errors": 0,
                "setup_error": None,
                "error": MODEL_ERROR,
            },
        ],
    }


def _login(client: TestClient, username: str = "admin", password: str = "testpass123") -> str:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return client.cookies.get("warden_csrf", "")


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("T15_USER", "u")
    monkeypatch.setenv("T15_PASS", "p")
    (tmp_path / "models").mkdir()
    path = tmp_path / "config.yaml"
    path.write_text(
        CONFIG_TEMPLATE.format(output_dir=tmp_path / "rec", models_dir=tmp_path / "models"),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def env(db_with_user: str, config_path: Path) -> SimpleNamespace:
    """App with a display runtime (live status) and a MagicMock mutation runtime."""
    cfg = load_config(config_path)
    display = SimpleNamespace(cameras=[], detection_status=_live_status)
    mutate = MagicMock()
    app = create_app(
        WebSettings(),
        cfg=cfg,
        runtime_provider=lambda: display,
        config_path=config_path,
        runtime=mutate,
    )
    client = TestClient(app)
    csrf = _login(client)
    return SimpleNamespace(
        app=app, cfg=cfg, client=client, csrf=csrf, runtime=mutate, path=config_path
    )


def _post(env: SimpleNamespace, url: str, data: dict[str, str] | None = None, htmx: bool = True):
    headers = {"X-CSRF-Token": env.csrf}
    if htmx:
        headers["HX-Request"] = "true"
    return env.client.post(
        url,
        data={**(data or {}), "csrf_token": env.csrf},
        headers=headers,
        follow_redirects=False,
    )


def _raw_camera(path: Path, name: str = "yard") -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return next(c for c in raw["cameras"] if c["name"] == name)


# --- the panel ---------------------------------------------------------------------------


def test_detail_page_includes_the_detection_panel_shell(env: SimpleNamespace) -> None:
    r = env.client.get("/cameras/yard")
    assert r.status_code == 200
    assert 'id="detection-panel"' in r.text
    assert 'hx-get="/cameras/yard/detection"' in r.text


def test_panel_shows_detect_fps_rules_and_restart_note(env: SimpleNamespace) -> None:
    r = env.client.get("/cameras/yard/detection")
    assert r.status_code == 200
    assert 'name="detect_fps"' in r.text
    assert 'value="5"' in r.text
    assert "restarts this camera's ingest" in r.text
    assert 'name="track_grace_seconds"' in r.text
    assert 'name="min_track_frames"' in r.text
    assert "person-any-time" in r.text
    assert "cars-only" in r.text
    assert "phone" in r.text
    assert "Fire test event" in r.text
    assert 'hx-post="/cameras/yard/rules/test"' in r.text
    assert "120 frames analysed" in r.text
    assert "7 dropped" in r.text
    assert "/cameras/yard/detection-classes" in r.text
    assert "/cameras/yard/zones" in r.text


def test_panel_shows_night_mode(env: SimpleNamespace) -> None:
    r = env.client.get("/cameras/yard/detection")
    assert r.status_code == 200
    assert "night mode on" in r.text


def test_panel_lists_detectors_with_index_keyed_toggles(env: SimpleNamespace) -> None:
    r = env.client.get("/cameras/yard/detection")
    for index in range(4):
        assert f'hx-post="/cameras/yard/detectors/{index}/enabled"' in r.text
    assert 'hx-target="#detector-list"' in r.text
    assert "/detectors/motion/enabled" not in r.text
    assert "yolox-s" in r.text and "yolox-nano" in r.text
    assert "min_area=500" in r.text


def test_panel_shows_provider_fallback_and_model_error(env: SimpleNamespace) -> None:
    """(review focus) An onnx model that cannot be downloaded shows its error on the panel."""
    r = env.client.get("/cameras/yard/detection")
    assert r.status_code == 200
    assert ">CPU<" in r.text
    assert FALLBACK in r.text
    assert MODEL_ERROR in r.text
    r = env.client.get("/cameras/yard/detectors")
    assert r.status_code == 200
    assert MODEL_ERROR in r.text


def test_panel_without_a_runtime_says_detection_is_not_running(
    db_with_user: str, config_path: Path
) -> None:
    app = create_app(WebSettings(), cfg=load_config(config_path), runtime_provider=lambda: None)
    client = TestClient(app)
    _login(client)
    r = client.get("/cameras/yard/detection")
    assert r.status_code == 200
    assert "Detection is not running for this camera" in r.text
    assert "not running" in client.get("/cameras/yard/detectors").text


def test_panel_and_detector_list_404_for_unknown_camera(env: SimpleNamespace) -> None:
    assert env.client.get("/cameras/ghost/detection").status_code == 404
    assert env.client.get("/cameras/ghost/detectors").status_code == 404


def test_viewer_sees_no_controls_and_cannot_post(db_with_user: str, config_path: Path) -> None:
    create_user("viewer", hash_password("viewerpass123"), is_admin=False)
    app = create_app(WebSettings(), cfg=load_config(config_path), config_path=config_path)
    client = TestClient(app)
    csrf = _login(client, "viewer", "viewerpass123")
    r = client.get("/cameras/yard/detection")
    assert r.status_code == 200
    assert "/enabled" not in r.text
    assert "Fire test event" not in r.text
    assert 'name="detect_fps"' not in r.text
    headers = {"X-CSRF-Token": csrf}
    for url in (
        "/cameras/yard/detectors/0/enabled",
        "/cameras/yard/rules/test",
        "/cameras/yard/detection",
    ):
        r = client.post(url, data={"csrf_token": csrf}, headers=headers, follow_redirects=False)
        assert r.status_code == 403, url


# --- detector toggle by index ------------------------------------------------------------


def test_toggle_by_index_changes_only_that_entry(env: SimpleNamespace) -> None:
    before = _raw_camera(env.path)
    r = _post(env, "/cameras/yard/detectors/1/enabled", {"enabled": "false"})
    assert r.status_code == 200
    assert 'id="detector-1"' in r.text
    dets = env.cfg.cameras[0].detectors
    assert [d.enabled for d in dets] == [True, False, True, True]
    after = _raw_camera(env.path)
    assert after["detectors"][0] == before["detectors"][0]
    assert after["detectors"][1] == {**before["detectors"][1], "enabled": False}
    assert after["detectors"][1]["note"] == "an unknown key that must survive"
    assert after["detectors"][2] == before["detectors"][2]
    text = env.path.read_text(encoding="utf-8")
    assert "${T15_USER}:${T15_PASS}" in text
    env.runtime.rebuild_camera_detectors.assert_called_once_with("yard")


def test_toggle_plain_form_post_redirects(env: SimpleNamespace) -> None:
    r = _post(env, "/cameras/yard/detectors/2/enabled", {"enabled": "false"}, htmx=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/yard"
    assert env.cfg.cameras[0].detectors[2].enabled is False
    assert _raw_camera(env.path)["detectors"][2]["enabled"] is False


def test_toggle_unknown_index_is_404_and_non_integer_is_422(env: SimpleNamespace) -> None:
    assert _post(env, "/cameras/yard/detectors/9/enabled", {"enabled": "false"}).status_code == 404
    assert _post(env, "/cameras/yard/detectors/-1/enabled", {"enabled": "false"}).status_code == 404
    assert (
        _post(env, "/cameras/yard/detectors/abc/enabled", {"enabled": "false"}).status_code == 422
    )
    assert _post(env, "/cameras/ghost/detectors/0/enabled", {"enabled": "false"}).status_code == 404
    env.runtime.rebuild_camera_detectors.assert_not_called()


def test_toggle_refuses_when_the_file_changed_under_it(env: SimpleNamespace) -> None:
    raw = yaml.safe_load(env.path.read_text(encoding="utf-8"))
    del raw["cameras"][0]["detectors"][0]  # index 1 is now the first onnx entry
    env.path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    before = env.path.read_text(encoding="utf-8")
    r = _post(env, "/cameras/yard/detectors/1/enabled", {"enabled": "false"})
    assert r.status_code == 409
    assert env.path.read_text(encoding="utf-8") == before
    assert env.cfg.cameras[0].detectors[1].enabled is True
    env.runtime.rebuild_camera_detectors.assert_not_called()


def test_toggle_survives_a_failing_rebuild(env: SimpleNamespace) -> None:
    """(review focus) A rebuild that raises (model unavailable) does not break the toggle."""
    env.runtime.rebuild_camera_detectors.side_effect = RuntimeError(MODEL_ERROR)
    r = _post(env, "/cameras/yard/detectors/3/enabled", {"enabled": "false"})
    assert r.status_code == 200
    assert env.cfg.cameras[0].detectors[3].enabled is False
    assert _raw_camera(env.path)["detectors"][3]["enabled"] is False


def test_toggle_reports_a_config_write_failure(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path, data: dict) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(detection_service, "_locked_write_yaml", refuse)
    r = _post(env, "/cameras/yard/detectors/0/enabled", {"enabled": "false"})
    assert r.status_code == 200
    assert "Permission denied" in r.text
    assert str(env.path) in r.text
    assert env.cfg.cameras[0].detectors[0].enabled is False


# --- detect_fps and tracking settings ----------------------------------------------------


def _settings(
    detect_fps: str = "5", grace: str = "3", frames: str = "2", stationary: str = "0.6"
) -> dict[str, str]:
    return {
        "detect_fps": detect_fps,
        "track_grace_seconds": grace,
        "min_track_frames": frames,
        "stationary_iou": stationary,
    }


def test_save_detect_fps_persists_and_rebuilds_once(env: SimpleNamespace) -> None:
    """The rebuild restarts the ingest itself (R18); the route must not restart it again."""
    r = _post(env, "/cameras/yard/detection", _settings(detect_fps="2.5"))
    assert r.status_code == 200
    assert "ingest restarts to use 2.5 fps" in r.text
    assert env.cfg.cameras[0].detect_fps == 2.5
    raw = _raw_camera(env.path)
    assert raw["detect_fps"] == 2.5
    assert "track_grace_seconds" not in raw
    assert "min_track_frames" not in raw
    env.runtime.rebuild_camera_detectors.assert_called_once_with("yard")
    env.runtime.request_restart_camera.assert_not_called()


def test_save_tracking_only_reloads_detectors_without_a_restart(env: SimpleNamespace) -> None:
    r = _post(env, "/cameras/yard/detection", _settings(grace="4.5", frames="3"))
    assert r.status_code == 200
    assert "Detectors reloaded" in r.text
    cam = env.cfg.cameras[0]
    assert (cam.track_grace_seconds, cam.min_track_frames) == (4.5, 3)
    raw = _raw_camera(env.path)
    assert (raw["track_grace_seconds"], raw["min_track_frames"]) == (4.5, 3)
    assert "detect_fps" not in raw
    env.runtime.rebuild_camera_detectors.assert_called_once_with("yard")
    env.runtime.request_restart_camera.assert_not_called()


def test_save_without_changes_writes_nothing(env: SimpleNamespace) -> None:
    before = env.path.read_text(encoding="utf-8")
    r = _post(env, "/cameras/yard/detection", _settings())
    assert r.status_code == 200
    assert "No changes to save" in r.text
    assert env.path.read_text(encoding="utf-8") == before
    env.runtime.rebuild_camera_detectors.assert_not_called()
    env.runtime.request_restart_camera.assert_not_called()


@pytest.mark.parametrize(
    "data",
    [
        _settings(detect_fps="0.1"),
        _settings(detect_fps="31"),
        _settings(detect_fps="abc"),
        _settings(detect_fps="nan"),
        _settings(grace="0"),
        _settings(grace="inf"),
        _settings(frames="0"),
        _settings(frames="2.5"),
        _settings(stationary="1.5"),
        _settings(stationary="-0.1"),
        _settings(stationary="abc"),
        {"detect_fps": "5"},
    ],
)
def test_save_rejects_bad_values(env: SimpleNamespace, data: dict[str, str]) -> None:
    before = env.path.read_text(encoding="utf-8")
    assert _post(env, "/cameras/yard/detection", data).status_code == 422
    assert env.path.read_text(encoding="utf-8") == before
    assert env.cfg.cameras[0].detect_fps == 5.0


def test_save_rejects_detect_fps_below_an_explicit_detector_fps(env: SimpleNamespace) -> None:
    before = env.path.read_text(encoding="utf-8")
    r = _post(env, "/cameras/yard/detection", _settings(detect_fps="1"))
    assert r.status_code == 422
    assert "detectors[2] (onnx)" in r.json()["detail"]
    assert env.path.read_text(encoding="utf-8") == before
    env.runtime.rebuild_camera_detectors.assert_not_called()


def test_save_reports_a_config_write_failure_and_keeps_the_change(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path, data: dict) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(detection_service, "_locked_write_yaml", refuse)
    r = _post(env, "/cameras/yard/detection", _settings(detect_fps="2"))
    assert r.status_code == 200
    assert "Permission denied" in r.text
    assert str(env.path) in r.text
    assert env.cfg.cameras[0].detect_fps == 2.0
    env.runtime.rebuild_camera_detectors.assert_called_once_with("yard")


def test_save_plain_form_post_redirects(env: SimpleNamespace) -> None:
    r = _post(env, "/cameras/yard/detection", _settings(grace="5"), htmx=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/yard"


# --- fire test event --------------------------------------------------------------------


@pytest.fixture
def dispatch_env(env: SimpleNamespace) -> SimpleNamespace:
    """Replace the MagicMock runtime with one whose dispatch_event records its calls.

    The fake answers with a real RuleEngine decision, the way AppRuntime.dispatch_event
    (Task 12) does; the end-to-end test below uses the real AppRuntime.
    """
    calls: list[tuple[Any, dict[str, Any]]] = []

    def dispatch_event(info: Any, **kwargs: Any) -> Any:
        calls.append((info, kwargs))
        cam = next(c for c in env.cfg.cameras if c.name == info.camera)
        return RuleEngine(cam.name, cam.rules).evaluate(info, bypass_cooldown=True)

    env.app.state.runtime = SimpleNamespace(dispatch_event=dispatch_event)
    env.calls = calls
    return env


def _event_id(text: str) -> int:
    marker = "Test event #"
    start = text.index(marker) + len(marker)
    return int(text[start : text.index("<", start)])


def test_fire_test_event_records_a_test_row_and_dispatches_it(
    dispatch_env: SimpleNamespace,
) -> None:
    r = _post(dispatch_env, "/cameras/yard/rules/test")
    assert r.status_code == 200
    event_id = _event_id(r.text)

    row = schema.get_event(event_id)
    assert row.event_type == "test"
    assert row.label == "person"
    assert row.camera_name == "yard"
    assert row.ended_at == row.created_at
    assert row.thumbnail_path == f"yard/thumbnails/{event_id}.jpg"
    thumb = Path(dispatch_env.cfg.cameras[0].record.output_dir) / row.thumbnail_path
    assert thumb.read_bytes()[:2] == b"\xff\xd8"

    [(info, kwargs)] = dispatch_env.calls
    assert kwargs == {"bypass_cooldown": True, "allow_clip": False}
    assert (info.id, info.event_type, info.label) == (event_id, "test", "person")
    assert info.thumbnail_path == row.thumbnail_path
    assert info.ended_at == info.started_at

    assert "person-any-time" in r.text
    assert "Not matched" in r.text and "cars-only" in r.text
    assert "Sent to the action queue: <strong>phone</strong>" in r.text
    assert f'href="/events/{event_id}"' in r.text


def test_fire_test_event_on_a_camera_without_rules(dispatch_env: SimpleNamespace) -> None:
    r = _post(dispatch_env, "/cameras/porch/rules/test")
    assert r.status_code == 200
    assert "no rules" in r.text
    assert "Sent to the action queue" not in r.text
    assert schema.get_event(_event_id(r.text)).event_type == "test"


def test_fire_test_event_without_a_runtime_still_explains_the_rules(env: SimpleNamespace) -> None:
    env.app.state.runtime = None
    r = _post(env, "/cameras/yard/rules/test")
    assert r.status_code == 200
    assert "person-any-time" in r.text
    assert "Actions were not sent" in r.text
    assert schema.list_action_runs(_event_id(r.text)) == []


def test_fire_test_event_unknown_camera_404(dispatch_env: SimpleNamespace) -> None:
    assert _post(dispatch_env, "/cameras/ghost/rules/test").status_code == 404
    assert dispatch_env.calls == []


def test_fire_test_event_runs_the_real_rules_and_action_queue(
    tmp_path: Path, db_with_user: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end with Task 12's AppRuntime: the action runs, no clip, no cooldown kept."""
    hook = RecordingAction("hook")
    monkeypatch.setattr(app_mod, "build_actions", lambda cfg: {"hook": hook})
    rt = make_runtime(tmp_path, [camera("front", tmp_path, rules=[PERSON_RULE])], actions=(HOOK,))
    rt.build()
    assert rt.action_queue is not None
    rt.action_queue.start()
    try:
        app = create_app(WebSettings(), cfg=rt.cfg, runtime_provider=lambda: rt, runtime=rt)
        client = TestClient(app)
        csrf = _login(client)
        headers = {"X-CSRF-Token": csrf, "HX-Request": "true"}
        ids = []
        for _ in range(2):  # PERSON_RULE has a 60 s cooldown; the test bypasses it both times
            r = client.post("/cameras/front/rules/test", data={"csrf_token": csrf}, headers=headers)
            assert r.status_code == 200
            ids.append(_event_id(r.text))

        for event_id in ids:
            runs = wait_for(lambda event_id=event_id: schema.list_action_runs(event_id))
            assert [(run.action_name, run.status) for run in runs] == [("hook", "ok")]
            assert event_id not in rt._clip_events  # allow_clip=False: no clip job
        payload, attachment = hook.sent[0]
        assert payload.test is True
        assert payload.event_url == f"http://warden.test:8080/events/{ids[0]}"
        assert attachment == tmp_path / "rec" / f"front/thumbnails/{ids[0]}.jpg"

        # No cooldown stamp was recorded: a real person event right after still fires.
        cam_rt = rt.find_camera("front")
        real = EventInfo(
            id=ids[-1] + 1,
            camera="front",
            label="person",
            confidence=0.9,
            zone="",
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            thumbnail_path=None,
            clip_path=None,
            track_id=1,
            event_type="object",
        )
        decision = cam_rt.rule_engine.evaluate(real)
        assert [m.rule.name for m in decision.matched] == ["person-any-time"]
    finally:
        rt.stop_all()


# --- per-detector fps (spec 6: a hot reload, never an ingest restart) ---------------------


def test_detector_fps_is_saved_and_hot_reloaded(env: SimpleNamespace) -> None:
    before = _raw_camera(env.path)
    r = _post(env, "/cameras/yard/detectors/3/fps", {"fps": "1.5"})
    assert r.status_code == 200
    assert 'id="detector-3"' in r.text
    assert env.cfg.cameras[0].detectors[3].fps == 1.5
    after = _raw_camera(env.path)
    assert after["detectors"][3] == {**before["detectors"][3], "fps": 1.5}
    assert after["detectors"][2] == before["detectors"][2]
    assert "${T15_USER}:${T15_PASS}" in env.path.read_text(encoding="utf-8")
    env.runtime.rebuild_camera_detectors.assert_called_once_with("yard")
    env.runtime.request_restart_camera.assert_not_called()


def test_detector_fps_left_empty_runs_at_the_camera_rate(env: SimpleNamespace) -> None:
    r = _post(env, "/cameras/yard/detectors/2/fps", {"fps": ""})
    assert r.status_code == 200
    assert env.cfg.cameras[0].detectors[2].fps is None
    assert _raw_camera(env.path)["detectors"][2]["fps"] is None
    r = _post(env, "/cameras/yard/detectors/2/fps", {"fps": "2"}, htmx=False)
    assert r.status_code == 303
    assert env.cfg.cameras[0].detectors[2].fps == 2.0


def test_detector_fps_out_of_range_is_refused(env: SimpleNamespace) -> None:
    before = env.path.read_text(encoding="utf-8")
    for bad in ("6", "0", "-1", "abc", "nan"):
        r = _post(env, "/cameras/yard/detectors/0/fps", {"fps": bad})
        assert r.status_code == 422, bad
    assert _post(env, "/cameras/yard/detectors/9/fps", {"fps": "1"}).status_code == 404
    assert _post(env, "/cameras/ghost/detectors/0/fps", {"fps": "1"}).status_code == 404
    assert env.path.read_text(encoding="utf-8") == before
    assert env.cfg.cameras[0].detectors[0].fps is None
    env.runtime.rebuild_camera_detectors.assert_not_called()


def test_detector_list_offers_an_fps_input_to_admins_only(
    env: SimpleNamespace, config_path: Path
) -> None:
    r = env.client.get("/cameras/yard/detectors")
    assert 'hx-post="/cameras/yard/detectors/0/fps"' in r.text
    assert 'name="fps"' in r.text
    create_user("viewer", hash_password("viewerpass123"), is_admin=False)
    app = create_app(WebSettings(), cfg=load_config(config_path), config_path=config_path)
    client = TestClient(app)
    csrf = _login(client, "viewer", "viewerpass123")
    assert "/fps" not in client.get("/cameras/yard/detectors").text
    r = client.post(
        "/cameras/yard/detectors/0/fps",
        data={"fps": "1", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 403


# --- rebuilds and config writes stay off the event loop -----------------------------------


@pytest.mark.parametrize(
    ("url", "data"),
    [
        ("/cameras/yard/detectors/1/enabled", {"enabled": "false"}),
        ("/cameras/yard/detectors/1/fps", {"fps": "1"}),
        (
            "/cameras/yard/detection",
            {
                "detect_fps": "4",
                "track_grace_seconds": "3",
                "min_track_frames": "2",
                "stationary_iou": "0.6",
            },
        ),
        ("/cameras/yard/sensitivity", {"sensitivity": "60", "action": "save_and_reload"}),
        ("/cameras/yard/detection-classes", {"classes_mode": "all", "action": "save_and_reload"}),
        ("/cameras/yard/reload", {}),
    ],
)
def test_rebuilds_and_writes_run_off_the_event_loop(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, url: str, data: dict[str, str]
) -> None:
    """A rebuild loads ONNX sessions and joins the old runner's worker; a write is file IO.
    Inside the async routes both must run in the threadpool, never on the event loop."""
    import asyncio

    import rtsp_warden.web.routes.detection as detection_routes

    on_loop: list[bool] = []

    def has_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    env.runtime.rebuild_camera_detectors.side_effect = lambda *a, **k: on_loop.append(has_loop())
    for helper in ("_persist_camera_field", "_persist_detector_entry"):
        real = getattr(detection_routes, helper)

        def wrapped(*args: Any, _real: Any = real, **kwargs: Any) -> Any:
            on_loop.append(has_loop())
            return _real(*args, **kwargs)

        monkeypatch.setattr(detection_routes, helper, wrapped)

    r = _post(env, url, data)

    assert r.status_code in (200, 303), r.text
    env.runtime.rebuild_camera_detectors.assert_called_once()
    assert on_loop and not any(on_loop)


def test_toggle_during_stalled_download_answers_fast(
    db_with_user: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(review finding) While the camera's onnx model is still downloading (its runner's
    worker is stuck in the download), a detector toggle answers at once: the new runner
    is set up without downloading and the old one is torn down in the background."""
    import threading
    import time

    from rtsp_warden.detectors.builtin import onnx as onnx_mod
    from tests.helpers_onnx import write_model_dir, yolox_output
    from tests.helpers_runtime import jpeg_frame

    models_dir = tmp_path / "models"
    write_model_dir(models_dir, yolox_output((64, 64), 2, {}))
    (models_dir / "tiny" / "tiny.onnx").unlink()  # not downloaded yet
    started, release = threading.Event(), threading.Event()

    def stalled_download(desc: Any, models: Path, **_kw: object) -> Path:
        started.set()
        release.wait(timeout=30)
        raise OSError("network is unreachable")

    monkeypatch.setattr(onnx_mod, "ensure_model_file", stalled_download)
    raw = {
        "cameras": [
            {
                "name": "yard",
                "main_url": "rtsp://u:p@h/m",
                "record": {"enabled": False, "output_dir": str(tmp_path / "rec")},
                "proxy": {"enabled": False},
                "detectors": [
                    {"type": "onnx", "model": "tiny", "device": "cpu"},
                    {"type": "motion"},
                ],
            }
        ],
        "runtime": {"models_dir": str(models_dir)},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = load_config(path)
    rt = app_mod.AppRuntime(cfg=cfg)
    rt.build()
    for runner in rt.detector_runners:
        runner.setup()
    try:
        cam_rt = rt.find_camera("yard")
        assert cam_rt is not None
        cam_rt.dispatcher.dispatch(
            camera="yard", stream="main", jpeg_bytes=jpeg_frame(0), ts_unix=time.time()
        )
        assert started.wait(timeout=5.0)  # the runner's worker is now inside the download

        app = create_app(
            WebSettings(), cfg=cfg, runtime_provider=lambda: rt, config_path=path, runtime=rt
        )
        client = TestClient(app)
        csrf = _login(client)
        t0 = time.monotonic()
        r = client.post(
            "/cameras/yard/detectors/1/enabled",
            data={"enabled": "false", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf, "HX-Request": "true"},
            follow_redirects=False,
        )
        elapsed = time.monotonic() - t0

        assert r.status_code == 200
        assert elapsed < 2.0
        assert cfg.cameras[0].detectors[1].enabled is False
        assert client.get("/healthz").status_code in (200, 503)
    finally:
        release.set()
        rt.stop_all()


# --- review fixes (2026-10-03 code review) -----------------------------------------------


def test_retention_save_rejects_bad_numbers_with_422(env: SimpleNamespace) -> None:
    """Form text that is not a number, or a value RetentionConfig refuses, is a 422, not a 500."""
    before = env.path.read_text(encoding="utf-8")
    r = _post(env, "/cameras/yard/retention", {"max_days": "seven"}, htmx=False)
    assert r.status_code == 422
    r = _post(env, "/cameras/yard/retention", {"max_gb": "-1"}, htmx=False)
    assert r.status_code == 422
    assert env.cfg.cameras[0].retention is None
    assert env.path.read_text(encoding="utf-8") == before


def test_retention_save_patches_only_that_camera(env: SimpleNamespace) -> None:
    """A retention block another camera got by hand after startup survives a save."""
    raw = yaml.safe_load(env.path.read_text(encoding="utf-8"))
    raw["cameras"][1]["retention"] = {"max_days": 3}
    env.path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    r = _post(
        env,
        "/cameras/yard/retention",
        {"max_days": "7", "max_gb": "", "keep_last_n": "", "cleanup_interval_seconds": ""},
        htmx=False,
    )
    assert r.status_code == 303
    raw = yaml.safe_load(env.path.read_text(encoding="utf-8"))
    assert raw["cameras"][0]["retention"] == {
        "max_days": 7,
        "keep_last_n": 0,
        "cleanup_interval_seconds": 300,
    }
    assert raw["cameras"][1]["retention"] == {"max_days": 3}


def test_detection_settings_refuse_when_camera_left_the_file(env: SimpleNamespace) -> None:
    """A camera renamed in config.yaml since startup: 409, nothing changed, no rebuild."""
    raw = yaml.safe_load(env.path.read_text(encoding="utf-8"))
    raw["cameras"][0]["name"] = "renamed"
    env.path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    before = env.path.read_text(encoding="utf-8")
    r = _post(
        env,
        "/cameras/yard/detection",
        {
            "detect_fps": "4",
            "track_grace_seconds": "3",
            "min_track_frames": "2",
            "stationary_iou": "0.6",
        },
    )
    assert r.status_code == 409
    assert env.cfg.cameras[0].detect_fps == 5.0
    assert env.path.read_text(encoding="utf-8") == before
    env.runtime.rebuild_camera_detectors.assert_not_called()


# --- stationary suppression in the panel (RW-4) ------------------------------------------


def test_panel_shows_the_stationary_field_and_the_held_back_count(env: SimpleNamespace) -> None:
    r = env.client.get("/cameras/yard/detection")
    assert r.status_code == 200
    assert 'name="stationary_iou"' in r.text
    assert 'value="0.6"' in r.text
    assert "4 objects that never moved were held back" in r.text
    assert "1 held back right now" in r.text


def test_save_stationary_iou_persists_and_reloads_detectors(env: SimpleNamespace) -> None:
    r = _post(env, "/cameras/yard/detection", _settings(stationary="0"))
    assert r.status_code == 200
    assert "Detectors reloaded" in r.text
    assert env.cfg.cameras[0].stationary_iou == 0.0
    raw = _raw_camera(env.path)
    assert raw["stationary_iou"] == 0
    assert "detect_fps" not in raw
    env.runtime.rebuild_camera_detectors.assert_called_once_with("yard")
    env.runtime.request_restart_camera.assert_not_called()

"""Config for detection and automation (RW-3 Task 1).

Covers the one-time deprecation helper, the new DetectorSpec fields and
deprecations, the camera-level detection fields, rules, and the frame tap
settings the recorder hands to ffmpeg.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rtsp_warden import deprecations
from rtsp_warden.config import (
    DETECT_FPS_MAX,
    DETECT_FPS_MIN,
    TAP_MIN_WIDTH,
    AppConfig,
    CameraConfig,
    DetectorSpec,
    RuleConfig,
    RuntimeConfig,
    compute_tap_settings,
    load_config,
    parse_between,
)
from rtsp_warden.frame_tap import FrameTapDispatcher
from rtsp_warden.recorder import CameraRecorder


@pytest.fixture
def warnings_seen(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture deprecation warnings, starting from an empty once-only registry.

    The module logger is patched directly (not caplog): Alembic's fileConfig
    disables existing loggers once any DB test has run.
    """
    messages: list[str] = []
    monkeypatch.setattr(deprecations, "_WARNED", set())
    monkeypatch.setattr(deprecations.log, "warning", lambda msg, *args: messages.append(msg % args))
    return messages


# ---------------------------------------------------------------------------
# deprecations.warn_once
# ---------------------------------------------------------------------------


def test_warn_once_logs_first_call_only(warnings_seen: list[str]) -> None:
    deprecations.warn_once("k1", "old %s is deprecated", "thing")
    deprecations.warn_once("k1", "old %s is deprecated", "thing")
    deprecations.warn_once("k2", "other %s", "key")
    assert warnings_seen == ["old thing is deprecated", "other key"]


# ---------------------------------------------------------------------------
# DetectorSpec
# ---------------------------------------------------------------------------


def test_detector_spec_new_field_defaults() -> None:
    spec = DetectorSpec(type="motion")
    assert spec.fps is None
    assert spec.model is None
    assert spec.device == "auto"
    assert spec.events is None
    assert spec.interval_seconds is None


def test_onnx_type_and_fields_accepted() -> None:
    spec = DetectorSpec.model_validate(
        {"type": "onnx", "model": "yolox-nano", "device": "cuda", "fps": 2, "min_confidence": 0.6}
    )
    assert spec.type == "onnx"
    assert spec.model == "yolox-nano"
    assert spec.device == "cuda"
    assert spec.fps == 2.0
    assert spec.min_confidence == 0.6


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValidationError, match="device"):
        DetectorSpec.model_validate({"type": "onnx", "device": "gpu"})


@pytest.mark.parametrize("bad", [0, -1.0])
def test_fps_must_be_positive(bad: float) -> None:
    with pytest.raises(ValidationError, match="fps must be > 0"):
        DetectorSpec(type="motion", fps=bad)


def test_interval_seconds_converts_to_fps_with_one_warning(warnings_seen: list[str]) -> None:
    spec = DetectorSpec.model_validate({"type": "motion", "interval_seconds": 2.0})
    again = DetectorSpec.model_validate({"type": "motion", "interval_seconds": 2.0})
    assert spec.fps == 0.5
    assert again.fps == 0.5
    assert spec.interval_seconds is None
    assert "interval_seconds" not in spec.model_dump()
    assert len(warnings_seen) == 1
    assert "interval_seconds" in warnings_seen[0]
    assert "use fps" in warnings_seen[0]


def test_interval_seconds_warns_again_for_a_new_value(warnings_seen: list[str]) -> None:
    DetectorSpec.model_validate({"type": "motion", "interval_seconds": 2.0})
    DetectorSpec.model_validate({"type": "motion", "interval_seconds": 4.0})
    assert len(warnings_seen) == 2


def test_explicit_fps_wins_over_interval_seconds(warnings_seen: list[str]) -> None:
    spec = DetectorSpec.model_validate({"type": "motion", "interval_seconds": 2.0, "fps": 3})
    assert spec.fps == 3.0
    assert len(warnings_seen) == 1
    assert "ignored because fps is also set" in warnings_seen[0]


@pytest.mark.parametrize("bad", [0, -2, "soon"])
def test_bad_interval_seconds_rejected(bad: object) -> None:
    with pytest.raises(ValidationError, match="interval_seconds must be"):
        DetectorSpec.model_validate({"type": "motion", "interval_seconds": bad})


@pytest.mark.parametrize("legacy", ["person", "vehicle", "dnn"])
def test_legacy_types_warn_once_naming_onnx(legacy: str, warnings_seen: list[str]) -> None:
    DetectorSpec(type=legacy)
    DetectorSpec(type=legacy)
    assert len(warnings_seen) == 1
    assert legacy in warnings_seen[0]
    assert "type: onnx" in warnings_seen[0]


@pytest.mark.parametrize("current", ["motion", "custom", "onnx"])
def test_current_types_do_not_warn(current: str, warnings_seen: list[str]) -> None:
    DetectorSpec(type=current)
    assert warnings_seen == []


# ---------------------------------------------------------------------------
# CameraConfig: detect_fps, tracking fields, detector rates
# ---------------------------------------------------------------------------

URL = "rtsp://u:p@h/m"


def _cam(**kw: object) -> CameraConfig:
    return CameraConfig(name="yard", main_url=URL, **kw)


def test_camera_detection_defaults() -> None:
    cam = _cam()
    assert cam.detect_fps == 5.0
    assert cam.track_grace_seconds == 3.0
    assert cam.min_track_frames == 2


@pytest.mark.parametrize("ok", [DETECT_FPS_MIN, 2.5, DETECT_FPS_MAX])
def test_detect_fps_accepts_range(ok: float) -> None:
    assert _cam(detect_fps=ok).detect_fps == ok


@pytest.mark.parametrize("bad", [0.4, 0, 30.5])
def test_detect_fps_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValidationError, match="detect_fps must be between 0.5 and 30"):
        _cam(detect_fps=bad)


def test_track_grace_seconds_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="track_grace_seconds must be > 0"):
        _cam(track_grace_seconds=0)


def test_min_track_frames_at_least_one() -> None:
    with pytest.raises(ValidationError, match="min_track_frames must be >= 1"):
        _cam(min_track_frames=0)


def test_detector_fps_equal_to_detect_fps_is_allowed() -> None:
    cam = _cam(detect_fps=5, detectors=[{"type": "onnx", "fps": 5}])
    assert cam.detectors[0].fps == 5.0


def test_detector_fps_above_detect_fps_names_the_index() -> None:
    with pytest.raises(ValidationError, match=r"detectors\[1\] \(onnx\): fps 6 is above"):
        _cam(
            detect_fps=5,
            detectors=[{"type": "motion", "fps": 5}, {"type": "onnx", "fps": 6}],
        )


def test_interval_seconds_above_detect_fps_is_clamped(warnings_seen: list[str]) -> None:
    cam = _cam(detect_fps=5, detectors=[{"type": "motion", "interval_seconds": 0.1}])
    assert cam.detectors[0].fps == 5.0
    assert any("using fps 5" in m for m in warnings_seen)


def test_effective_fps() -> None:
    cam = _cam(detect_fps=4, detectors=[{"type": "motion"}, {"type": "onnx", "fps": 2}])
    assert cam.effective_fps(cam.detectors[0]) == 4.0
    assert cam.effective_fps(cam.detectors[1]) == 2.0
    # An in-memory edit (no validation on assignment) can never exceed the tap rate.
    cam.detectors[1].fps = 9.0
    assert cam.effective_fps(cam.detectors[1]) == 4.0
    assert isinstance(cam.effective_fps(cam.detectors[0]), float)


@pytest.mark.parametrize(
    ("motion_events", "onnx_enabled", "expected"),
    [
        (None, None, True),  # motion only: motion writes events
        (None, True, False),  # enabled onnx present: motion is quiet by default
        (None, False, True),  # disabled onnx does not count
        (True, True, True),  # explicit true wins
        (False, None, False),  # explicit false wins
    ],
)
def test_motion_events_enabled(
    motion_events: bool | None, onnx_enabled: bool | None, expected: bool
) -> None:
    detectors: list[dict] = [{"type": "motion", "events": motion_events}]
    if onnx_enabled is not None:
        detectors.append({"type": "onnx", "enabled": onnx_enabled})
    cam = _cam(detectors=detectors)
    assert cam.motion_events_enabled(cam.detectors[0]) is expected


def test_existing_interval_config_loads_and_warns_once_per_process(
    tmp_path: Path, warnings_seen: list[str]
) -> None:
    """(review focus) interval_seconds: 1.0 loads, runs at 1 fps, warns exactly once."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cameras": [
                    {
                        "name": "front",
                        "main_url": URL,
                        "detectors": [{"type": "motion", "enabled": True, "interval_seconds": 1.0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    first = load_config(path)
    second = load_config(path)
    for cfg in (first, second):
        cam = cfg.cameras[0]
        assert cam.detectors[0].fps == 1.0
        assert cam.effective_fps(cam.detectors[0]) == 1.0
    assert len(warnings_seen) == 1
    assert "interval_seconds" in warnings_seen[0]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_camera_rules_default_empty() -> None:
    assert _cam().rules == []


def test_spec_example_rule_parses() -> None:
    raw = yaml.safe_load(
        """
name: yard
main_url: rtsp://u:p@h/m
detect_fps: 5
detectors:
  - type: motion
    fps: 5
  - type: onnx
    model: yolox-s
    device: auto
    fps: 2
    min_confidence: 0.5
rules:
  - name: person-any-time
    labels: [person]
    zones: []
    min_confidence: 0.6
    between: null
    cooldown_seconds: 60
    clip: true
    actions: [phone]
"""
    )
    cam = CameraConfig.model_validate(raw)
    assert cam.rules[0] == RuleConfig(
        name="person-any-time",
        labels=["person"],
        zones=[],
        min_confidence=0.6,
        between=None,
        cooldown_seconds=60.0,
        clip=True,
        actions=["phone"],
    )


def test_rule_defaults() -> None:
    rule = RuleConfig(name="any", actions=[])
    assert rule.labels == []
    assert rule.zones == []
    assert rule.min_confidence == 0.0
    assert rule.between is None
    assert rule.cooldown_seconds == 60.0
    assert rule.clip is False


def test_rule_requires_actions_key() -> None:
    with pytest.raises(ValidationError, match="actions"):
        RuleConfig.model_validate({"name": "r"})


def test_rule_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError, match="lables"):
        RuleConfig.model_validate({"name": "r", "lables": ["person"], "actions": []})


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_rule_min_confidence_range(bad: float) -> None:
    with pytest.raises(ValidationError, match="min_confidence must be between 0.0 and 1.0"):
        RuleConfig(name="r", min_confidence=bad, actions=[])


def test_rule_negative_cooldown_rejected() -> None:
    with pytest.raises(ValidationError, match="cooldown_seconds must be >= 0"):
        RuleConfig(name="r", cooldown_seconds=-1, actions=[])


@pytest.mark.parametrize("bad", ["", "   ", "x" * 65])
def test_rule_name_validated(bad: str) -> None:
    with pytest.raises(ValidationError, match="rule name"):
        RuleConfig(name=bad, actions=[])


def test_rule_blank_label_rejected() -> None:
    with pytest.raises(ValidationError, match="non-empty names"):
        RuleConfig(name="r", labels=["person", " "], actions=[])


@pytest.mark.parametrize(
    ("given", "stored"),
    [
        ("22:00-06:00", "22:00-06:00"),
        ("7:05 - 9:00", "07:05-09:00"),
        ("", None),
        (None, None),
    ],
)
def test_rule_between_normalised(given: str | None, stored: str | None) -> None:
    assert RuleConfig(name="r", between=given, actions=[]).between == stored


@pytest.mark.parametrize("bad", ["22:00", "25:00-06:00", "22:60-06:00", "22:00-22:00", "ten-six"])
def test_rule_between_rejected(bad: str) -> None:
    with pytest.raises(ValidationError, match="between"):
        RuleConfig(name="r", between=bad, actions=[])


def test_rule_between_unquoted_yaml_number_rejected() -> None:
    # PyYAML reads an unquoted 22:00 as the sexagesimal integer 1320.
    raw = yaml.safe_load("name: r\nbetween: 22:00\nactions: []\n")
    assert raw["between"] == 1320
    with pytest.raises(ValidationError, match="quoted string"):
        RuleConfig.model_validate(raw)


def test_parse_between_returns_times() -> None:
    assert parse_between("22:00-06:00") == (time(22, 0), time(6, 0))
    assert parse_between("00:00-23:59") == (time(0, 0), time(23, 59))


def test_duplicate_rule_names_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate rule name 'night'"):
        _cam(rules=[{"name": "night", "actions": []}, {"name": "night", "actions": []}])


def test_clip_rule_without_recording_warns(warnings_seen: list[str]) -> None:
    _cam(record={"enabled": False}, rules=[{"name": "r", "clip": True, "actions": []}])
    assert len(warnings_seen) == 1
    assert "record.enabled is false" in warnings_seen[0]


def test_clip_rule_with_recording_is_quiet(warnings_seen: list[str]) -> None:
    _cam(rules=[{"name": "r", "clip": True, "actions": []}])
    assert warnings_seen == []


# ---------------------------------------------------------------------------
# Round trips (model_dump -> YAML -> validate), as web write-back tests do
# ---------------------------------------------------------------------------


def _raw_config() -> dict:
    return {
        "cameras": [
            {
                "name": "yard",
                "main_url": URL,
                "detect_fps": 4,
                "track_grace_seconds": 2.5,
                "min_track_frames": 3,
                "detectors": [
                    {"type": "motion", "interval_seconds": 2.0, "events": True},
                    {"type": "person"},
                    {"type": "onnx", "model": "yolox-nano", "device": "cpu", "fps": 2},
                ],
                "rules": [
                    {
                        "name": "night",
                        "labels": ["person"],
                        "between": "22:00-06:00",
                        "actions": ["phone"],
                    }
                ],
            }
        ]
    }


def test_round_trip_keeps_new_fields() -> None:
    cfg = AppConfig.model_validate(_raw_config())
    dumped = cfg.model_dump(mode="json")
    again = AppConfig.model_validate(yaml.safe_load(yaml.safe_dump(dumped, sort_keys=False)))
    assert again.model_dump(mode="json") == dumped
    cam = again.cameras[0]
    assert (cam.detect_fps, cam.track_grace_seconds, cam.min_track_frames) == (4.0, 2.5, 3)
    assert [d.fps for d in cam.detectors] == [0.5, None, 2.0]
    assert cam.detectors[2].model == "yolox-nano"
    assert cam.detectors[2].device == "cpu"
    assert cam.detectors[0].events is True
    assert cam.rules[0].between == "22:00-06:00"


def test_round_trip_dump_has_no_interval_seconds(warnings_seen: list[str]) -> None:
    cfg = AppConfig.model_validate(_raw_config())
    dumped = cfg.model_dump(mode="json")
    assert all("interval_seconds" not in d for d in dumped["cameras"][0]["detectors"])
    warnings_seen.clear()
    deprecations._WARNED.clear()
    AppConfig.model_validate(yaml.safe_load(yaml.safe_dump(dumped, sort_keys=False)))
    # Only the legacy type (still in the dump) can warn again; interval_seconds is gone.
    assert all("interval_seconds" not in m for m in warnings_seen)


# ---------------------------------------------------------------------------
# Frame tap settings
# ---------------------------------------------------------------------------


def _widths(spec: DetectorSpec) -> int | None:
    return {"onnx": 640, "custom": 160}.get(spec.type)


def test_tap_settings_default_motion_only() -> None:
    fps, width = compute_tap_settings(_cam(detectors=[{"type": "motion"}]))
    assert (fps, width) == (5.0, TAP_MIN_WIDTH)
    assert isinstance(fps, float)


def test_tap_settings_use_detect_fps() -> None:
    assert compute_tap_settings(_cam(detect_fps=2.5)) == (2.5, 320)


def test_tap_width_is_largest_enabled_input_width() -> None:
    cam = _cam(detectors=[{"type": "motion"}, {"type": "onnx"}])
    assert compute_tap_settings(cam, _widths) == (5.0, 640)


def test_tap_width_ignores_disabled_detectors() -> None:
    cam = _cam(detectors=[{"type": "motion"}, {"type": "onnx", "enabled": False}])
    assert compute_tap_settings(cam, _widths) == (5.0, 320)


def test_tap_width_never_below_minimum() -> None:
    cam = _cam(detectors=[{"type": "custom", "import_path": "m:C"}])
    assert compute_tap_settings(cam, _widths) == (5.0, 320)


def test_recorder_passes_detect_fps_to_the_ingestor() -> None:
    recorder = CameraRecorder(
        camera=_cam(detect_fps=2.5),
        runtime=RuntimeConfig(),
        frame_tap_dispatcher=FrameTapDispatcher(),
    )
    assert recorder.main is not None
    assert recorder.main.frame_tap_fps == 2.5
    assert recorder.main.frame_tap_scale_width == 320


def test_recorder_uses_explicit_tap_settings() -> None:
    recorder = CameraRecorder(
        camera=_cam(detect_fps=2.5),
        runtime=RuntimeConfig(),
        frame_tap_dispatcher=FrameTapDispatcher(),
        tap_settings=(0.5, 640),
    )
    assert recorder.main is not None
    assert recorder.main.frame_tap_fps == 0.5
    assert recorder.main.frame_tap_scale_width == 640

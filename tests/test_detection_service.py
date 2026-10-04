"""web/services/detection.py: raw-YAML detector patch, label groups, detector rows, test event.

RW-3 Task 15. Pure helpers: no app, no database (the test event gets a fake db),
no network, no model file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from rtsp_warden.actions.rules import RuleEngine
from rtsp_warden.config import AppConfig, CameraConfig, RuleConfig
from rtsp_warden.detectors.registry import DetectorSpec
from rtsp_warden.web.services import detection as svc
from rtsp_warden.web.services.detection import (
    _persist_detector_entry,
    class_groups_for,
    detector_rows,
    detector_summary,
    fire_test_event,
    label_choices,
    runtime_detection_status,
    write_failed_message,
)

RAW_CONFIG = """\
cameras:
  - name: a
    main_url: rtsp://${CAM_USER}:${CAM_PASS}@h/main
    detectors:
      - type: onnx
        model: yolox-s
        fps: 2
        device: cuda
      - type: onnx
        model: yolox-nano
        events: true
        note: keep me
  - name: b
    main_url: rtsp://h2/main
    detectors:
      - type: custom
        import_path: x.y:Z
        config:
          token: ${SECRET_TOKEN}
"""


def _write(tmp_path: Path, text: str = RAW_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _labels_model(models_dir: Path, name: str, labels: list[str]) -> None:
    """A user model descriptor plus labels file (no .onnx file: labels need none)."""
    d = models_dir / name
    d.mkdir(parents=True)
    descriptor = {
        "name": name,
        "file": f"{name}.onnx",
        "labels": "labels.txt",
        "input_size": [64, 64],
        "postprocess": "yolox",
    }
    (d / "model.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
    (d / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def _cam(**fields: Any) -> CameraConfig:
    return CameraConfig(name="yard", main_url="rtsp://u:p@h/m", **fields)


# --- _persist_detector_entry -------------------------------------------------------------


def test_patch_touches_only_the_indexed_entry(tmp_path: Path) -> None:
    path = _write(tmp_path)
    before = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _persist_detector_entry(path, "a", 1, {"enabled": False}, expected_type="onnx") is True
    text = path.read_text(encoding="utf-8")
    assert "${CAM_USER}:${CAM_PASS}" in text
    assert "${SECRET_TOKEN}" in text
    after = yaml.safe_load(text)
    assert after["cameras"][0]["detectors"][0] == before["cameras"][0]["detectors"][0]
    assert after["cameras"][0]["detectors"][1] == {
        "type": "onnx",
        "model": "yolox-nano",
        "events": True,
        "note": "keep me",
        "enabled": False,
    }
    assert after["cameras"][1] == before["cameras"][1]


@pytest.mark.parametrize(
    ("camera", "index", "expected_type"),
    [
        ("a", 0, "motion"),  # the entry is an onnx spec: file changed since load
        ("a", 2, None),  # out of range
        ("a", -1, None),  # negative index
        ("ghost", 0, None),  # camera not in the file
        ("b", 0, "onnx"),  # custom entry, wrong type
    ],
)
def test_patch_refuses_and_writes_nothing(
    tmp_path: Path, camera: str, index: int, expected_type: str | None
) -> None:
    path = _write(tmp_path)
    before = path.read_bytes()
    assert (
        _persist_detector_entry(
            path, camera, index, {"enabled": False}, expected_type=expected_type
        )
        is False
    )
    assert path.read_bytes() == before


def test_patch_matches_a_name_written_with_spaces(tmp_path: Path) -> None:
    path = _write(tmp_path, RAW_CONFIG.replace("- name: a\n", "- name: ' a '\n"))
    assert _persist_detector_entry(path, "a", 0, {"fps": 1.0}) is True
    assert (
        yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"][0]["detectors"][0]["fps"] == 1.0
    )


def test_write_failed_message_names_the_path_and_reason(tmp_path: Path) -> None:
    text = write_failed_message(tmp_path / "config.yaml", PermissionError(13, "Permission denied"))
    assert str(tmp_path / "config.yaml") in text
    assert "Permission denied" in text
    assert "until the next restart" in text


# --- labels for the classes page ---------------------------------------------------------


def test_class_groups_put_unknown_labels_under_other() -> None:
    assert class_groups_for(["person", "raccoon", "car", "cat", "fox", "toaster"]) == [
        ("person", ["person"]),
        ("pet", ["cat"]),
        ("vehicle", ["car"]),
        ("critter", ["raccoon", "fox"]),
        ("other", ["toaster"]),
    ]
    assert class_groups_for([]) == []


def test_class_groups_know_the_wildlife_labels() -> None:
    wildlife = [
        "cat",
        "dog",
        "fox",
        "raccoon",
        "skunk",
        "opossum",
        "squirrel",
        "rabbit",
        "coyote",
        "bobcat",
        "deer",
        "bear",
        "bird",
        "chipmunk",
        "woodchuck",
        "horse",
        "person",
        "vehicle",
    ]
    groups = dict(class_groups_for(wildlife))
    assert groups["pet"] == ["cat", "dog"]
    assert groups["vehicle"] == ["vehicle"]
    assert groups["person"] == ["person"]
    assert "other" not in groups


def test_class_groups_cover_all_80_default_labels(tmp_path: Path) -> None:
    labels = label_choices(_cam(), tmp_path)
    groups = class_groups_for(labels)
    assert [name for name, _ in groups] == ["person", "pet", "vehicle", "critter", "other"]
    assert sum(len(items) for _, items in groups) == 80


def test_label_choices_without_onnx_are_the_default_coco_labels(tmp_path: Path) -> None:
    labels = label_choices(_cam(detectors=[DetectorSpec(type="motion")]), tmp_path)
    assert len(labels) == 80
    assert labels[0] == "person"
    assert "toothbrush" in labels


def test_label_choices_union_of_the_onnx_models_in_first_use_order(tmp_path: Path) -> None:
    _labels_model(tmp_path, "critters", ["raccoon", "person", "fox"])
    _labels_model(tmp_path, "parcels", ["parcel", "person"])
    cam = _cam(
        detectors=[
            DetectorSpec(type="onnx", model="critters"),
            DetectorSpec(type="onnx", model="parcels", enabled=False),
        ]
    )
    assert label_choices(cam, tmp_path) == ["raccoon", "person", "fox", "parcel"]


def test_label_choices_fall_back_to_defaults_when_a_model_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(svc.log, "warning", lambda msg, *args: warnings.append(msg % args))
    cam = _cam(detectors=[DetectorSpec(type="onnx", model="ghost-model")])
    assert len(label_choices(cam, tmp_path)) == 80
    assert len(warnings) == 1 and "ghost-model" in warnings[0]


# --- detector rows -----------------------------------------------------------------------


def _cfg(models_dir: Path) -> AppConfig:
    cam = _cam(
        detect_fps=5.0,
        detectors=[
            DetectorSpec(type="motion", min_area=500),
            DetectorSpec(type="onnx", model="yolox-s", device="cuda", fps=2.0),
            DetectorSpec(type="person", enabled=False),
        ],
    )
    return AppConfig(cameras=[cam], runtime={"models_dir": models_dir})


def _runtime(status: Any) -> SimpleNamespace:
    return SimpleNamespace(detection_status=lambda name: status if name == "yard" else None)


def test_detector_rows_without_a_runtime_show_config_only(tmp_path: Path) -> None:
    rows = detector_rows(_cfg(tmp_path), None, "yard")
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert rows[0]["summary_str"] == "min_area=500"
    assert rows[0]["fps_label"] == "5 (camera)"
    assert rows[0]["motion_events"] is False  # an onnx detector is enabled
    assert (rows[1]["model"], rows[1]["device"], rows[1]["fps_label"]) == ("yolox-s", "cuda", "2")
    assert rows[2]["deprecated"] is True and rows[2]["enabled"] is False
    assert all(r["running"] is False for r in rows)
    assert detector_rows(_cfg(tmp_path), None, "ghost") == []


def test_detector_rows_merge_live_status_by_index(tmp_path: Path) -> None:
    status = {
        "detectors": [
            {"index": 1, "provider": "CUDAExecutionProvider", "processed": 9, "skipped": 18},
            {"index": 2, "provider": "CPUExecutionProvider", "error": "ignored: spec disabled"},
            {"index": 0, "setup_error": "RuntimeError: boom", "errors": "3"},
        ]
    }
    rows = detector_rows(_cfg(tmp_path), _runtime(status), "yard")
    assert rows[0]["running"] is True and rows[0]["error"] == "RuntimeError: boom"
    assert rows[0]["errors"] == 3
    assert rows[1]["provider_label"] == "CUDA"
    assert (rows[1]["processed"], rows[1]["skipped"]) == (9, 18)
    assert rows[2]["running"] is False and rows[2]["provider"] is None


def test_runtime_detection_status_tolerates_fakes_and_failures() -> None:
    assert runtime_detection_status(None, "yard") is None
    assert runtime_detection_status(SimpleNamespace(), "yard") is None
    assert runtime_detection_status(MagicMock(), "yard") is None  # returns a MagicMock, not a dict

    def broken(name: str) -> dict:
        raise RuntimeError("status failed")

    assert runtime_detection_status(SimpleNamespace(detection_status=broken), "yard") is None


# --- fire_test_event ---------------------------------------------------------------------


class FakeDb:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def insert_event(self, **fields: Any) -> int:
        self.calls.append(("insert", fields))
        return 41

    def close_event(self, event_id: int, ended_at: datetime, **fields: Any) -> None:
        self.calls.append(("close", (event_id, ended_at, fields)))


def _rule_cam(tmp_path: Path) -> CameraConfig:
    return _cam(
        record={"output_dir": tmp_path / "rec"},
        rules=[
            RuleConfig(name="person-any-time", labels=["person"], actions=["phone"]),
            RuleConfig(name="cars-only", labels=["car"], actions=["phone"]),
            RuleConfig(name="anything", labels=[], actions=["phone", "hook"]),
        ],
    )


def test_fire_test_event_writes_naive_utc_and_a_placeholder(tmp_path: Path) -> None:
    db = FakeDb()
    now = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
    result = fire_test_event(_rule_cam(tmp_path), dispatch=None, now=now, db=db)

    insert = db.calls[0][1]
    assert insert["camera_name"] == "yard"
    assert insert["event_type"] == "test"
    assert insert["label"] == "person"
    assert insert["zone"] == ""
    assert insert["created_at"] == datetime(2026, 10, 2, 12, 0, 0)  # naive UTC
    assert db.calls[1] == (
        "close",
        (41, datetime(2026, 10, 2, 12, 0, 0), {"thumbnail_path": "yard/thumbnails/41.jpg"}),
    )
    thumb = tmp_path / "rec" / "yard" / "thumbnails" / "41.jpg"
    assert thumb.read_bytes()[:2] == b"\xff\xd8"
    # No runtime: the rules are evaluated locally so the answer explains them, nothing is sent.
    assert [m.rule.name for m in result.decision.matched] == ["person-any-time", "anything"]
    assert "cars-only" in result.decision.reasons
    assert result.queued == []
    assert result.note == "Actions were not sent: the detection runtime is not available."


def test_fire_test_event_hands_the_event_to_dispatch(tmp_path: Path) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    cam = _rule_cam(tmp_path)

    def dispatch(info: Any, **kwargs: Any) -> Any:
        calls.append((info, kwargs))
        return RuleEngine(cam.name, cam.rules).evaluate(info, bypass_cooldown=True)

    result = fire_test_event(cam, dispatch=dispatch, db=FakeDb())
    [(info, kwargs)] = calls
    assert kwargs == {"bypass_cooldown": True, "allow_clip": False}
    assert (info.id, info.camera, info.event_type) == (41, "yard", "test")
    assert info.started_at.tzinfo is not None and info.ended_at == info.started_at
    assert info.thumbnail_path == "yard/thumbnails/41.jpg"
    assert result.queued == ["phone", "hook"]  # each action once, in rule order
    assert result.note == ""


@pytest.mark.parametrize("answer", ["none", "not-a-decision", "raises"])
def test_fire_test_event_falls_back_when_dispatch_gives_no_decision(
    tmp_path: Path, answer: str
) -> None:
    def dispatch(info: Any, **kwargs: Any) -> Any:
        if answer == "raises":
            raise RuntimeError("queue gone")
        return None if answer == "none" else MagicMock()

    result = fire_test_event(_rule_cam(tmp_path), dispatch=dispatch, db=FakeDb())
    assert [m.rule.name for m in result.decision.matched] == ["person-any-time", "anything"]
    assert result.queued == []
    assert "not sent" in result.note


def test_fire_test_event_survives_a_thumbnail_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Any, rel_path: str, frame: Any, bbox: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(svc.EventBuilder, "write_thumbnail", refuse)
    db = FakeDb()
    result = fire_test_event(_rule_cam(tmp_path), dispatch=None, db=db)
    assert result.thumbnail_path is None
    assert db.calls[1][1][2] == {"thumbnail_path": None}


def test_concurrent_toggle_and_field_save_keep_both_changes(tmp_path: Path) -> None:
    """Two write-backs racing on one config.yaml (a detector patch and a camera field save,
    from two threadpool threads) must not drop each other's change: each reads, changes and
    writes the file as one step."""
    import threading

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cameras": [
                    {
                        "name": "a",
                        "main_url": "${CAM_URL}",
                        "detect_fps": 5,
                        "detectors": [{"type": "motion", "fps": 1}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rounds = 50

    def patch_detector() -> None:
        for i in range(1, rounds + 1):
            assert _persist_detector_entry(path, "a", 0, {"fps": float(i)}) is True

    def save_field() -> None:
        for i in range(1, rounds + 1):
            svc._persist_camera_field(path, "a", "track_grace_seconds", float(i))

    threads = [threading.Thread(target=patch_detector), threading.Thread(target=save_field)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    cam = yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"][0]
    assert cam["detectors"][0]["fps"] == float(rounds)
    assert cam["track_grace_seconds"] == float(rounds)
    assert cam["main_url"] == "${CAM_URL}"


# --- RW-5: when ---------------------------------------------------------------------------------


def test_detector_summary_shows_when_unless_always() -> None:
    assert "when=" not in detector_summary(DetectorSpec(type="onnx"))
    assert detector_summary(DetectorSpec(type="onnx", when="night")) == "when=night"
    assert detector_summary(DetectorSpec(type="motion", min_area=500, when="day")) == (
        "min_area=500, when=day"
    )

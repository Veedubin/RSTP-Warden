"""web.services.detection.camera_badge and partials/_detection_badge.html."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rtsp_warden.web.routes._common import templates
from rtsp_warden.web.services.detection import camera_badge

FALLBACK = "CUDA requested but unavailable; running on CPUExecutionProvider"
MODEL_ERROR = "model yolox-s unavailable: download failed (offline)"


def _onnx(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": 1,
        "type": "onnx",
        "model": "yolox-s",
        "device": "auto",
        "provider": "CPUExecutionProvider",
        "fallback_warning": None,
        "fps": 2.0,
        "processed": 10,
        "skipped": 30,
        "errors": 0,
    }
    row.update(overrides)
    return row


def _motion() -> dict[str, Any]:
    return {"index": 0, "type": "motion", "provider": None, "fps": 5.0, "processed": 40}


def _runtime(*rows: dict[str, Any], dropped: int = 0, processed: int = 40) -> SimpleNamespace:
    raw = {
        "frames_processed": processed,
        "frames_dropped": dropped,
        "errors_total": 0,
        "detectors": list(rows),
    }
    return SimpleNamespace(detection_status=lambda name: raw if name == "cam" else None)


@pytest.mark.parametrize(
    ("rows", "text", "level"),
    [
        ((_onnx(provider=None, error=MODEL_ERROR),), "detection error", "error"),
        ((_onnx(device="cuda", fallback_warning=FALLBACK),), "CPU fallback", "warn"),
        ((_onnx(provider="CUDAExecutionProvider"),), "GPU", "ok"),
        ((_motion(), _onnx(provider="CUDAExecutionProvider")), "GPU", "ok"),
        ((_onnx(provider=None),), "model loading", "ok"),
        ((_onnx(),), "CPU", "ok"),
        ((_motion(),), "CPU", "ok"),
    ],
)
def test_camera_badge_text_and_level(rows: tuple, text: str, level: str) -> None:
    badge = camera_badge(_runtime(*rows), "cam")
    assert badge is not None
    assert (badge["text"], badge["level"]) == (text, level)


def test_camera_badge_title_and_counters() -> None:
    rt = _runtime(_onnx(device="cuda", fallback_warning=FALLBACK), dropped=7, processed=40)
    assert camera_badge(rt, "cam") == {
        "text": "CPU fallback",
        "level": "warn",
        "title": f"detector 1 (onnx): {FALLBACK}",
        "dropped": 7,
        "processed": 40,
    }


def test_camera_badge_title_names_the_provider_without_warnings() -> None:
    badge = camera_badge(_runtime(_onnx(provider="CUDAExecutionProvider")), "cam")
    assert badge is not None and badge["title"] == "CUDAExecutionProvider"
    badge = camera_badge(_runtime(_motion()), "cam")
    assert badge is not None and badge["title"] == "OpenCV on CPU"


def test_camera_badge_is_none_without_detection() -> None:
    assert camera_badge(None, "cam") is None
    assert camera_badge(SimpleNamespace(cameras=[]), "cam") is None
    assert camera_badge(_runtime(_motion()), "other") is None


def _render(camera: dict[str, Any]) -> str:
    return templates.env.get_template("partials/_detection_badge.html").render(camera=camera)


def test_badge_partial_renders_nothing_without_detection() -> None:
    assert _render({"name": "cam", "detection": None}).strip() == ""
    assert _render({"name": "cam"}).strip() == ""


def test_badge_partial_shows_warning_and_dropped_frames() -> None:
    detection = {
        "text": "CPU fallback",
        "level": "warn",
        "title": f"detector 1 (onnx): {FALLBACK}",
        "dropped": 7,
        "processed": 40,
    }
    html = _render({"name": "cam", "detection": detection})
    assert 'class="detection-badge detection-badge-warn"' in html
    assert ">CPU fallback</span>" in html
    assert f'title="detector 1 (onnx): {FALLBACK}"' in html
    assert "7 dropped" in html


def test_badge_partial_hides_zero_dropped() -> None:
    detection = {"text": "GPU", "level": "ok", "title": "x", "dropped": 0, "processed": 1}
    html = _render({"name": "cam", "detection": detection})
    assert ">GPU</span>" in html
    assert "dropped" not in html


def test_badge_css_rules_exist() -> None:
    from rtsp_warden.web.paths import STATIC_DIR

    css = (STATIC_DIR / "css" / "warden.css").read_text()
    for rule in (".detection-badge {", ".detection-badge-warn", ".detection-dropped"):
        assert rule in css

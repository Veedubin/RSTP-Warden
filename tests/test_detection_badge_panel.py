"""The detection badge in the Detection panel header (partials/detection_panel.html)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from rtsp_warden.config import AppConfig
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings

FALLBACK = "CUDA requested but unavailable; running on CPUExecutionProvider"


def _status(name: str) -> dict[str, Any] | None:
    """AppRuntime.detection_status("yard") for an onnx detector that fell back to CPU."""
    if name != "yard":
        return None
    return {
        "camera": "yard",
        "enabled": True,
        "frames_processed": 40,
        "frames_dropped": 7,
        "errors_total": 0,
        "tap_fps": 5.0,
        "tap_width": 640,
        "restart_pending": False,
        "detectors": [
            {
                "index": 0,
                "type": "onnx",
                "model": "yolox-s",
                "device": "cuda",
                "provider": "CPUExecutionProvider",
                "fallback_warning": FALLBACK,
                "fps": 2.0,
                "processed": 10,
                "skipped": 30,
                "errors": 0,
                "setup_error": None,
                "error": None,
            }
        ],
    }


def _client(tmp_path: Path, runtime: Any) -> TestClient:
    cfg = AppConfig.model_validate(
        {
            "cameras": [
                {
                    "name": "yard",
                    "main_url": "rtsp://u:p@h/m",
                    "detectors": [{"type": "onnx", "model": "yolox-s", "device": "cuda", "fps": 2}],
                }
            ],
            "runtime": {"models_dir": str(tmp_path / "models")},
        }
    )
    client = TestClient(create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: runtime))
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post(
        "/login", data={"username": "admin", "password": "testpass123", "csrf_token": token}
    )
    return client


def test_panel_header_shows_the_detection_badge(db_with_user: str, tmp_path: Path) -> None:
    r = _client(tmp_path, SimpleNamespace(cameras=[], detection_status=_status)).get(
        "/cameras/yard/detection"
    )
    assert r.status_code == 200
    header = r.text[r.text.index("<header>") : r.text.index("</header>")]
    assert 'class="detection-badge detection-badge-warn"' in header
    assert ">CPU fallback</span>" in header
    assert "7 dropped" in header


def test_panel_without_a_runtime_has_no_badge(db_with_user: str, tmp_path: Path) -> None:
    r = _client(tmp_path, None).get("/cameras/yard/detection")
    assert r.status_code == 200
    assert "detection-badge" not in r.text

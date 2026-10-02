from pathlib import Path

from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.server import WebUIServer


def test_create_app_sets_config_path_and_runtime(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    sentinel = object()
    app = create_app(WebSettings(), cfg=None, config_path=cfg_path, runtime=sentinel)
    assert app.state.config_path == str(cfg_path)
    assert app.state.runtime is sentinel


def test_create_app_defaults_to_none():
    app = create_app(WebSettings())
    assert app.state.config_path is None
    assert app.state.runtime is None


def test_web_server_passes_wiring_through(tmp_path: Path):
    sentinel = object()
    server = WebUIServer(
        WebSettings(host="127.0.0.1", port=8099),
        runtime_provider=lambda: sentinel,
        cfg=None,
        config_path=tmp_path / "c.yaml",
        runtime=sentinel,
    )
    assert server.app.state.runtime is sentinel
    assert server.app.state.config_path == str(tmp_path / "c.yaml")

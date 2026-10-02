from rtsp_warden.cli import _resolve_web_settings
from rtsp_warden.web.config import WebSettings


def test_default_bind_is_loopback(monkeypatch):
    monkeypatch.delenv("WARDEN_WEB_HOST", raising=False)
    monkeypatch.delenv("WARDEN_WEB_PORT", raising=False)
    s = _resolve_web_settings(None, None)
    assert (s.host, s.port) == ("127.0.0.1", 8080)


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("WARDEN_WEB_HOST", "0.0.0.0")
    monkeypatch.setenv("WARDEN_WEB_PORT", "9090")
    s = _resolve_web_settings(None, None)
    assert (s.host, s.port) == ("0.0.0.0", 9090)


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("WARDEN_WEB_HOST", "0.0.0.0")
    s = _resolve_web_settings("10.0.0.2", 8081)
    assert (s.host, s.port) == ("10.0.0.2", 8081)


def test_websettings_default_host(monkeypatch):
    monkeypatch.delenv("WARDEN_WEB_HOST", raising=False)
    assert WebSettings(_env_file=None).host == "127.0.0.1"

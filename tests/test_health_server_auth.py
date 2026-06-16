"""Tests for health_server.py — auth gating."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from rtsp_warden.auth import (
    BEARER_HEADER_PREFIX,
    SESSION_COOKIE_NAME,
    create_api_token,
    create_session,
    hash_password,
)
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import create_admin_user, ensure_schema
from rtsp_warden.health_server import _make_handler


def _make_environ(cookie: str = "", authorization: str = "") -> dict[str, str]:
    return {
        "HTTP_COOKIE": cookie,
        "HTTP_AUTHORIZATION": authorization,
    }


def _handle_request(
    handler_cls: type[BaseHTTPRequestHandler],
    path: str,
    environ: dict[str, str],
) -> tuple[int, dict[str, Any]]:
    """Simulate a GET request to the handler and return (status_code, response_body_dict)."""
    # We need to create a mock request to test the handler.
    # The handler uses self.path, self.headers, etc.
    # We'll create a minimal mock and call do_GET directly.

    class MockRequest:
        def makefile(self, *args: Any, **kwargs: Any) -> Any:
            import io

            return io.BytesIO()

        def close(self) -> None:
            pass

    class MockClientAddress:
        def __init__(self) -> None:
            self.parts = ("127.0.0.1", 0)

    # Create a minimal handler instance
    instance = handler_cls.__new__(handler_cls)
    instance.server = None  # type: ignore[attr-defined]
    instance.path = path
    instance.command = "GET"
    instance.request_version = "HTTP/1.1"
    instance.requestline = f"GET {path} HTTP/1.1"
    instance.headers = type(
        "Headers",
        (),
        {
            "get": lambda self, k, default="": environ.get(
                f"HTTP_{k.upper().replace('-', '_')}", default
            )
        },
    )()
    instance.rfile = type("RFile", (), {"readline": lambda *a: b"", "read": lambda *a: b""})()
    captured_wfile: list[bytes] = []

    class MockWFile:
        def write(self, data: bytes) -> None:
            captured_wfile.append(data)

        def close(self) -> None:
            pass

    instance.wfile = MockWFile()
    instance.close_connection = False

    # Capture the response
    response_status: list[int] = []
    response_headers: list[tuple[str, str]] = []

    def _send_response(code: int, message: str | None = None) -> None:
        response_status.append(code)

    def _send_header(key: str, value: str) -> None:
        response_headers.append((key, value))

    def _end_headers() -> None:
        pass

    instance.send_response = _send_response  # type: ignore[attr-defined]
    instance.send_header = _send_header  # type: ignore[attr-defined]
    instance.end_headers = _end_headers  # type: ignore[attr-defined]

    # Override _send_json to capture body
    captured_body: list[bytes] = []

    def _send_json_override(code: HTTPStatus, payload: Any) -> None:
        response_status.append(int(code))
        body = json.dumps(payload).encode("utf-8")
        captured_body.append(body)

    instance._send_json = _send_json_override  # type: ignore[attr-defined]

    # Also override _send_text
    def _send_text_override(code: HTTPStatus, body: str, content_type: str) -> None:
        response_status.append(int(code))
        captured_body.append(body.encode("utf-8"))

    instance._send_text = _send_text_override  # type: ignore[attr-defined]

    # Call do_GET
    instance.do_GET()  # type: ignore[attr-defined]

    status = response_status[0] if response_status else 500
    # Try captured_body first (_send_json/_send_text), then fall back to wfile writes
    if captured_body:
        body_dict = json.loads(captured_body[0].decode("utf-8"))
    elif captured_wfile:
        # The 401 path writes directly to wfile
        body_dict = json.loads(captured_wfile[0].decode("utf-8"))
    else:
        body_dict = {}
    return status, body_dict


@pytest.fixture
def setup_auth_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Set up a clean DB with an admin user and return session/bearer tokens."""
    db_url = f"sqlite:///{tmp_path}/health_test.db"
    monkeypatch.setenv("WARDEN_DB_URL", db_url)
    reset_engine()
    ensure_schema()

    pw_hash = hash_password("testpass123")
    user = create_admin_user("health_admin", pw_hash)

    session_obj = create_session(user)
    token_obj = create_api_token(user, name="health-test")

    return {
        "user": user,
        "session_token": session_obj.token,
        "bearer_raw": token_obj.raw,
    }


def test_health_server_auth_disabled_serves_all(setup_auth_db: dict[str, Any]) -> None:
    """auth_enabled=False (default), all endpoints serve 200."""
    handler_cls = _make_handler(lambda: {"ok": True}, auth_enabled=False)
    status, body = _handle_request(handler_cls, "/healthz", _make_environ())
    assert status == 200
    assert body.get("ok") is True


def test_health_server_auth_enabled_unauth_returns_401(setup_auth_db: dict[str, Any]) -> None:
    """auth_enabled=True, no auth → 401."""
    handler_cls = _make_handler(lambda: {"ok": True}, auth_enabled=True)
    status, body = _handle_request(handler_cls, "/status.json", _make_environ())
    assert status == 401


def test_health_server_auth_enabled_session_serves_200(setup_auth_db: dict[str, Any]) -> None:
    """auth_enabled=True + valid session cookie → 200."""
    handler_cls = _make_handler(lambda: {"ok": True}, auth_enabled=True)
    environ = _make_environ(cookie=f"{SESSION_COOKIE_NAME}={setup_auth_db['session_token']}")
    status, body = _handle_request(handler_cls, "/status.json", environ)
    assert status == 200


def test_health_server_auth_enabled_bearer_serves_200(setup_auth_db: dict[str, Any]) -> None:
    """auth_enabled=True + valid bearer → 200."""
    handler_cls = _make_handler(lambda: {"ok": True}, auth_enabled=True)
    environ = _make_environ(authorization=f"{BEARER_HEADER_PREFIX}{setup_auth_db['bearer_raw']}")
    status, body = _handle_request(handler_cls, "/status.json", environ)
    assert status == 200


def test_health_server_401_has_www_authenticate(setup_auth_db: dict[str, Any]) -> None:
    """Response includes WWW-Authenticate: Bearer realm='warden'."""
    handler_cls = _make_handler(lambda: {"ok": True}, auth_enabled=True)
    status, body = _handle_request(handler_cls, "/status.json", _make_environ())
    assert status == 401
    assert body.get("ok") is False
    assert "error" in body


def test_health_server_healthz_open_exempts_auth(
    setup_auth_db: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """auth_enabled=True + WARDEN_AUTH_HEALTHZ_OPEN=true env → /healthz returns 200 without auth."""
    monkeypatch.setenv("WARDEN_AUTH_HEALTHZ_OPEN", "true")
    handler_cls = _make_handler(lambda: {"ok": True}, auth_enabled=True)
    status, body = _handle_request(handler_cls, "/healthz", _make_environ())
    assert status == 200


def test_health_server_invalid_bearer_prefix_returns_401(setup_auth_db: dict[str, Any]) -> None:
    """Authorization: Bearer foo (no Warden- prefix) → 401."""
    handler_cls = _make_handler(lambda: {"ok": True}, auth_enabled=True)
    environ = _make_environ(authorization="Bearer some_token")
    status, body = _handle_request(handler_cls, "/status.json", environ)
    assert status == 401

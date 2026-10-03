"""Synchronous actions: ntfy, webhook, Apprise, the factory, and error redaction."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import apprise
import cv2
import httpx
import numpy as np
import pytest

from rtsp_warden.actions import (
    Action,
    ActionPayload,
    ActionResult,
    AppriseAction,
    NtfyAction,
    WebhookAction,
    build_action,
    build_actions,
    placeholder_jpeg,
    synthetic_payload,
)
from rtsp_warden.config import (
    AppConfig,
    AppriseActionSpec,
    NtfyActionSpec,
    WebhookActionSpec,
)


def _payload(camera: str = "yard", label: str = "person", zone: str = "") -> ActionPayload:
    return ActionPayload(
        camera=camera,
        label=label,
        confidence=0.87,
        zone=zone,
        started_at="2026-10-02T21:30:05+00:00",
        ended_at=None,
        thumbnail_url="http://warden.lan:8080/events/42/thumbnail.jpg",
        clip_url=None,
        event_url="http://warden.lan:8080/events/42",
    )


def _jpeg_file(tmp_path: Path, name: str = "42.jpg") -> Path:
    path = tmp_path / name
    path.write_bytes(cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))[1].tobytes())
    return path


def _capture(
    captures: list[dict[str, Any]],
    status: int = 200,
    body: dict[str, Any] | None = None,
) -> Callable[[], httpx.Client]:
    """Client factory whose transport records every request (body kept as bytes)."""

    def handler(request: httpx.Request) -> httpx.Response:
        captures.append(
            {
                "method": request.method,
                "url": request.url,
                "path": request.url.path,
                "params": dict(request.url.params),
                "headers": request.headers,
                "content": request.content,
            }
        )
        return httpx.Response(status, json=body if body is not None else {"id": "x"})

    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


def _raising(exc: Exception) -> Callable[[], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


def _replying(status: int, text: str) -> Callable[[], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=text)

    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


def _ntfy(**overrides: Any) -> NtfyActionSpec:
    data: dict[str, Any] = {
        "name": "phone",
        "type": "ntfy",
        "url": "https://ntfy.example/",
        "topic": "warden-home",
        "token": "tk_test_value",
    }
    data.update(overrides)
    return NtfyActionSpec.model_validate(data)


# --- payload -----------------------------------------------------------------


def test_payload_as_dict_has_the_spec_keys() -> None:
    d = _payload().as_dict()
    assert list(d) == [
        "camera",
        "label",
        "confidence",
        "zone",
        "started_at",
        "ended_at",
        "thumbnail_url",
        "clip_url",
        "event_url",
        "test",
    ]
    assert d["test"] is False
    json.dumps(d)  # plain JSON types only


def test_synthetic_payload_is_marked_test() -> None:
    p = synthetic_payload(base_url="http://warden.lan:8080/")
    assert p.test is True
    assert p.label == "person"
    assert p.event_url == "http://warden.lan:8080/events"


def test_placeholder_jpeg_is_a_jpeg() -> None:
    data = placeholder_jpeg()
    assert data[:3] == b"\xff\xd8\xff"
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert frame.shape == (180, 320, 3)


# --- ntfy --------------------------------------------------------------------


def test_ntfy_uploads_thumbnail_with_text_in_query(tmp_path: Path) -> None:
    thumb = _jpeg_file(tmp_path)
    captures: list[dict[str, Any]] = []
    action = NtfyAction(_ntfy(priority=4), client_factory=_capture(captures))

    result = action.send(_payload(camera="Cour arrière", zone="Allée"), attachment=thumb)

    assert result == ActionResult(ok=True)
    (req,) = captures
    assert req["method"] == "POST"
    assert req["path"] == "/warden-home"
    assert req["content"] == thumb.read_bytes()
    assert req["params"]["filename"] == "42.jpg"
    assert req["params"]["title"] == "person on Cour arrière"
    assert req["params"]["message"].startswith("person 87% in Allée on Cour arrière")
    assert req["params"]["message"].endswith("http://warden.lan:8080/events/42")
    assert req["params"]["priority"] == "4"
    assert req["headers"]["authorization"] == "Bearer tk_test_value"
    assert "title" not in req["headers"]


def test_ntfy_without_attachment_sends_text_body(tmp_path: Path) -> None:
    captures: list[dict[str, Any]] = []
    action = NtfyAction(_ntfy(token=None), client_factory=_capture(captures))

    missing = tmp_path / "expired.jpg"
    result = action.send(_payload(), attachment=missing)

    assert result.ok is True
    (req,) = captures
    assert "filename" not in req["params"]
    assert "priority" not in req["params"]
    assert req["content"].decode("utf-8") == req["params"]["message"]
    assert "authorization" not in req["headers"]


def test_ntfy_without_topic_posts_to_url(tmp_path: Path) -> None:
    captures: list[dict[str, Any]] = []
    spec = _ntfy(url="https://ntfy.example/warden-home", topic=None)
    NtfyAction(spec, client_factory=_capture(captures)).send(_payload())
    assert captures[0]["path"] == "/warden-home"


def test_ntfy_http_error_text_has_status_and_error_field_only() -> None:
    captures: list[dict[str, Any]] = []
    factory = _capture(captures, status=403, body={"code": 40301, "error": "forbidden"})
    result = NtfyAction(_ntfy(), client_factory=factory).send(_payload())
    assert result == ActionResult(ok=False, error="HTTP 403: forbidden")


def test_ntfy_http_error_without_json_body() -> None:
    factory = _replying(502, "<html>bad gateway</html>")
    result = NtfyAction(_ntfy(), client_factory=factory).send(_payload())
    assert result == ActionResult(ok=False, error="HTTP 502")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ConnectError("refused: https://ntfy.example/warden-home"), "connection failed"),
        (httpx.ReadTimeout("timed out reading https://ntfy.example/warden-home"), "timed out"),
        (RuntimeError("boom tk_test_value"), "unexpected error (RuntimeError)"),
    ],
)
def test_ntfy_exceptions_never_leak_url_topic_or_token(exc: Exception, expected: str) -> None:
    result = NtfyAction(_ntfy(), client_factory=_raising(exc)).send(_payload())
    assert result.ok is False
    assert expected in (result.error or "")
    for secret in ("ntfy.example", "warden-home", "tk_test_value"):
        assert secret not in (result.error or "")


def test_ntfy_creates_a_client_per_send() -> None:
    made: list[httpx.Client] = []
    captures: list[dict[str, Any]] = []
    inner = _capture(captures)

    def factory() -> httpx.Client:
        client = inner()
        made.append(client)
        return client

    action = NtfyAction(_ntfy(), client_factory=factory)
    action.send(_payload())
    action.send(_payload())
    assert len(made) == 2
    assert all(c.is_closed for c in made)


def test_ntfy_test_sends_placeholder_jpeg() -> None:
    captures: list[dict[str, Any]] = []
    result = NtfyAction(_ntfy(), client_factory=_capture(captures)).test()
    assert result.ok is True
    (req,) = captures
    assert req["content"][:3] == b"\xff\xd8\xff"
    assert req["params"]["filename"] == "test.jpg"
    assert req["params"]["title"] == "[test] person on test-camera"


# --- webhook -----------------------------------------------------------------


def test_webhook_posts_payload_as_json() -> None:
    captures: list[dict[str, Any]] = []
    spec = WebhookActionSpec(
        name="ha",
        type="webhook",
        url="http://ha.local:8123/api/webhook/warden",
        headers={"X-Api-Key": "k_test"},
    )
    payload = _payload()
    result = WebhookAction(spec, client_factory=_capture(captures)).send(payload)

    assert result.ok is True
    (req,) = captures
    assert req["method"] == "POST"
    assert str(req["url"]) == "http://ha.local:8123/api/webhook/warden"
    assert req["headers"]["content-type"] == "application/json"
    assert req["headers"]["x-api-key"] == "k_test"
    assert json.loads(req["content"]) == payload.as_dict()


def test_webhook_put_and_error_text() -> None:
    captures: list[dict[str, Any]] = []
    spec = WebhookActionSpec(name="ha", type="webhook", url="http://ha.local/x", method="PUT")
    factory = _capture(captures, status=500, body={"error": "Internal  server\nerror"})
    result = WebhookAction(spec, client_factory=factory).send(_payload())
    assert captures[0]["method"] == "PUT"
    assert result == ActionResult(ok=False, error="HTTP 500: Internal server error")


def test_webhook_test_sends_a_test_payload() -> None:
    captures: list[dict[str, Any]] = []
    spec = WebhookActionSpec(name="ha", type="webhook", url="http://ha.local/x")
    assert WebhookAction(spec, client_factory=_capture(captures)).test().ok is True
    body = json.loads(captures[0]["content"])
    assert body["test"] is True
    assert body["label"] == "person"


# --- apprise -----------------------------------------------------------------


def _fake_apprise(
    monkeypatch: pytest.MonkeyPatch,
    *,
    add_ok: bool = True,
    notify_result: bool | None = True,
    notify_exc: Exception | None = None,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"urls": [], "notify": []}

    class FakeApprise:
        def add(self, url: str) -> bool:
            calls["urls"].append(url)
            return add_ok

        def notify(self, **kwargs: Any) -> bool | None:
            if notify_exc is not None:
                raise notify_exc
            calls["notify"].append(kwargs)
            return notify_result

    monkeypatch.setattr(apprise, "Apprise", FakeApprise)
    return calls


def _apprise_spec() -> AppriseActionSpec:
    return AppriseActionSpec(
        name="mail",
        type="apprise",
        urls=["mailto://user:pw_test@smtp.example.com", "tgram://bot_test/123"],
    )


def test_apprise_attaches_thumbnail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _fake_apprise(monkeypatch)
    thumb = _jpeg_file(tmp_path)

    result = AppriseAction(_apprise_spec()).send(_payload(camera="Cour arrière"), attachment=thumb)

    assert result == ActionResult(ok=True)
    assert calls["urls"] == ["mailto://user:pw_test@smtp.example.com", "tgram://bot_test/123"]
    (kwargs,) = calls["notify"]
    assert kwargs["attach"] == str(thumb)
    assert kwargs["title"] == "person on Cour arrière"
    assert kwargs["body"].startswith("person 87% on Cour arrière")


def test_apprise_missing_thumbnail_sends_text_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_apprise(monkeypatch)
    AppriseAction(_apprise_spec()).send(_payload(), attachment=tmp_path / "gone.jpg")
    assert calls["notify"][0]["attach"] is None


def test_apprise_bad_url_error_does_not_echo_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_apprise(monkeypatch, add_ok=False)
    result = AppriseAction(_apprise_spec()).send(_payload())
    assert result.ok is False
    assert "#1" in (result.error or "")
    assert "pw_test" not in (result.error or "")
    assert "mailto" not in (result.error or "")


@pytest.mark.parametrize("notify_result", [False, None])
def test_apprise_failed_delivery(monkeypatch: pytest.MonkeyPatch, notify_result: Any) -> None:
    _fake_apprise(monkeypatch, notify_result=notify_result)
    result = AppriseAction(_apprise_spec()).send(_payload())
    assert result == ActionResult(ok=False, error="Apprise reported a failed delivery")


def test_apprise_exception_is_a_failed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_apprise(monkeypatch, notify_exc=RuntimeError("smtp says no to user:pw_test"))
    result = AppriseAction(_apprise_spec()).send(_payload())
    assert result == ActionResult(ok=False, error="unexpected error (RuntimeError)")


def test_apprise_test_attaches_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_apprise(monkeypatch)
    assert AppriseAction(_apprise_spec()).test().ok is True
    (kwargs,) = calls["notify"]
    assert kwargs["attach"].endswith("test.jpg")
    assert kwargs["title"].startswith("[test] ")


# --- factory and package -----------------------------------------------------


def test_build_action_picks_the_class_and_passes_the_client_factory() -> None:
    captures: list[dict[str, Any]] = []
    ntfy = build_action(_ntfy(), client_factory=_capture(captures))
    hook = build_action(
        WebhookActionSpec(name="ha", type="webhook", url="http://ha.local/x"),
        client_factory=_capture(captures),
    )
    mail = build_action(_apprise_spec())
    assert isinstance(ntfy, NtfyAction)
    assert isinstance(hook, WebhookAction)
    assert isinstance(mail, AppriseAction)
    for action in (ntfy, hook, mail):
        assert isinstance(action, Action)
    ntfy.send(_payload())
    hook.send(_payload())
    assert [c["path"] for c in captures] == ["/warden-home", "/x"]


def test_build_action_rejects_unknown_spec() -> None:
    with pytest.raises(ValueError, match="unknown action type"):
        build_action(object())  # type: ignore[arg-type]


def test_build_actions_keys_by_name_in_config_order() -> None:
    cfg = AppConfig.model_validate(
        {
            "cameras": [],
            "actions": [
                {"name": "phone", "type": "ntfy", "url": "https://ntfy.example", "topic": "t"},
                {"name": "ha", "type": "webhook", "url": "http://ha.local/x"},
                {"name": "mail", "type": "apprise", "urls": ["mailto://u:p@h"]},
            ],
        }
    )
    actions = build_actions(cfg)
    assert list(actions) == ["phone", "ha", "mail"]
    assert [a.type for a in actions.values()] == ["ntfy", "webhook", "apprise"]


def test_httpx_request_logging_is_quiet_after_import() -> None:
    """httpx logs every request URL at INFO; an ntfy topic is a password on ntfy.sh. Importing
    the actions package raises the httpx and httpcore loggers to WARNING, so a send writes
    no record that contains the topic."""
    import logging

    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.DEBUG)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.addHandler(handler)
    root = logging.getLogger()
    root_level = root.level
    root.setLevel(logging.INFO)  # what `serve` (verbosity info) configures
    captures: list[dict[str, Any]] = []
    try:
        result = NtfyAction(_ntfy(), client_factory=_capture(captures)).send(_payload())
    finally:
        root.setLevel(root_level)
        httpx_logger.removeHandler(handler)

    assert result.ok
    assert captures and captures[0]["path"] == "/warden-home"  # the topic was in the URL
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
    assert not any("warden-home" in record.getMessage() for record in records)


def test_alerts_package_and_alert_manager_are_gone() -> None:
    import importlib.util

    from rtsp_warden.web.app import create_app
    from rtsp_warden.web.config import WebSettings

    assert importlib.util.find_spec("rtsp_warden.alerts") is None
    app = create_app(WebSettings(), cfg=AppConfig(cameras=[]))
    assert not hasattr(app.state, "alert_manager")
    paths = [getattr(route, "path", "") for route in app.routes]
    assert not [p for p in paths if p.startswith("/alerts")]

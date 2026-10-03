"""Tests for the Actions page (GET /actions) and its per-action Test button.

Actions are defined in config.yaml; the page lists them with their run history from
``action_runs`` and offers a Test button that calls the real action class's ``test()``
(faked here by patching ``build_action`` in the route module). A test writes no
``action_runs`` row (ruling R13) and never shows a URL, topic or token.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rtsp_warden.actions.base import ActionResult
from rtsp_warden.auth import hash_password
from rtsp_warden.config import AppConfig
from rtsp_warden.db.schema import action_stats, create_user, insert_action_run, insert_event
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.routes import actions as actions_routes

NTFY_URL = "https://ntfy.example"
NTFY_TOPIC = "warden-home-topic"
NTFY_TOKEN = "tk_not_a_real_token"
HOOK_URL = "https://hooks.example/warden"
SECRETS = ("ntfy.example", NTFY_TOPIC, NTFY_TOKEN, "hooks.example")


def _cfg(with_actions: bool = True) -> AppConfig:
    raw: dict[str, Any] = {"cameras": [{"name": "yard", "main_url": "rtsp://u:p@h/m"}]}
    if with_actions:
        raw["cameras"][0]["rules"] = [{"name": "person-any-time", "actions": ["phone"]}]
        raw["actions"] = [
            {
                "name": "phone",
                "type": "ntfy",
                "url": NTFY_URL,
                "topic": NTFY_TOPIC,
                "token": NTFY_TOKEN,
            },
            {"name": "hook", "type": "webhook", "url": HOOK_URL},
        ]
    return AppConfig.model_validate(raw)


def _login(client: TestClient, username: str, password: str) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303


def _csrf(client: TestClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies.get("warden_csrf", "")}


def _row(html: str, name: str) -> str:
    """Return the inner HTML of the table row for one action."""
    m = re.search(rf'<tr[^>]*data-action="{re.escape(name)}"[^>]*>(.*?)</tr>', html, re.S)
    assert m is not None, f"no row for action {name!r}"
    return m.group(1)


class FakeAction:
    """Stands in for the real action class built by ``build_action``."""

    def __init__(self, name: str, type_: str) -> None:
        self.name = name
        self.type = type_
        self.result = ActionResult(ok=True)
        self.exc: Exception | None = None
        self.calls = 0
        self.saw_running_loop: bool | None = None

    def send(self, payload: Any, attachment: Any = None) -> ActionResult:
        raise AssertionError("the Test button must call test(), not send()")

    def test(self) -> ActionResult:
        self.calls += 1
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.saw_running_loop = False
        else:
            self.saw_running_loop = True
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.fixture
def app(db_with_user: str):
    return create_app(WebSettings(), cfg=_cfg(), runtime_provider=lambda: None)


@pytest.fixture
def admin_client(app) -> TestClient:
    client = TestClient(app)
    _login(client, "admin", "testpass123")
    return client


@pytest.fixture
def viewer_client(app) -> TestClient:
    create_user("viewer", hash_password("viewerpass123"), is_admin=False)
    client = TestClient(app)
    _login(client, "viewer", "viewerpass123")
    return client


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeAction]:
    built = {"phone": FakeAction("phone", "ntfy"), "hook": FakeAction("hook", "webhook")}

    def fake_build_action(spec: Any, **kwargs: Any) -> FakeAction:
        return built[spec.name]

    monkeypatch.setattr(actions_routes, "build_action", fake_build_action)
    return built


# ---------------------------------------------------------------------------
# action_rows (pure)
# ---------------------------------------------------------------------------


def test_action_rows_follow_config_order_with_defaults() -> None:
    rows = actions_routes.action_rows(_cfg(), {})
    assert [r["name"] for r in rows] == ["phone", "hook"]
    assert [r["type"] for r in rows] == ["ntfy", "webhook"]
    assert rows[0]["used_by"] == ["yard/person-any-time"]
    assert rows[1]["used_by"] == []
    for r in rows:
        assert r["last_run"] is None
        assert r["last_run_iso"] is None
        assert r["last_status"] is None
        assert r["failures"] == 0


def test_action_rows_treat_naive_last_run_as_utc() -> None:
    stats = {
        "phone": {"last_run": datetime(2026, 10, 2, 9, 30), "last_status": "ok", "failures": 0}
    }
    row = actions_routes.action_rows(_cfg(), stats)[0]
    expected_local = datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc).astimezone()
    assert row["last_run_iso"] == "2026-10-02T09:30:00+00:00"
    assert row["last_run"] == expected_local.strftime("%Y-%m-%d %H:%M")
    assert row["last_status"] == "ok"


def test_action_rows_cast_failures_to_int() -> None:
    stats = {"hook": {"last_run": None, "last_status": None, "failures": None}}
    rows = actions_routes.action_rows(_cfg(), stats)
    assert rows[1]["failures"] == 0
    assert type(rows[1]["failures"]) is int


def test_action_rows_never_carry_destinations() -> None:
    rows = actions_routes.action_rows(_cfg(), {})
    assert set(rows[0]) == {
        "name",
        "type",
        "used_by",
        "last_run",
        "last_run_iso",
        "last_status",
        "failures",
    }
    flat = repr(rows)
    for secret in SECRETS:
        assert secret not in flat


# ---------------------------------------------------------------------------
# GET /actions
# ---------------------------------------------------------------------------


def test_actions_page_lists_configured_actions(admin_client: TestClient) -> None:
    r = admin_client.get("/actions")
    assert r.status_code == 200
    phone = _row(r.text, "phone")
    hook = _row(r.text, "hook")
    assert "ntfy" in phone
    assert "webhook" in hook
    assert "yard/person-any-time" in phone
    assert "never" in phone
    assert 'data-failures="0"' in phone
    assert 'hx-post="/actions/phone/test"' in phone
    assert 'hx-post="/actions/hook/test"' in hook
    # The page says where actions are edited.
    assert "config.yaml" in r.text


def test_actions_page_never_shows_urls_topics_or_tokens(admin_client: TestClient) -> None:
    r = admin_client.get("/actions")
    assert r.status_code == 200
    for secret in SECRETS:
        assert secret not in r.text


def test_actions_page_shows_run_history_from_action_runs(admin_client: TestClient) -> None:
    event_id = insert_event(
        camera_name="yard",
        event_type="person",
        label="person",
        confidence=0.91,
        zone="",
        track_id=1,
        message="person detected on yard",
        created_at=datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc),
    )
    insert_action_run(event_id=event_id, action_name="phone", status="ok", error=None)
    insert_action_run(event_id=event_id, action_name="phone", status="failed", error="HTTP 500")
    insert_action_run(event_id=event_id, action_name="phone", status="failed", error="HTTP 500")
    assert action_stats()["phone"]["failures"] == 2  # the seed reached the table the page reads

    r = admin_client.get("/actions")
    assert r.status_code == 200
    phone = _row(r.text, "phone")
    hook = _row(r.text, "hook")
    assert 'data-failures="2"' in phone
    assert "never" not in phone
    assert "<time datetime=" in phone
    assert 'data-failures="0"' in hook
    assert "never" in hook


def test_actions_page_formats_last_run_and_status(
    admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    last_run = datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        actions_routes,
        "action_stats",
        lambda: {"phone": {"last_run": last_run, "last_status": "failed", "failures": 3}},
    )
    r = admin_client.get("/actions")
    assert r.status_code == 200
    phone = _row(r.text, "phone")
    assert 'datetime="2026-10-02T09:30:00+00:00"' in phone
    assert last_run.astimezone().strftime("%Y-%m-%d %H:%M") in phone
    assert "<mark>failed</mark>" in phone
    assert 'data-failures="3"' in phone


def test_actions_page_survives_a_stats_error(
    admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> dict:
        raise RuntimeError("database is locked")

    warnings: list[tuple] = []
    monkeypatch.setattr(actions_routes, "action_stats", broken)
    monkeypatch.setattr(actions_routes.log, "warning", lambda *a, **k: warnings.append(a))

    r = admin_client.get("/actions")
    assert r.status_code == 200
    assert "Run history is unavailable" in r.text
    assert 'data-action="phone"' in r.text
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0]


def test_actions_page_empty_state(db_with_user: str) -> None:
    app = create_app(WebSettings(), cfg=_cfg(with_actions=False), runtime_provider=lambda: None)
    client = TestClient(app)
    _login(client, "admin", "testpass123")
    r = client.get("/actions")
    assert r.status_code == 200
    assert "No actions are configured" in r.text
    assert "data-action=" not in r.text


def test_nav_links_to_actions_not_alerts(admin_client: TestClient) -> None:
    r = admin_client.get("/actions")
    assert r.status_code == 200
    assert '<a href="/actions">Actions</a>' in r.text
    assert 'href="/alerts"' not in r.text


def test_actions_page_requires_admin(viewer_client: TestClient) -> None:
    assert viewer_client.get("/actions").status_code == 403


def test_actions_page_redirects_anonymous_browser_to_login(app) -> None:
    client = TestClient(app)
    r = client.get("/actions", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=/actions"


# ---------------------------------------------------------------------------
# POST /actions/{name}/test
# ---------------------------------------------------------------------------


def test_test_button_ok_returns_fragment(
    admin_client: TestClient, fakes: dict[str, FakeAction]
) -> None:
    r = admin_client.post("/actions/phone/test", headers=_csrf(admin_client))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'data-test-status="ok"' in r.text
    assert "<html" not in r.text
    assert fakes["phone"].calls == 1
    assert fakes["hook"].calls == 0


def test_test_button_failed_shows_error_without_secrets(
    admin_client: TestClient, fakes: dict[str, FakeAction]
) -> None:
    fakes["phone"].result = ActionResult(ok=False, error="HTTP 401 (unauthorized)")
    r = admin_client.post("/actions/phone/test", headers=_csrf(admin_client))
    assert r.status_code == 200
    assert 'data-test-status="failed"' in r.text
    assert "HTTP 401 (unauthorized)" in r.text
    for secret in SECRETS:
        assert secret not in r.text


def test_test_button_reports_an_exception_by_class_only(
    admin_client: TestClient, fakes: dict[str, FakeAction], monkeypatch: pytest.MonkeyPatch
) -> None:
    """str(exc) can hold the URL and topic (R17); neither the page nor the log shows it."""
    fakes["phone"].exc = RuntimeError(f"POST {NTFY_URL}/{NTFY_TOPIC} failed")
    warnings: list[tuple] = []
    monkeypatch.setattr(actions_routes.log, "warning", lambda *a, **k: warnings.append(a))

    r = admin_client.post("/actions/phone/test", headers=_csrf(admin_client))
    assert r.status_code == 200
    assert 'data-test-status="failed"' in r.text
    assert "RuntimeError" in r.text
    assert len(warnings) == 1
    logged = repr(warnings[0])
    for secret in SECRETS:
        assert secret not in r.text
        assert secret not in logged


def test_test_runs_off_the_event_loop(
    admin_client: TestClient, fakes: dict[str, FakeAction]
) -> None:
    r = admin_client.post("/actions/phone/test", headers=_csrf(admin_client))
    assert r.status_code == 200
    assert fakes["phone"].saw_running_loop is False


def test_test_writes_no_action_runs_row(
    admin_client: TestClient, fakes: dict[str, FakeAction]
) -> None:
    fakes["phone"].result = ActionResult(ok=False, error="HTTP 500")
    r = admin_client.post("/actions/phone/test", headers=_csrf(admin_client))
    assert r.status_code == 200
    assert action_stats() == {}


def test_test_unknown_action_is_404(admin_client: TestClient, fakes: dict[str, FakeAction]) -> None:
    r = admin_client.post("/actions/nope/test", headers=_csrf(admin_client))
    assert r.status_code == 404
    assert fakes["phone"].calls == 0
    assert fakes["hook"].calls == 0


def test_test_is_post_only(admin_client: TestClient, fakes: dict[str, FakeAction]) -> None:
    assert admin_client.get("/actions/phone/test").status_code == 405
    assert fakes["phone"].calls == 0


def test_test_requires_csrf(admin_client: TestClient, fakes: dict[str, FakeAction]) -> None:
    r = admin_client.post("/actions/phone/test")
    assert r.status_code == 403
    assert fakes["phone"].calls == 0


def test_test_requires_admin(viewer_client: TestClient, fakes: dict[str, FakeAction]) -> None:
    r = viewer_client.post("/actions/phone/test", headers=_csrf(viewer_client))
    assert r.status_code == 403
    assert fakes["phone"].calls == 0

"""htmx forms on the users and API-token pages swap fragments, not whole pages.

Toggle-admin and delete on /users, and create and revoke on /api-tokens, are
htmx forms: with ``HX-Request: true`` the routes return only the partial the
form targets. Without htmx they keep their old status codes (303 for user
toggle/delete, 200 full page for tokens). The new-user and reset-password pages
are plain forms because their response is a whole page.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from rtsp_warden.auth import create_api_token, hash_password, list_api_tokens
from rtsp_warden.db import create_user, get_user_by_id
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.rate_limit import get_login_limiter

HX = {"HX-Request": "true"}
_FORM_RE = re.compile(r"<form\b.*?</form>", re.DOTALL)


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303


def _post(client: TestClient, url: str, data: dict | None = None, *, htmx: bool = False):
    """POST a form with the hidden csrf_token field, optionally as htmx."""
    form = dict(data or {})
    form["csrf_token"] = client.cookies.get("warden_csrf", "")
    headers = dict(HX) if htmx else {}
    return client.post(url, data=form, headers=headers, follow_redirects=False)


def _assert_fragment(body: str, root_id: str) -> None:
    """The body is one partial rooted at ``<div id=root_id>``, not a page."""
    assert body.lstrip().startswith(f'<div id="{root_id}"')
    assert "<html" not in body
    assert "<nav" not in body
    assert "<footer" not in body


def _form_with_action(html: str, action: str) -> str:
    for form in _FORM_RE.findall(html):
        if f'action="{action}"' in form:
            return form
    raise AssertionError(f"no form with action={action!r}")


@pytest.fixture
def client(db_with_user: str, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("WARDEN_AUTH_ENABLED", "true")
    get_login_limiter()._attempts.clear()
    c = TestClient(create_app(WebSettings()))
    _login(c)
    return c


def test_is_htmx_reads_the_hx_request_header() -> None:
    from rtsp_warden.web.routes._common import is_htmx

    def req(headers: list[tuple[bytes, bytes]]) -> Request:
        return Request({"type": "http", "method": "POST", "path": "/", "headers": headers})

    assert is_htmx(req([(b"hx-request", b"true")])) is True
    assert is_htmx(req([])) is False
    assert is_htmx(req([(b"hx-request", b"false")])) is False


class TestUsersTable:
    def test_users_page_forms_target_the_table_wrapper(self, client: TestClient) -> None:
        create_user("bob", hash_password("bobpass12345"), is_admin=False)
        html = client.get("/users").text
        assert 'id="users-table"' in html
        htmx_forms = [f for f in _FORM_RE.findall(html) if "hx-post" in f]
        assert len(htmx_forms) == 4  # 2 users x (toggle-admin + delete)
        for form in htmx_forms:
            assert 'hx-target="#users-table"' in form
            assert 'hx-swap="outerHTML"' in form
        assert "hx-headers" not in html
        assert 'style="grid-template-columns' not in html

    def test_toggle_admin_htmx_returns_the_table_partial(self, client: TestClient) -> None:
        create_user("bob", hash_password("bobpass12345"), is_admin=False)
        r = _post(client, "/users/2/toggle-admin", htmx=True)
        assert r.status_code == 200
        _assert_fragment(r.text, "users-table")
        assert "bob is now an admin." in r.text
        assert get_user_by_id(2).role == "admin"

    def test_toggle_admin_without_htmx_redirects_with_flash(self, client: TestClient) -> None:
        create_user("bob", hash_password("bobpass12345"), is_admin=False)
        r = _post(client, "/users/2/toggle-admin")
        assert r.status_code == 303
        assert r.headers["location"] == "/users"
        assert any(c.startswith("warden_flash=") for c in r.headers.get_list("set-cookie"))

    def test_delete_user_htmx_returns_the_table_without_the_row(self, client: TestClient) -> None:
        create_user("carol", hash_password("carolpass1234"), is_admin=False)
        r = _post(client, "/users/2/delete", htmx=True)
        assert r.status_code == 200
        _assert_fragment(r.text, "users-table")
        assert "User carol deleted." in r.text
        assert "/users/2/delete" not in r.text
        assert get_user_by_id(2) is None

    def test_delete_self_htmx_shows_the_error_inside_the_table(self, client: TestClient) -> None:
        r = _post(client, "/users/1/delete", htmx=True)
        assert r.status_code == 200
        _assert_fragment(r.text, "users-table")
        assert "You cannot delete your own account." in r.text
        assert get_user_by_id(1) is not None

    def test_toggle_unknown_user_htmx_shows_not_found(self, client: TestClient) -> None:
        r = _post(client, "/users/99/toggle-admin", htmx=True)
        assert r.status_code == 200
        _assert_fragment(r.text, "users-table")
        assert "User not found." in r.text

    def test_new_user_and_reset_password_are_plain_forms(self, client: TestClient) -> None:
        for url in ("/users/new", "/users/1/reset-password"):
            form = _form_with_action(client.get(url).text, url)
            assert 'method="POST"' in form
            assert "hx-" not in form
            assert 'style="grid-template-columns' not in form


class TestTokensPanel:
    def test_tokens_page_forms_target_the_panel(self, client: TestClient) -> None:
        create_api_token(get_user_by_id(1), name="old", ttl_seconds=3600)
        html = client.get("/api-tokens").text
        assert 'id="tokens-panel"' in html
        htmx_forms = [f for f in _FORM_RE.findall(html) if "hx-post" in f]
        assert len(htmx_forms) == 2  # the create form + one revoke form
        for form in htmx_forms:
            assert 'hx-target="#tokens-panel"' in form
            assert 'hx-swap="outerHTML"' in form
        assert 'hx-target="article"' not in html
        assert "hx-headers" not in html
        assert 'style="grid-template-columns' not in html

    def test_create_token_htmx_returns_the_panel_with_the_one_time_token(
        self, client: TestClient
    ) -> None:
        r = _post(client, "/api-tokens", {"name": "ci-deploy", "expires_in_days": "30"}, htmx=True)
        assert r.status_code == 200
        _assert_fragment(r.text, "tokens-panel")
        assert 'class="token-created"' in r.text
        assert "will not be shown again" in r.text
        assert "ci-deploy" in r.text
        assert len(list_api_tokens(1)) == 1

    def test_create_token_htmx_validation_error_stays_in_the_panel(
        self, client: TestClient
    ) -> None:
        r = _post(client, "/api-tokens", {"name": "  ", "expires_in_days": "30"}, htmx=True)
        assert r.status_code == 200
        _assert_fragment(r.text, "tokens-panel")
        assert "Token name is required." in r.text
        assert list_api_tokens(1) == []

    def test_create_token_without_htmx_still_renders_the_page(self, client: TestClient) -> None:
        r = _post(client, "/api-tokens", {"name": "ci-deploy", "expires_in_days": "30"})
        assert r.status_code == 200
        assert "<html" in r.text
        assert 'id="tokens-panel"' in r.text
        assert 'class="token-created"' in r.text

    def test_revoke_token_htmx_returns_the_panel_without_the_token(
        self, client: TestClient
    ) -> None:
        create_api_token(get_user_by_id(1), name="old", ttl_seconds=3600)
        token_id = list_api_tokens(1)[0]["id"]
        r = _post(client, f"/api-tokens/{token_id}/revoke", htmx=True)
        assert r.status_code == 200
        _assert_fragment(r.text, "tokens-panel")
        assert "Token old revoked." in r.text
        assert f"/api-tokens/{token_id}/revoke" not in r.text
        assert list_api_tokens(1) == []

    def test_revoke_unknown_token(self, client: TestClient) -> None:
        r = _post(client, "/api-tokens/999/revoke", htmx=True)
        assert r.status_code == 200
        _assert_fragment(r.text, "tokens-panel")
        assert "Token not found." in r.text
        assert _post(client, "/api-tokens/999/revoke").status_code == 404


class TestFormsAndSecrets:
    def test_every_form_keeps_the_hidden_csrf_token(self, client: TestClient) -> None:
        create_user("bob", hash_password("bobpass12345"), is_admin=False)
        create_api_token(get_user_by_id(1), name="old", ttl_seconds=3600)
        token = client.cookies.get("warden_csrf", "")
        assert token
        bodies = {
            "GET /users": client.get("/users").text,
            "GET /users/new": client.get("/users/new").text,
            "GET /users/1/reset-password": client.get("/users/1/reset-password").text,
            "GET /api-tokens": client.get("/api-tokens").text,
            "htmx toggle-admin": _post(client, "/users/2/toggle-admin", htmx=True).text,
            "htmx token create": _post(
                client, "/api-tokens", {"name": "ci", "expires_in_days": "1"}, htmx=True
            ).text,
        }
        for label, body in bodies.items():
            forms = _FORM_RE.findall(body)
            assert forms, label
            for form in forms:
                assert f'name="csrf_token" value="{token}"' in form, label

    def test_one_time_password_is_escaped_and_never_inside_javascript(
        self, client: TestClient
    ) -> None:
        password = "it's\"<b>x1"
        r = _post(client, "/users/new", {"username": "dave", "password": password})
        assert r.status_code == 200
        assert "created successfully" in r.text
        assert password not in r.text
        assert 'value="it&#39;s&#34;&lt;b&gt;x1"' in r.text
        assert 'x-ref="secret"' in r.text
        assert ":value=" not in r.text
        assert "writeText('" not in r.text

    def test_one_time_token_is_in_the_value_attribute(self, client: TestClient) -> None:
        r = _post(client, "/api-tokens", {"name": "ci", "expires_in_days": "1"}, htmx=True)
        assert r.status_code == 200
        assert 'value="wdt_' in r.text
        assert ":value=" not in r.text

    @pytest.mark.parametrize(
        "url", ["/users", "/users/new", "/users/1/reset-password", "/api-tokens"]
    )
    def test_page_uses_the_shared_page_header(self, client: TestClient, url: str) -> None:
        html = client.get(url).text
        assert html.count('<header class="page-header">') == 1

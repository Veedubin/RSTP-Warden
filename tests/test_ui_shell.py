"""UI shell: one Jinja2Templates instance, flash messages, nav, page header, CSS primitives."""

from __future__ import annotations

import base64
import importlib
import json
import re
import string
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.testclient import TestClient
from starlette.responses import Response

from rtsp_warden import __version__
from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.flash import (
    FLASH_COOKIE,
    MAX_FLASH_CHARS,
    Flash,
    decode_flash,
    encode_flash,
)
from rtsp_warden.web.paths import STATIC_DIR
from rtsp_warden.web.routes import _common
from rtsp_warden.web.routes._common import set_flash, templates

ROUTES_DIR = Path(_common.__file__).parent

# Modules that render through the shared instance after this task.
SHARED_TEMPLATE_MODULES = [
    "auth",
    "cameras",
    "dashboard",
    "health",
    "onvif",
    "settings",
    "tokens",
    "users",
    "zones",
]

# Route modules owned by the detection track (RW-3), which converts or deletes them.
RW3_OWNED_ROUTE_MODULES = {"clips.py", "events.py", "recordings.py"}


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303


@pytest.fixture
def app(db_with_user: str, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """App with one camera, no runtime, and two test-only routes that set a flash."""
    monkeypatch.setenv("WARDEN_AUTH_ENABLED", "true")
    cfg = AppConfig(cameras=[CameraConfig(name="front", main_url="rtsp://u:p@h/m")])
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: None)

    @app.get("/test-flash/{level}")
    async def _flash_then_redirect(level: str) -> RedirectResponse:
        response = RedirectResponse(url="/cameras", status_code=303)
        set_flash(response, f"Saved <b>{level}</b>", level)
        return response

    @app.get("/test-flash-page")
    async def _page_that_sets_a_new_flash() -> HTMLResponse:
        response = HTMLResponse("<p>page</p>")
        set_flash(response, "newer message", "error")
        return response

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    c = TestClient(app)
    _login(c)
    return c


# --- one templates instance -------------------------------------------------


@pytest.mark.parametrize("module_name", SHARED_TEMPLATE_MODULES)
def test_route_module_renders_through_shared_templates(module_name: str) -> None:
    module = importlib.import_module(f"rtsp_warden.web.routes.{module_name}")
    assert module.templates is templates
    assert not hasattr(module, "_templates")


def test_only_rw3_route_modules_still_build_their_own_templates() -> None:
    offenders = {
        path.name
        for path in ROUTES_DIR.glob("*.py")
        if path.name != "_common.py" and "Jinja2Templates(" in path.read_text(encoding="utf-8")
    }
    assert offenders <= RW3_OWNED_ROUTE_MODULES


def test_app_version_is_a_template_global() -> None:
    assert templates.env.globals["app_version"] == __version__


# --- flash codec ------------------------------------------------------------


def test_flash_round_trip_keeps_quotes_commas_and_non_ascii() -> None:
    message = 'Camera "Tür ✓" saved, restart; done'
    raw = encode_flash(message, "success")
    assert set(raw) <= set(string.ascii_letters + string.digits + "-_")
    assert decode_flash(raw) == Flash(message=message, level="success")


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "%%%",
        "not base64 ✓",
        base64.urlsafe_b64encode(b"not json").decode(),
        base64.urlsafe_b64encode(b"[1, 2]").decode(),
        base64.urlsafe_b64encode(b'{"m": 5, "l": "info"}').decode(),
        base64.urlsafe_b64encode(b'{"m": "", "l": "info"}').decode(),
    ],
)
def test_decode_flash_rejects_garbage(raw: str | None) -> None:
    assert decode_flash(raw) is None


def test_unknown_level_becomes_info() -> None:
    forged = base64.urlsafe_b64encode(
        json.dumps({"m": "hi", "l": 'x" onclick="alert(1)'}).encode()
    ).decode()
    assert decode_flash(forged) == Flash(message="hi", level="info")
    assert decode_flash(encode_flash("hi", "danger")) == Flash(message="hi", level="info")


def test_long_flash_message_is_truncated() -> None:
    flash = decode_flash(encode_flash("a" * (MAX_FLASH_CHARS + 100)))
    assert flash is not None
    assert len(flash.message) == MAX_FLASH_CHARS


def test_set_flash_cookie_attributes() -> None:
    response = Response()
    set_flash(response, "Saved", "success")
    header = response.headers["set-cookie"]
    assert header.startswith(f"{FLASH_COOKIE}=")
    assert "Max-Age=60" in header
    assert "Path=/" in header
    assert "HttpOnly" in header
    assert "SameSite=lax" in header


# --- flash through the middleware -------------------------------------------


def test_flash_is_shown_once_after_redirect(client: TestClient) -> None:
    r = client.get("/test-flash/success")
    assert r.status_code == 200
    assert r.url.path == "/cameras"
    assert '<div id="flash" class="flash flash-success" role="status">' in r.text
    assert "Saved &lt;b&gt;success&lt;/b&gt;" in r.text
    assert client.cookies.get(FLASH_COOKIE) is None

    again = client.get("/cameras")
    assert 'id="flash"' not in again.text


def test_error_flash_uses_alert_role(client: TestClient) -> None:
    r = client.get("/test-flash/error")
    assert '<div id="flash" class="flash flash-error" role="alert">' in r.text


def test_htmx_fragment_does_not_consume_pending_flash(client: TestClient) -> None:
    r = client.get("/test-flash/info", follow_redirects=False)
    assert r.status_code == 303
    assert client.cookies.get(FLASH_COOKIE)

    fragment = client.get("/health/partial", headers={"HX-Request": "true"})
    assert fragment.status_code == 200
    assert 'id="flash"' not in fragment.text
    assert client.cookies.get(FLASH_COOKIE)

    page = client.get("/cameras")
    assert "flash-info" in page.text
    assert client.cookies.get(FLASH_COOKIE) is None


def test_redirect_does_not_consume_pending_flash(client: TestClient) -> None:
    client.get("/test-flash/info", follow_redirects=False)
    # A logged-in GET /login answers 303 -> "/": the flash must wait for the page.
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 303
    assert client.cookies.get(FLASH_COOKIE)

    page = client.get("/")
    assert "flash-info" in page.text
    assert client.cookies.get(FLASH_COOKIE) is None


def test_page_that_sets_a_new_flash_keeps_it(client: TestClient) -> None:
    client.get("/test-flash/info", follow_redirects=False)
    r = client.get("/test-flash-page")
    flash_headers = [
        v for v in r.headers.get_list("set-cookie") if v.startswith(f"{FLASH_COOKIE}=")
    ]
    assert len(flash_headers) == 1
    assert "Max-Age=60" in flash_headers[0]
    assert decode_flash(client.cookies.get(FLASH_COOKIE)) == Flash("newer message", "error")


def test_logout_flash_on_login_page(client: TestClient) -> None:
    token = client.cookies.get("warden_csrf", "")
    r = client.post("/logout", data={"csrf_token": token})
    assert r.url.path == "/login"
    html = r.text
    assert "You have been logged out." in html
    assert "hx-post" not in html
    assert html.count("<form") == 1  # only the login form; the flash adds none

    again = client.get("/login")
    assert "You have been logged out." not in again.text


# --- footer, health partial, nav --------------------------------------------


@pytest.mark.parametrize("path", ["/", "/cameras", "/settings", "/users", "/api-tokens", "/health"])
def test_footer_shows_version(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 200
    assert f"rtsp-warden v{__version__}</small>" in r.text


def test_footer_shows_version_on_login_page(app: FastAPI) -> None:
    html = TestClient(app).get("/login").text
    assert f"rtsp-warden v{__version__}</small>" in html


def test_health_partial_keeps_version_cell(app: FastAPI) -> None:
    r = TestClient(app).get("/health/partial")
    assert r.status_code == 200
    assert f"<td>{__version__}</td>" in r.text


def test_nav_marks_current_section(client: TestClient) -> None:
    html = client.get("/cameras/front").text
    assert '<a href="/cameras" aria-current="page">Cameras</a>' in html
    assert '<a href="/events">Events</a>' in html
    assert '<a href="/">Dashboard</a>' in html

    home = client.get("/").text
    assert '<a href="/" aria-current="page">Dashboard</a>' in home
    assert '<a href="/cameras">Cameras</a>' in home


def test_nav_has_wrap_class_health_link_and_account_dropdown(client: TestClient) -> None:
    html = client.get("/cameras").text
    assert '<nav class="container-fluid site-nav">' in html
    assert '<a href="/health">Health</a>' in html
    assert '<details class="dropdown">' in html
    assert '<a href="/api-tokens">API Tokens</a>' in html
    assert 'action="/logout"' in html


# --- page header partial ----------------------------------------------------


def test_page_header_with_subtitle_and_actions() -> None:
    html = templates.env.from_string(
        '{% set title = "Cameras" %}{% set subtitle = "2 configured" %}'
        '{% set header_actions %}<a href="/cameras/new" role="button">Add camera</a>{% endset %}'
        '{% include "partials/page_header.html" %}'
    ).render()
    assert '<header class="page-header">' in html
    assert "<h1>Cameras</h1>" in html
    assert "<p>2 configured</p>" in html
    assert (
        '<div class="page-actions"><a href="/cameras/new" role="button">Add camera</a></div>'
        in html
    )


def test_page_header_title_only_and_escaped() -> None:
    html = templates.get_template("partials/page_header.html").render(title="<b>Users</b>")
    assert "<h1>&lt;b&gt;Users&lt;/b&gt;</h1>" in html
    assert "<p>" not in html
    assert "page-actions" not in html


# --- CSS primitives ---------------------------------------------------------


def test_css_defines_layout_primitives(app: FastAPI) -> None:
    css = (STATIC_DIR / "css" / "warden.css").read_text(encoding="utf-8")
    for needle in (
        ".site-nav,\n.site-nav > ul {\n  flex-wrap: wrap;",
        ".page-header {",
        ".page-actions {",
        ".flash {",
        ".flash-success {",
        ".flash-error {",
        ".table-wrap {\n  overflow-x: auto;",
        ".status-badge {",
        ".small {",
        ".sr-only {",
        ".field {",
        "a.button {",
        ".status-dot.status-idle",
        ".status-dot.status-waiting",
        "@media (max-width: 600px) {",
        "grid-template-columns: 1fr !important;",
    ):
        assert needle in css, needle
    # Status colours used on text labels must not paint a background box.
    assert re.search(r"^\.status-(ok|error|unknown|stopped)\s*\{", css, re.MULTILINE) is None
    served = TestClient(app).get("/static/css/warden.css")
    assert served.status_code == 200
    assert ".flash-error" in served.text

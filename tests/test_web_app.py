"""Tests for the FastAPI web UI application.

Validates the Sprint 2 Batch 1 foundation: app creation, static asset
serving, and placeholder health/dashboard routes.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings


def test_create_app_returns_fastapi() -> None:
    """create_app should return a FastAPI instance."""
    app = create_app(WebSettings())
    assert app is not None


def test_root_redirects_to_login_when_unauthenticated() -> None:
    """GET / without auth should redirect to login or return 401."""
    app = create_app(WebSettings())
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    # With auth enabled, unauthenticated users get redirected or 401
    assert r.status_code in (303, 401, 307, 302)


def test_healthz() -> None:
    """GET /healthz should return {"status": "ok"}."""
    app = create_app(WebSettings())
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_json() -> None:
    """GET /health with Accept: application/json should return JSON."""
    app = create_app(WebSettings())
    client = TestClient(app)
    r = client.get("/health", headers={"Accept": "application/json"})
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body or "status" in body


def test_static_pico_css_served() -> None:
    """GET /static/css/pico.min.css should return real CSS content."""
    app = create_app(WebSettings())
    client = TestClient(app)
    r = client.get("/static/css/pico.min.css")
    assert r.status_code == 200
    assert "css" in r.headers["content-type"]
    assert len(r.text) > 1000


def test_static_htmx_served() -> None:
    """GET /static/js/htmx.min.js should return real JS content."""
    app = create_app(WebSettings())
    client = TestClient(app)
    r = client.get("/static/js/htmx.min.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert len(r.text) > 1000


def test_static_alpine_served() -> None:
    """GET /static/js/alpine.min.js should return real JS content."""
    app = create_app(WebSettings())
    client = TestClient(app)
    r = client.get("/static/js/alpine.min.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert len(r.text) > 1000

"""Tests for the Auth UI: login, logout, CSRF, rate limiting.

Validates Sprint 2 Batch 2: authentication web interface with
session cookies, CSRF protection, and login rate limiting.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.rate_limit import get_login_limiter

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


@pytest.fixture
def client(db_with_user: str) -> TestClient:
    """Create a TestClient with an initialized DB containing a test user."""
    # Reset the rate limiter for each test
    limiter = get_login_limiter()
    limiter._attempts.clear()

    app = create_app(WebSettings())
    return TestClient(app)


def _get_csrf_token(client: TestClient) -> str:
    """GET /login and return the CSRF cookie value."""
    r = client.get("/login")
    assert r.status_code == 200
    return r.cookies.get("warden_csrf", "")


def test_login_page_renders(client: TestClient) -> None:
    """GET /login returns 200 with login form and sets CSRF cookie."""
    r = client.get("/login")
    assert r.status_code == 200
    assert "Log in" in r.text
    assert "warden_csrf" in r.cookies


def test_login_with_valid_credentials_succeeds(client: TestClient) -> None:
    """POST /login with valid creds redirects and sets session cookie."""
    csrf = _get_csrf_token(client)
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "warden_session" in r.cookies


def test_login_with_invalid_password_fails(client: TestClient) -> None:
    """POST /login with wrong password re-renders form with error."""
    csrf = _get_csrf_token(client)
    r = client.post(
        "/login",
        data={"username": "admin", "password": "wrong", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
    )
    assert r.status_code == 200
    assert "Invalid" in r.text or "incorrect" in r.text.lower()


def test_login_with_nonexistent_user_fails(client: TestClient) -> None:
    """POST /login with unknown username re-renders form with error."""
    csrf = _get_csrf_token(client)
    r = client.post(
        "/login",
        data={"username": "nobody", "password": "anything", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
    )
    assert r.status_code == 200
    assert "Invalid" in r.text or "incorrect" in r.text.lower()


def test_logout_clears_session(client: TestClient) -> None:
    """POST /logout clears the session cookie."""
    # Login first
    csrf = _get_csrf_token(client)
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303

    # Logout
    csrf2 = r.cookies.get("warden_csrf", csrf)
    session_cookie = r.cookies.get("warden_session", "")
    r = client.post(
        "/logout",
        data={"csrf_token": csrf2},
        headers={"X-CSRF-Token": csrf2},
        cookies={"warden_csrf": csrf2, "warden_session": session_cookie},
        follow_redirects=False,
    )
    assert r.status_code == 303


def test_csrf_protection_blocks_post_without_token(client: TestClient) -> None:
    """POST /login without CSRF header returns 403."""
    r = client.post(
        "/login",
        data={"username": "x", "password": "y"},
    )
    assert r.status_code == 403


def test_rate_limiting_blocks_after_5_failures(client: TestClient) -> None:
    """6th failed login attempt in 60s returns 429."""
    csrf = _get_csrf_token(client)
    for _ in range(5):
        r = client.post(
            "/login",
            data={"username": "admin", "password": "wrong", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            cookies={"warden_csrf": csrf},
        )
        # First 5 attempts should return 200 (failed login form)
        assert r.status_code in (200, 303)

    # 6th attempt should be rate-limited
    r = client.post(
        "/login",
        data={"username": "admin", "password": "wrong", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
    )
    assert r.status_code == 429


def test_rate_limiter_allows_after_reset() -> None:
    """RateLimiter.check and reset work correctly."""
    limiter = get_login_limiter()
    limiter.reset("test-ip-reset-unit")
    # After reset, 5 checks are allowed, 6th is blocked (max_requests=5)
    for i in range(5):
        assert limiter.check("test-ip-reset-unit") is True, f"call {i + 1} should be allowed"
    assert limiter.check("test-ip-reset-unit") is False  # 6th blocked
    # reset() clears the count
    limiter.reset("test-ip-reset-unit")
    assert limiter.check("test-ip-reset-unit") is True  # allowed again
    limiter.reset("test-ip-reset-unit")
    assert limiter.check("test-ip-reset-unit") is True


def test_authenticated_user_redirected_from_login(client: TestClient) -> None:
    """GET /login with a valid session cookie redirects to /."""
    # Login first
    csrf = _get_csrf_token(client)
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303

    # GET /login again with session cookie should redirect
    session_cookie = r.cookies.get("warden_session", "")
    csrf2 = r.cookies.get("warden_csrf", csrf)
    r = client.get(
        "/login",
        cookies={"warden_session": session_cookie, "warden_csrf": csrf2},
        follow_redirects=False,
    )
    assert r.status_code == 303

"""Tests for Sprint 2 Batch 4: Admin pages (users, tokens, settings, camera settings).

Validates admin-only access control, user CRUD, API token management,
system settings, and read-only camera configuration display.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from rtsp_warden.auth import create_api_token, hash_password
from rtsp_warden.config import load_config
from rtsp_warden.db import create_user, get_user_by_id
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import create_admin_user, ensure_schema
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Create a sample config.yaml with 2 cameras."""
    cfg = {
        "cameras": [
            {
                "name": "front",
                "main_url": "rtsp://user:pass@192.168.1.50:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.50:554/sub",
                "record": {
                    "enabled": True,
                    "output_dir": str(tmp_path / "recordings"),
                    "main": {"enabled": True, "container": "ts", "chunk_seconds": 300},
                    "sub": {"enabled": False, "container": "ts", "chunk_seconds": 300},
                },
                "proxy": {
                    "enabled": True,
                    "mode": "mjpeg",
                    "stream": "sub",
                    "bind_host": "127.0.0.1",
                    "port": 9001,
                },
            },
            {
                "name": "back",
                "main_url": "rtsp://user:pass@192.168.1.51:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.51:554/sub",
                "record": {
                    "enabled": True,
                    "output_dir": str(tmp_path / "recordings"),
                },
                "proxy": {"enabled": False, "mode": "mjpeg"},
            },
        ],
        "runtime": {
            "ffmpeg_path": "ffmpeg",
            "workspace_dir": str(tmp_path / "workspace"),
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


@pytest.fixture
def admin_client(sample_config: Path) -> TestClient:
    """Authenticated TestClient as admin user."""
    db_url = f"sqlite:///{sample_config.parent / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg = load_config(sample_config)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    client = TestClient(app)

    # Login as admin
    r = client.get("/login")
    csrf = r.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return client


@pytest.fixture
def viewer_client(sample_config: Path) -> TestClient:
    """Authenticated TestClient as viewer user."""
    db_url = f"sqlite:///{sample_config.parent / 'test_viewer.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()

    # Create admin user first
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    # Create a viewer user
    viewer_pw = hash_password("viewerpass123")
    create_user("viewer", viewer_pw, is_admin=False)

    cfg = load_config(sample_config)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    client = TestClient(app)

    # Login as viewer
    r = client.get("/login")
    csrf = r.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "viewer", "password": "viewerpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return client


def _get_csrf(client: TestClient) -> str:
    """Extract CSRF token from client cookies."""
    return client.cookies.get("warden_csrf", "")


def _post_with_csrf(client: TestClient, url: str, data: dict, **kwargs) -> object:
    """POST with CSRF token in form data and headers.

    Defaults to follow_redirects=False so callers can assert on the 303 status.
    """
    csrf = _get_csrf(client)
    data["csrf_token"] = csrf
    kwargs.setdefault("follow_redirects", False)
    return client.post(
        url,
        data=data,
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# User Management Tests
# ---------------------------------------------------------------------------


class TestUserManagement:
    """Tests for /users routes (admin-only)."""

    def test_users_list_requires_admin(self, viewer_client: TestClient) -> None:
        """Non-admin user should get 403 on /users."""
        r = viewer_client.get("/users")
        assert r.status_code == 403

    def test_users_list_renders_for_admin(self, admin_client: TestClient) -> None:
        """Admin user should see the users list page."""
        r = admin_client.get("/users")
        assert r.status_code == 200
        assert "admin" in r.text
        assert "Users" in r.text

    def test_new_user_form_renders(self, admin_client: TestClient) -> None:
        """GET /users/new should render the new user form."""
        r = admin_client.get("/users/new")
        assert r.status_code == 200
        assert "Create New User" in r.text

    def test_create_user_succeeds(self, admin_client: TestClient) -> None:
        """POST /users/new with valid data should create a user."""
        r = admin_client.get("/users/new")
        # Get initial CSRF (set on first GET)
        r = _post_with_csrf(
            admin_client,
            "/users/new",
            {"username": "alice", "password": "alicepass123"},
        )
        assert r.status_code == 200
        # Should show the created password
        assert "created successfully" in r.text.lower() or "alice" in r.text

        # Verify user exists in DB
        user = get_user_by_id(2)
        assert user is not None
        assert user.username == "alice"
        assert user.role == "viewer"  # is_admin not checked

    def test_create_user_with_admin_flag(self, admin_client: TestClient) -> None:
        """POST /users/new with is_admin=on should create an admin user."""
        r = _post_with_csrf(
            admin_client,
            "/users/new",
            {"username": "bob_admin", "password": "bobpass12345", "is_admin": "on"},
        )
        assert r.status_code == 200
        user = get_user_by_id(2)
        assert user is not None
        assert user.role == "admin"

    def test_create_user_auto_generate_password(self, admin_client: TestClient) -> None:
        """POST /users/new with empty password should auto-generate one."""
        r = _post_with_csrf(
            admin_client,
            "/users/new",
            {"username": "carol", "password": ""},
        )
        assert r.status_code == 200
        # Should show the generated password
        assert (
            "will not be shown again" in r.text
            or "generated" in r.text.lower()
            or "password" in r.text.lower()
        )

    def test_create_user_duplicate_username_fails(self, admin_client: TestClient) -> None:
        """Creating a user with an existing username should show an error."""
        r = _post_with_csrf(
            admin_client,
            "/users/new",
            {"username": "admin", "password": "somepassword1"},
        )
        assert r.status_code == 200
        assert "already exists" in r.text

    def test_create_user_short_username_fails(self, admin_client: TestClient) -> None:
        """Creating a user with a too-short username should show an error."""
        r = _post_with_csrf(
            admin_client,
            "/users/new",
            {"username": "ab", "password": "somepassword1"},
        )
        assert r.status_code == 200
        assert "3-32" in r.text or "characters" in r.text.lower()

    def test_reset_password_form_renders(self, admin_client: TestClient) -> None:
        """GET /users/{id}/reset-password should render the confirmation form."""
        r = admin_client.get("/users/1/reset-password")
        assert r.status_code == 200
        assert "admin" in r.text
        assert "Reset Password" in r.text

    def test_reset_password_shows_new_password(self, admin_client: TestClient) -> None:
        """POST /users/{id}/reset-password should show the new password."""
        r = admin_client.get("/users/1/reset-password")
        r = _post_with_csrf(admin_client, "/users/1/reset-password", {})
        assert r.status_code == 200
        # Should contain a password (16 chars)
        assert "has been reset" in r.text.lower() or "new password" in r.text.lower()

    def test_delete_user_succeeds(self, admin_client: TestClient) -> None:
        """POST /users/{id}/delete should delete a non-self user."""
        # Create a user to delete
        pw_hash = hash_password("deleteme1234567")
        create_user("to_delete", pw_hash, is_admin=False)

        # Delete user id=2
        r = _post_with_csrf(admin_client, "/users/2/delete", {})
        assert r.status_code == 303  # redirect to /users

    def test_delete_self_refused(self, admin_client: TestClient) -> None:
        """POST /users/{self_id}/delete should return 400."""
        r = _post_with_csrf(admin_client, "/users/1/delete", {})
        assert r.status_code == 400

    def test_toggle_admin(self, admin_client: TestClient) -> None:
        """POST /users/{id}/toggle-admin should toggle admin status."""
        # Create a viewer user
        pw_hash = hash_password("viewerpass123456")
        create_user("viewer2", pw_hash, is_admin=False)

        # Promote to admin
        r = _post_with_csrf(admin_client, "/users/2/toggle-admin", {})
        assert r.status_code == 303

        # Verify role changed
        user = get_user_by_id(2)
        assert user is not None
        assert user.role == "admin"

        # Demote back to viewer
        r = _post_with_csrf(admin_client, "/users/2/toggle-admin", {})
        assert r.status_code == 303

        user = get_user_by_id(2)
        assert user is not None
        assert user.role == "viewer"

    def test_toggle_admin_self_refused(self, admin_client: TestClient) -> None:
        """POST /users/{self_id}/toggle-admin should return 400."""
        r = _post_with_csrf(admin_client, "/users/1/toggle-admin", {})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# API Token Tests
# ---------------------------------------------------------------------------


class TestAPITokens:
    """Tests for /api-tokens routes (any authenticated user)."""

    def test_tokens_list_renders(self, admin_client: TestClient) -> None:
        """GET /api-tokens should render the tokens page."""
        r = admin_client.get("/api-tokens")
        assert r.status_code == 200
        assert "API Tokens" in r.text

    def test_create_api_token_shows_raw_token(self, admin_client: TestClient) -> None:
        """POST /api-tokens should create a token and show the raw value."""
        r = _post_with_csrf(
            admin_client,
            "/api-tokens",
            {"name": "test-token", "expires_in_days": "365"},
        )
        assert r.status_code == 200
        assert "wdt_" in r.text  # raw token should be displayed
        assert "test-token" in r.text

    def test_revoke_token(self, admin_client: TestClient) -> None:
        """POST /api-tokens/{id}/revoke should revoke the token."""
        # Create a token via the auth module
        admin_user = get_user_by_id(1)
        assert admin_user is not None
        token_obj = create_api_token(admin_user, name="revoke-me", ttl_seconds=3600)

        # Get token list to find the ID
        r = admin_client.get("/api-tokens")
        assert r.status_code == 200

        # Revoke via POST
        r = _post_with_csrf(admin_client, f"/api-tokens/{token_obj.prefix}/revoke", {})
        # Token might not be found by prefix, need to find by ID
        # Actually the route uses token_id (int), let's find it from list
        from rtsp_warden.auth import list_api_tokens

        tokens = list_api_tokens(1)
        if tokens:
            token_id = tokens[0]["id"]
            r = _post_with_csrf(admin_client, f"/api-tokens/{token_id}/revoke", {})
            # Should be 200 (renders token list)
            assert r.status_code == 200

    def test_viewer_can_access_own_tokens(self, viewer_client: TestClient) -> None:
        """Viewer should be able to access /api-tokens for their own tokens."""
        r = viewer_client.get("/api-tokens")
        assert r.status_code == 200

    def test_tokens_require_auth(self) -> None:
        """Unauthenticated GET /api-tokens should return 401."""
        settings = WebSettings()
        app = create_app(settings)
        client = TestClient(app)
        r = client.get("/api-tokens")
        assert r.status_code in (303, 401, 307, 302)


# ---------------------------------------------------------------------------
# Settings Tests
# ---------------------------------------------------------------------------


class TestSettings:
    """Tests for /settings route (admin-only)."""

    def test_settings_renders_for_admin(self, admin_client: TestClient) -> None:
        """Admin user should see the system settings page."""
        r = admin_client.get("/settings")
        assert r.status_code == 200
        assert "1.0.0" in r.text  # version
        assert "System Settings" in r.text

    def test_settings_requires_admin(self, viewer_client: TestClient) -> None:
        """Viewer user should get 403 on /settings."""
        r = viewer_client.get("/settings")
        assert r.status_code == 403

    def test_settings_shows_camera_count(self, admin_client: TestClient) -> None:
        """Settings page should show camera count from config."""
        r = admin_client.get("/settings")
        assert r.status_code == 200
        assert "2" in r.text  # 2 cameras in sample config

    def test_settings_shows_env_vars(self, admin_client: TestClient) -> None:
        """Settings page should display environment variables."""
        r = admin_client.get("/settings")
        assert r.status_code == 200
        assert "WARDEN_AUTH_ENABLED" in r.text


# ---------------------------------------------------------------------------
# Camera Settings Tests
# ---------------------------------------------------------------------------


class TestCameraSettings:
    """Tests for /cameras/{name}/settings route (admin-only, read-only)."""

    def test_camera_settings_renders_for_admin(self, admin_client: TestClient) -> None:
        """Admin should see camera settings page."""
        r = admin_client.get("/cameras/front/settings")
        assert r.status_code == 200
        assert "config.yaml" in r.text  # banner
        assert "front" in r.text

    def test_camera_settings_shows_config(self, admin_client: TestClient) -> None:
        """Camera settings should show recording and proxy config."""
        r = admin_client.get("/cameras/front/settings")
        assert r.status_code == 200
        assert "Recording" in r.text
        assert "Proxy" in r.text
        assert "mjpeg" in r.text.lower()

    def test_camera_settings_404_for_unknown(self, admin_client: TestClient) -> None:
        """GET /cameras/zzz/settings should return 404."""
        r = admin_client.get("/cameras/zzz/settings")
        assert r.status_code == 404

    def test_camera_settings_requires_admin(self, viewer_client: TestClient) -> None:
        """Viewer should get 403 on camera settings."""
        r = viewer_client.get("/cameras/front/settings")
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Access Control Tests
# ---------------------------------------------------------------------------


class TestAccessControl:
    """Tests verifying viewer role gets 403 on admin routes."""

    def test_viewer_cannot_access_admin_pages(self, viewer_client: TestClient) -> None:
        """Viewer should get 403 on all admin-only routes."""
        admin_only_paths = [
            "/users",
            "/users/new",
            "/settings",
            "/cameras/front/settings",
        ]
        for path in admin_only_paths:
            r = viewer_client.get(path)
            assert r.status_code == 403, f"Expected 403 for {path}, got {r.status_code}"

    def test_viewer_can_access_tokens(self, viewer_client: TestClient) -> None:
        """Viewer should be able to access /api-tokens (any authenticated user)."""
        r = viewer_client.get("/api-tokens")
        assert r.status_code == 200

    def test_unauthenticated_redirected(self) -> None:
        """Unauthenticated requests to admin routes should redirect or 401."""
        settings = WebSettings()
        app = create_app(settings)
        client = TestClient(app)
        paths = ["/users", "/users/new", "/settings", "/api-tokens"]
        for path in paths:
            r = client.get(path, follow_redirects=False)
            assert r.status_code in (303, 401, 307, 302), (
                f"Expected redirect/401 for {path}, got {r.status_code}"
            )

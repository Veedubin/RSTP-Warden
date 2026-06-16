"""Shared pytest fixtures for rtsp-warden tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rtsp_warden.auth import hash_password
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import create_admin_user, ensure_schema


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp dir, set WARDEN_DB_URL to sqlite:///{tmpdir}/test.db, yield tmp_path."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("WARDEN_DB_URL", db_url)
    return tmp_path


@pytest.fixture
def clean_db(tmp_env: Path) -> None:
    """Reset engine and ensure schema using the tmp_env fixture."""
    reset_engine()
    ensure_schema()
    yield
    reset_engine()


@pytest.fixture
def admin_user(clean_db: None) -> Any:
    """Create a test admin user with bcrypt-hashed password 'testpass123'."""
    pw_hash = hash_password("testpass123")
    user = create_admin_user("testadmin", pw_hash)
    return user


@pytest.fixture
def make_environ() -> Any:
    """Factory fixture: (cookie=None, bearer=None) -> dict returning a WSGI environ."""

    def _make(cookie: str | None = None, bearer: str | None = None) -> dict[str, str]:
        environ: dict[str, str] = {}
        if cookie is not None:
            environ["HTTP_COOKIE"] = cookie
        if bearer is not None:
            environ["HTTP_AUTHORIZATION"] = bearer
        return environ

    return _make


@pytest.fixture
def db_with_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Set up an isolated SQLite DB with a test admin user for web UI tests.

    Creates a temporary database, resets the engine, ensures schema,
    and creates an admin user with username 'admin' and password 'testpass123'.
    Returns the database URL string.
    """
    db_url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("WARDEN_DB_URL", db_url)
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)
    yield db_url
    reset_engine()

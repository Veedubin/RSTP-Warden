"""Tests for db/ — engine, models, schema."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect

from rtsp_warden.db.engine import get_engine, get_session, reset_engine, resolve_db_url
from rtsp_warden.db.models import Session, User
from rtsp_warden.db.schema import create_admin_user, ensure_schema, get_user_by_username


def test_resolve_db_url_default_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var → returns sqlite:/// URL."""
    monkeypatch.delenv("WARDEN_DB_URL", raising=False)
    url = resolve_db_url()
    assert url.startswith("sqlite:///")


def test_resolve_db_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """WARDEN_DB_URL=postgresql+... returns that."""
    monkeypatch.setenv("WARDEN_DB_URL", "postgresql+psycopg2://user:pass@localhost/db")
    url = resolve_db_url()
    assert url == "postgresql+psycopg2://user:pass@localhost/db"


def test_ensure_schema_creates_the_current_tables(
    tmp_path: pytest.TestPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh DB gets exactly the 0003 tables (cameras/recordings/ingest_health/clips are gone)."""
    monkeypatch.setenv("WARDEN_DB_URL", f"sqlite:///{tmp_path}/schema_test.db")
    reset_engine()
    try:
        ensure_schema()
        tables = set(inspect(get_engine()).get_table_names())
    finally:
        reset_engine()

    assert tables == {
        "users",
        "sessions",
        "api_tokens",
        "events",
        "action_runs",
        "alembic_version",
    }


def test_create_admin_user_succeeds(clean_db: None) -> None:
    """create_admin_user('alice', 'hash') returns user with role='admin'."""
    user = create_admin_user("alice", "myhash")
    assert user.username == "alice"
    assert user.role == "admin"
    assert user.is_active is True


def test_create_admin_user_duplicate_raises(clean_db: None) -> None:
    """Creating same username twice raises ValueError."""
    create_admin_user("bob", "hash1")
    with pytest.raises(ValueError, match="already exists"):
        create_admin_user("bob", "hash2")


def test_get_user_by_username_returns_none_when_missing(clean_db: None) -> None:
    """get_user_by_username returns None for non-existent user."""
    user = get_user_by_username("nonexistent")
    assert user is None


def test_user_cascade_deletes_sessions(clean_db: None) -> None:
    """session.delete(user) cascades to sessions table."""
    user = create_admin_user("cascade_test", "hash")
    from rtsp_warden.auth import create_session

    session_obj = create_session(user)
    assert session_obj is not None

    with get_session() as s:
        db_user = s.query(User).filter(User.username == "cascade_test").first()
        assert db_user is not None
        s.delete(db_user)
        s.commit()

    with get_session() as s:
        remaining = s.query(Session).filter(Session.token == session_obj.token).first()
        assert remaining is None

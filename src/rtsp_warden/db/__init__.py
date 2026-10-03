"""rtsp-warden database package: SQLAlchemy 2.x dual-DB (SQLite + PostgreSQL)."""

from __future__ import annotations

from .engine import get_engine, get_session, reset_engine, resolve_db_url
from .models import (
    ActionRun,
    ApiToken,
    Base,
    Event,
    Session,
    User,
)
from .schema import (
    as_utc,
    create_admin_user,
    create_user,
    delete_user,
    ensure_schema,
    get_latest_event_for_camera,
    get_user_by_id,
    get_user_by_username,
    list_users,
    set_user_admin,
    update_user_password,
)

__all__ = [
    "Base",
    "get_engine",
    "get_session",
    "reset_engine",
    "resolve_db_url",
    "User",
    "Session",
    "ApiToken",
    "Event",
    "ActionRun",
    "as_utc",
    "ensure_schema",
    "create_admin_user",
    "create_user",
    "delete_user",
    "get_latest_event_for_camera",
    "get_user_by_id",
    "get_user_by_username",
    "list_users",
    "set_user_admin",
    "update_user_password",
]

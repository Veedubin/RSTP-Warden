from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import case, update
from sqlalchemy import func as sa_func

from .engine import get_engine, get_session
from .models import ActionRun, Event, User

if TYPE_CHECKING:
    from alembic.config import Config
    from sqlalchemy.engine import Connection, Engine

log = logging.getLogger(__name__)

# Migrations ship inside the package, so wheels and Docker images (which copy only src/) have them.
SCRIPT_LOCATION = "rtsp_warden:migrations"


def _alembic_config(connection: Connection | None = None) -> Config:
    """Build the Alembic Config in code.

    No ini file is read, so ``env.py`` never touches logging. With *connection*, migrations run
    on that connection (``config.attributes["connection"]``) and no URL is serialised, which
    keeps a Postgres password out of the config and avoids ConfigParser's ``%`` interpolation.
    """
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", SCRIPT_LOCATION)
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def _backend_name(engine: Engine) -> str:
    """The SQLAlchemy backend name of *engine* ("sqlite", "postgresql", ...)."""
    return engine.url.get_backend_name()


def _backup_sqlite(engine: Engine, revision: str | None) -> Path | None:
    """Copy a file-backed SQLite database to ``<db>.bak-<revision>`` before an upgrade.

    SQLite DDL is not transactional and migration 0003 rebuilds the events table, so a failed
    upgrade can leave a half-migrated file. An existing backup is never overwritten (``.2``,
    ``.3``, ... are appended). Returns None for other backends and in-memory databases.
    """
    database = engine.url.database
    if _backend_name(engine) != "sqlite" or not database or database == ":memory:":
        return None
    source = Path(database)
    if not source.is_file():
        return None
    first = source.with_name(f"{source.name}.bak-{revision or 'base'}")
    target = first
    n = 1
    while target.exists():
        n += 1
        target = first.with_name(f"{first.name}.{n}")
    engine.dispose()  # no pooled connection may hold the file while it is copied
    shutil.copy2(source, target)
    return target


def ensure_schema() -> None:
    """Bring the database to the newest schema. Safe to call on every start.

    - Empty database: run every migration.
    - Tables but no ``alembic_version`` (a pre-Alembic ``create_all`` database): stamp head.
    - Behind head: copy a SQLite file to ``<db>.bak-<revision>``, then upgrade to head. A
      database that cannot be copied (PostgreSQL) is upgraded only when ``WARDEN_DB_UPGRADE=1``
      is set, because an upgrade can drop tables; without it ``SystemExit`` says what to do.
    - At head: nothing.
    - At a revision this release does not know (written by a newer release): ``SystemExit``
      with a message saying what to do. Nothing is changed.
    """
    from alembic import command
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from alembic.util import CommandError
    from sqlalchemy import inspect

    engine = get_engine()
    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    tables = set(inspect(engine).get_table_names())

    if not tables:
        with engine.begin() as conn:
            command.upgrade(_alembic_config(conn), "head")
        return

    if "alembic_version" not in tables:
        # Legacy create_all DB: stamp it as current rather than running migrations.
        log.info("[db] legacy schema detected; stamping as alembic head")
        with engine.begin() as conn:
            command.stamp(_alembic_config(conn), "head")
        return

    with engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    if current == head:
        log.info("[db] alembic current revision: %s", current)
        return

    if current is not None:
        try:
            ScriptDirectory.from_config(_alembic_config()).get_revision(current)
        except CommandError:
            raise SystemExit(
                f"[db] the database is at schema revision {current!r}, which this version of "
                f"rtsp-warden does not know (newest known: {head!r}). It was probably written by "
                "a newer release: upgrade rtsp-warden, or restore the <db>.bak-<revision> backup "
                "taken before that upgrade. "
                f"Database: {engine.url.render_as_string(hide_password=True)}"
            ) from None

    backup = _backup_sqlite(engine, current)
    if (
        backup is None
        and _backend_name(engine) != "sqlite"
        and os.environ.get("WARDEN_DB_UPGRADE") != "1"
    ):
        raise SystemExit(
            f"[db] the database is at schema revision {current!r}, this release needs {head!r}, "
            "and the upgrade drops tables and columns. Take a backup first (for example with "
            "pg_dump), then start once with WARDEN_DB_UPGRADE=1. "
            f"Database: {engine.url.render_as_string(hide_password=True)}"
        )
    log.warning(
        "[db] upgrading the database schema from %s to %s (backup: %s)",
        current,
        head,
        backup if backup is not None else "none, not a SQLite file",
    )
    with engine.begin() as conn:
        command.upgrade(_alembic_config(conn), "head")
    log.info("[db] schema upgrade to %s done", head)


def create_admin_user(username: str, password_hash: str) -> User:
    """Create an admin user. Returns the created User. Raises if username exists.

    Use this in `rtsp-warden install` to bootstrap the first admin.
    """
    from .engine import get_session  # local import to avoid circular

    with get_session() as session:
        existing = session.query(User).filter(User.username == username).first()
        if existing is not None:
            raise ValueError(f"user {username!r} already exists")

        user = User(
            username=username,
            password_hash=password_hash,
            role="admin",
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        # Detach so the caller can use it after the session closes
        session.expunge(user)
        return user


def create_user(username: str, password_hash: str, is_admin: bool = False) -> User:
    """Create a new user. Returns the created User. Raises if username exists.

    Use this from the admin UI to create viewers or additional admins.
    """
    from .engine import get_session

    with get_session() as session:
        existing = session.query(User).filter(User.username == username).first()
        if existing is not None:
            raise ValueError(f"user {username!r} already exists")

        role = "admin" if is_admin else "viewer"
        user = User(
            username=username,
            password_hash=password_hash,
            role=role,
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        session.expunge(user)
        return user


def get_user_by_username(username: str) -> User | None:
    """Fetch a user by username. Returns None if not found."""
    from .engine import get_session

    with get_session() as session:
        user = session.query(User).filter(User.username == username).first()
        if user is not None:
            session.expunge(user)
        return user


def get_user_by_id(user_id: int) -> User | None:
    """Fetch a user by primary key. Returns None if not found."""
    from .engine import get_session

    with get_session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if user is not None:
            session.expunge(user)
        return user


def list_users() -> list[User]:
    """Return all users ordered by id."""
    from .engine import get_session

    with get_session() as session:
        users = session.query(User).order_by(User.id).all()
        for u in users:
            session.expunge(u)
        return users


def delete_user(user_id: int) -> bool:
    """Delete a user by primary key. Returns True if a row was deleted."""
    from .engine import get_session

    with get_session() as session:
        count = session.query(User).filter(User.id == user_id).delete()
        session.commit()
        return count > 0


def update_user_password(user_id: int, password_hash: str) -> bool:
    """Update a user's password hash. Returns True if a row was updated."""
    from .engine import get_session

    with get_session() as session:
        count = (
            session.query(User).filter(User.id == user_id).update({"password_hash": password_hash})
        )
        session.commit()
        return count > 0


def set_user_admin(user_id: int, is_admin: bool) -> bool:
    """Set or unset a user's admin role. Returns True if a row was updated."""
    from .engine import get_session

    role = "admin" if is_admin else "viewer"
    with get_session() as session:
        count = session.query(User).filter(User.id == user_id).update({"role": role})
        session.commit()
        return count > 0


# ---------------------------------------------------------------------------
# Datetimes: every DB datetime is written as naive UTC and read back with as_utc().
# SQLite drops tzinfo (it would store an aware non-UTC value's local wall time), so the
# helpers below convert before writing.
# ---------------------------------------------------------------------------


def as_utc(dt: datetime | None) -> datetime | None:
    """Return *dt* as an aware UTC datetime: naive values are UTC, aware ones are converted."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _naive_utc(dt: datetime) -> datetime:
    """Return *dt* in the stored form: naive UTC. A naive input is taken to be UTC already."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Event CRUD
# ---------------------------------------------------------------------------

# Columns update_event may set; "metadata" is a dict stored as metadata_json.
_EVENT_UPDATE_FIELDS = frozenset(
    {
        "event_type",
        "severity",
        "message",
        "label",
        "confidence",
        "zone",
        "track_id",
        "ended_at",
        "thumbnail_path",
        "clip_path",
        "metadata",
    }
)


def insert_event(
    *,
    camera_name: str,
    event_type: str,
    label: str | None,
    confidence: float | None,
    zone: str,
    track_id: int | None,
    message: str,
    created_at: datetime,
    metadata: dict | None = None,
    thumbnail_path: str | None = None,
    severity: str = "info",
) -> int:
    """Insert one event row and return its id. *created_at* is stored as naive UTC."""
    with get_session() as session:
        event = Event(
            camera_name=camera_name,
            event_type=event_type,
            severity=severity,
            label=label,
            confidence=None if confidence is None else float(confidence),
            zone=zone,
            track_id=track_id,
            message=message[:512],
            metadata_json=json.dumps(metadata, default=str) if metadata else "{}",
            created_at=_naive_utc(created_at),
            thumbnail_path=thumbnail_path,
        )
        session.add(event)
        session.commit()
        return int(event.id)


def update_event(event_id: int, **fields: Any) -> None:
    """Set columns on one event (see ``_EVENT_UPDATE_FIELDS``). A missing event is ignored.

    Datetimes are stored as naive UTC, ``confidence`` as a plain float, and ``metadata`` (a
    dict) replaces ``metadata_json``. An unknown field raises ValueError.
    """
    unknown = set(fields) - _EVENT_UPDATE_FIELDS
    if unknown:
        raise ValueError(f"update_event: unknown field(s) {sorted(unknown)}")
    values: dict[str, Any] = {}
    for key, value in fields.items():
        if key == "metadata":
            values["metadata_json"] = json.dumps(value, default=str) if value else "{}"
        elif isinstance(value, datetime):
            values[key] = _naive_utc(value)
        elif key == "confidence" and value is not None:
            values[key] = float(value)
        else:
            values[key] = value
    if not values:
        return
    with get_session() as session:
        session.execute(update(Event).where(Event.id == event_id).values(**values))
        session.commit()


def close_event(event_id: int, ended_at: datetime, **fields: Any) -> None:
    """Set ``ended_at`` (and any other ``update_event`` fields) on one event."""
    update_event(event_id, ended_at=ended_at, **fields)


def get_event(event_id: int) -> Event | None:
    """Fetch one event by id, detached from the session. None if it does not exist."""
    with get_session() as session:
        event = session.get(Event, event_id)
        if event is not None:
            session.expunge(event)
        return event


def _filter_events(
    query: Any,
    camera_name: str | None,
    label: str | None,
    since: datetime | None,
    until: datetime | None,
) -> Any:
    if camera_name is not None:
        query = query.filter(Event.camera_name == camera_name)
    if label is not None:
        query = query.filter(Event.label == label)
    if since is not None:
        query = query.filter(Event.created_at >= _naive_utc(since))
    if until is not None:
        query = query.filter(Event.created_at <= _naive_utc(until))
    return query


def list_events(
    *,
    camera_name: str | None = None,
    label: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Event]:
    """Events matching the filters, newest first (ties by id), detached from the session."""
    with get_session() as session:
        query = _filter_events(session.query(Event), camera_name, label, since, until)
        events = (
            query.order_by(Event.created_at.desc(), Event.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        for event in events:
            session.expunge(event)
        return events


def count_events(
    *,
    camera_name: str | None = None,
    label: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> int:
    """Number of events matching the same filters as ``list_events``."""
    with get_session() as session:
        query = _filter_events(
            session.query(sa_func.count(Event.id)), camera_name, label, since, until
        )
        return int(query.scalar() or 0)


def get_latest_event_for_camera(camera_name: str, since_seconds: int = 0) -> Event | None:
    """Most recent event of *camera_name*, optionally only within the last *since_seconds*.

    ``since_seconds=0`` means all time. Used by event-mode recording.
    """
    with get_session() as session:
        query = session.query(Event).filter(Event.camera_name == camera_name)
        if since_seconds > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=since_seconds)
            query = query.filter(Event.created_at >= _naive_utc(cutoff))
        event = query.order_by(Event.created_at.desc(), Event.id.desc()).first()
        if event is not None:
            session.expunge(event)
        return event


# ---------------------------------------------------------------------------
# Action runs
# ---------------------------------------------------------------------------


def insert_action_run(*, event_id: int, action_name: str, status: str, error: str | None) -> int:
    """Record one action attempt for an event; returns the row id. *status* is "ok" or "failed"."""
    with get_session() as session:
        run = ActionRun(
            event_id=event_id,
            action_name=action_name,
            status=status,
            error=error,
            created_at=_naive_utc(datetime.now(timezone.utc)),
        )
        session.add(run)
        session.commit()
        return int(run.id)


def list_action_runs(event_id: int) -> list[ActionRun]:
    """Action runs of one event, oldest first, detached from the session."""
    with get_session() as session:
        runs = (
            session.query(ActionRun)
            .filter(ActionRun.event_id == event_id)
            .order_by(ActionRun.created_at, ActionRun.id)
            .all()
        )
        for run in runs:
            session.expunge(run)
        return runs


def action_stats() -> dict[str, dict]:
    """Per action name: ``{"last_run": aware UTC datetime, "last_status": str, "failures": int}``.

    ``failures`` counts every run with status "failed". Names with no runs are absent.
    """
    with get_session() as session:
        rows = (
            session.query(
                ActionRun.action_name,
                sa_func.max(ActionRun.id),
                sa_func.sum(case((ActionRun.status == "failed", 1), else_=0)),
            )
            .group_by(ActionRun.action_name)
            .all()
        )
        last_ids = [row[1] for row in rows]
        last_runs = {
            run.id: run for run in session.query(ActionRun).filter(ActionRun.id.in_(last_ids)).all()
        }
        stats: dict[str, dict] = {}
        for name, last_id, failures in rows:
            last = last_runs[last_id]
            stats[name] = {
                "last_run": as_utc(last.created_at),
                "last_status": last.status,
                "failures": int(failures or 0),
            }
        return stats

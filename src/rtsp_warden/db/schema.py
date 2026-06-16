from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func as sa_func

from .engine import get_engine, get_session
from .models import Camera, Clip, Event, User

log = logging.getLogger(__name__)

_ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent.parent / "alembic.ini"


def ensure_schema() -> None:
    """Create all tables if they don't exist. Safe to call on every startup.

    Uses Alembic to apply migrations. If the alembic version table doesn't
    exist, runs ``alembic upgrade head``. If it does exist, just checks that
    we're at head (does not auto-upgrade -- that's the operator's job in
    production).

    For development: also stamps the DB as current if it has tables but no
    alembic version (i.e., it was created with the old ``create_all`` approach).
    """
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import inspect

    engine = get_engine()

    # Check if alembic version table exists
    insp = inspect(engine)
    has_alembic = "alembic_version" in insp.get_table_names()
    has_any_tables = len(insp.get_table_names()) > 0

    if not has_alembic and not has_any_tables:
        # Fresh DB: run migrations
        cfg = Config(str(_ALEMBIC_INI))
        cfg.set_main_option(
            "script_location",
            str(Path(__file__).resolve().parent.parent.parent.parent / "migrations"),
        )
        cfg.set_main_option("sqlalchemy.url", str(engine.url))
        command.upgrade(cfg, "head")
    elif not has_alembic and has_any_tables:
        # DB has tables but no alembic version: this is a legacy create_all DB.
        # Stamp it as current rather than running migrations (which would fail).
        log.info("[db] legacy schema detected; stamping as alembic head")
        cfg = Config(str(_ALEMBIC_INI))
        cfg.set_main_option(
            "script_location",
            str(Path(__file__).resolve().parent.parent.parent.parent / "migrations"),
        )
        cfg.set_main_option("sqlalchemy.url", str(engine.url))
        command.stamp(cfg, "head")
    else:
        # Alembic is initialized; verify we're at head (warn if not)
        with engine.connect() as conn:
            mc = MigrationContext.configure(conn)
            current_rev = mc.get_current_revision()
            log.info(f"[db] alembic current revision: {current_rev}")


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
# Event CRUD
# ---------------------------------------------------------------------------


def create_event(
    camera_name: str | None = None,
    event_type: str = "motion",
    severity: str = "info",
    message: str = "",
    metadata: dict | None = None,
) -> Event:
    """Create a new Event row. Returns the created Event.

    If camera_name is given, looks up the camera_id from the cameras table.
    If the camera is not found, camera_id is set to None.
    """
    camera_id: int | None = None
    if camera_name is not None:
        with get_session() as session:
            cam = session.query(Camera).filter(Camera.name == camera_name).first()
            if cam is not None:
                camera_id = cam.id

    metadata_json = json.dumps(metadata, default=str) if metadata else "{}"

    with get_session() as session:
        event = Event(
            camera_id=camera_id,
            event_type=event_type,
            severity=severity,
            message=message,
            metadata_json=metadata_json,
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        # Eagerly load camera relationship so caller can access camera.name
        if event.camera_id is not None:
            _ = event.camera  # noqa: F841 — force load before detach
        session.expunge(event)
        return event


def list_events(
    camera_name: str | None = None,
    event_type: str | None = None,
    severity: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Event]:
    """Return events filtered by optional criteria, ordered by created_at desc.

    When camera_name is given, JOINs to cameras table to filter.
    """
    with get_session() as session:
        q = session.query(Event)

        if camera_name is not None:
            q = q.join(Event.camera).filter(Camera.name == camera_name)

        if event_type is not None:
            q = q.filter(Event.event_type == event_type)

        if severity is not None:
            q = q.filter(Event.severity == severity)

        if start is not None:
            q = q.filter(Event.created_at >= start)

        if end is not None:
            q = q.filter(Event.created_at <= end)

        q = q.order_by(Event.created_at.desc())
        q = q.offset(offset).limit(limit)

        events = q.all()
        for e in events:
            # Eagerly load camera relationship before detaching
            if e.camera_id is not None:
                _ = e.camera  # noqa: F841
            session.expunge(e)
        return events


def count_events(
    camera_name: str | None = None,
    event_type: str | None = None,
    severity: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    """Return the count of events matching the given filters."""
    with get_session() as session:
        q = session.query(sa_func.count(Event.id))

        if camera_name is not None:
            q = q.join(Event.camera).filter(Camera.name == camera_name)

        if event_type is not None:
            q = q.filter(Event.event_type == event_type)

        if severity is not None:
            q = q.filter(Event.severity == severity)

        if start is not None:
            q = q.filter(Event.created_at >= start)

        if end is not None:
            q = q.filter(Event.created_at <= end)

        result = q.scalar()
        return result if result is not None else 0


def get_event_by_id(event_id: int) -> Event | None:
    """Fetch a single Event by primary key. Returns None if not found."""
    with get_session() as session:
        event = session.query(Event).filter(Event.id == event_id).first()
        if event is not None:
            # Eagerly load camera relationship before detaching
            if event.camera_id is not None:
                _ = event.camera  # noqa: F841
            session.expunge(event)
        return event


def get_latest_event_for_camera(camera_name: str, since_seconds: int = 0) -> Event | None:
    """Return the most recent event for *camera_name* within the last *since_seconds*.

    Args:
        camera_name: Camera name to filter by (joined from cameras table).
        since_seconds: Look-back window in seconds. 0 means "all time".

    Returns:
        The most recent Event in the window, or None if no events exist.
    """
    with get_session() as session:
        q = session.query(Event).join(Event.camera).filter(Camera.name == camera_name)

        if since_seconds > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=since_seconds)
            q = q.filter(Event.created_at >= cutoff)

        q = q.order_by(Event.created_at.desc())
        event = q.first()

        if event is not None:
            if event.camera_id is not None:
                _ = event.camera  # noqa: F841
            session.expunge(event)
        return event


# ---------------------------------------------------------------------------
# Clip CRUD
# ---------------------------------------------------------------------------


def create_clip(
    event_id: int,
    camera_id: int | None,
    recording_id: str,
    path: str,
    duration_seconds: float = 0.0,
    size_bytes: int = 0,
    status: str = "pending",
    error_message: str | None = None,
) -> Clip:
    """Create a new Clip row. Returns the created Clip."""
    from .engine import get_session

    with get_session() as session:
        clip = Clip(
            event_id=event_id,
            camera_id=camera_id,
            recording_id=recording_id,
            path=path,
            duration_seconds=duration_seconds,
            size_bytes=size_bytes,
            status=status,
            error_message=error_message,
        )
        session.add(clip)
        session.commit()
        session.refresh(clip)
        session.expunge(clip)
        return clip


def get_clip(clip_id: int) -> Clip | None:
    """Fetch a single Clip by primary key. Returns None if not found."""
    from .engine import get_session

    with get_session() as session:
        clip = session.query(Clip).filter(Clip.id == clip_id).first()
        if clip is not None:
            if clip.event_id is not None:
                _ = clip.event  # noqa: F841
            if clip.camera_id is not None:
                _ = clip.camera  # noqa: F841
            session.expunge(clip)
        return clip


def list_clips_for_event(event_id: int) -> list[Clip]:
    """Return all clips for a given event, ordered by created_at desc."""
    from .engine import get_session

    with get_session() as session:
        clips = (
            session.query(Clip)
            .filter(Clip.event_id == event_id)
            .order_by(Clip.created_at.desc())
            .all()
        )
        for c in clips:
            session.expunge(c)
        return clips


def update_clip_status(clip_id: int, status: str, error_message: str | None = None) -> Clip | None:
    """Update a clip's status and optionally its error_message.

    Returns the updated Clip or None if clip_id not found.
    """
    from .engine import get_session

    with get_session() as session:
        clip = session.query(Clip).filter(Clip.id == clip_id).first()
        if clip is None:
            return None
        clip.status = status
        if error_message is not None:
            clip.error_message = error_message
        session.commit()
        session.refresh(clip)
        session.expunge(clip)
        return clip

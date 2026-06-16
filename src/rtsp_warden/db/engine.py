from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def resolve_db_url() -> str:
    """Resolve the database URL.

    Resolution order:
      1. WARDEN_DB_URL env var (supports sqlite:/// and postgresql+psycopg2:// schemes)
      2. Default: SQLite at $XDG_DATA_HOME/rtsp-warden/warden.db
                  (or ~/.local/share/rtsp-warden/warden.db on systems without XDG)

    Returns:
        Database URL string suitable for SQLAlchemy create_engine.
    """
    url = os.getenv("WARDEN_DB_URL", "").strip()
    if url:
        return url

    xdg = os.getenv("XDG_DATA_HOME", "").strip()
    if xdg:
        data_dir = Path(xdg) / "rtsp-warden"
    else:
        data_dir = Path.home() / ".local" / "share" / "rtsp-warden"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir}/warden.db"


def get_engine() -> Engine:
    """Lazily create and return the global SQLAlchemy Engine.

    SQLite: uses check_same_thread=False (so worker threads can use it).
    PostgreSQL: no special connect_args.
    """
    global _engine
    if _engine is None:
        url = resolve_db_url()
        connect_args: dict = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(url, connect_args=connect_args, echo=False, future=True)
    return _engine


def get_session() -> Session:
    """Return a new Session bound to the engine.

    Caller is responsible for closing (use `with` or try/finally).
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal()


def reset_engine() -> None:
    """Reset the global engine. Useful for tests."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None

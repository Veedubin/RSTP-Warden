"""Alembic environment for rtsp-warden.

The app runs migrations through ``rtsp_warden.db.schema.ensure_schema()``. It builds the
Alembic ``Config`` in code (no ini file) and passes an open connection in
``config.attributes["connection"]``. The ``alembic`` CLI (repo-root ``alembic.ini``) passes no
connection; this file then connects to ``resolve_db_url()``: ``WARDEN_DB_URL``, else the
default SQLite file under ``$XDG_DATA_HOME``.

This file never calls ``logging.config.fileConfig``. It used to, and that replaced the app's
log handlers and disabled every existing logger, including the one that prints the
first-start admin password. ``render_as_batch=True`` keeps SQLite ALTERs working.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from rtsp_warden.db.engine import resolve_db_url
from rtsp_warden.db.models import Base

config = context.config

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit the SQL (``alembic upgrade head --sql``) without connecting."""
    context.configure(
        url=resolve_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run on the caller's connection, or on a new one to ``resolve_db_url()``."""
    shared = config.attributes.get("connection")
    if shared is not None:
        context.configure(connection=shared, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
        return

    engine = create_engine(resolve_db_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection, target_metadata=target_metadata, render_as_batch=True
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

"""Tests for Alembic migration setup and ensure_schema integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.models import Base
from rtsp_warden.db.schema import ensure_schema

ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_INI = ROOT / "alembic.ini"
MIGRATIONS_DIR = ROOT / "migrations"
VERSIONS_DIR = MIGRATIONS_DIR / "versions"


class TestAlembicConfig:
    """Verify Alembic configuration files are valid."""

    def test_alembic_config_loads(self) -> None:
        """alembic.ini can be loaded by alembic.config.Config."""
        cfg = Config(str(ALEMBIC_INI))
        assert cfg.get_main_option("script_location") == "migrations"

    def test_alembic_env_py_imports(self) -> None:
        """migrations/env.py defines target_metadata and imports models."""
        content = (MIGRATIONS_DIR / "env.py").read_text()
        assert "target_metadata" in content
        assert "Base.metadata" in content

    def test_initial_migration_exists(self) -> None:
        """migrations/versions/0001_initial.py exists and is non-empty."""
        migration_file = VERSIONS_DIR / "0001_initial.py"
        assert migration_file.exists(), "initial migration file not found"
        content = migration_file.read_text()
        assert len(content.strip()) > 0, "initial migration file is empty"

    def test_initial_migration_has_upgrade_and_downgrade(self) -> None:
        """The migration file has def upgrade() and def downgrade()."""
        content = (VERSIONS_DIR / "0001_initial.py").read_text()
        assert "def upgrade()" in content
        assert "def downgrade()" in content


EXPECTED_TABLES = {
    "users",
    "sessions",
    "api_tokens",
    "cameras",
    "recordings",
    "events",
    "ingest_health",
}


class TestEnsureSchemaAlembic:
    """Verify ensure_schema uses Alembic correctly."""

    def test_ensure_schema_runs_on_fresh_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Call ensure_schema() on a fresh in-memory SQLite, verify all 7 tables exist."""
        db_url = f"sqlite:///{tmp_path}/fresh.db"
        monkeypatch.setenv("WARDEN_DB_URL", db_url)
        reset_engine()

        try:
            ensure_schema()
            engine = create_engine(db_url)
            insp = inspect(engine)
            tables = set(insp.get_table_names())
            # alembic_version table will also exist
            assert EXPECTED_TABLES.issubset(tables), f"missing tables: {EXPECTED_TABLES - tables}"
        finally:
            reset_engine()

    def test_ensure_schema_stamps_legacy_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Create tables with create_all, then call ensure_schema(), verify it stamps as head."""
        db_url = f"sqlite:///{tmp_path}/legacy.db"
        monkeypatch.setenv("WARDEN_DB_URL", db_url)
        reset_engine()

        try:
            # Create tables the old way (legacy)
            engine = create_engine(db_url)
            Base.metadata.create_all(bind=engine)
            insp = inspect(engine)
            tables_before = set(insp.get_table_names())
            assert "alembic_version" not in tables_before, "should not have alembic_version yet"

            # Now call ensure_schema which should stamp as head
            ensure_schema()

            insp2 = inspect(engine)
            tables_after = set(insp2.get_table_names())
            assert "alembic_version" in tables_after, (
                "alembic_version table should exist after stamping"
            )

            # Verify the revision is set
            with engine.connect() as conn:
                result = conn.execute(
                    __import__("sqlalchemy").text("SELECT version_num FROM alembic_version")
                )
                rows = result.fetchall()
                assert len(rows) == 1, "should have exactly one alembic revision"
        finally:
            reset_engine()

    def test_ensure_schema_does_not_downgrade(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Call ensure_schema() twice, second call is a no-op (no errors)."""
        db_url = f"sqlite:///{tmp_path}/nodowngrade.db"
        monkeypatch.setenv("WARDEN_DB_URL", db_url)
        reset_engine()

        try:
            ensure_schema()
            ensure_schema()  # Second call should be a no-op

            engine = create_engine(db_url)
            insp = inspect(engine)
            tables = set(insp.get_table_names())
            assert EXPECTED_TABLES.issubset(tables), f"missing tables: {EXPECTED_TABLES - tables}"
        finally:
            reset_engine()

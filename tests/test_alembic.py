"""Alembic migrations: packaging, migration 0003 (upgrade, backfill, downgrade), ensure_schema."""

from __future__ import annotations

import importlib.resources
import json
import logging
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from rtsp_warden.db import schema as schema_mod
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.models import Base
from rtsp_warden.db.schema import _alembic_config, ensure_schema

ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_INI = ROOT / "alembic.ini"
MIGRATIONS_DIR = ROOT / "src" / "rtsp_warden" / "migrations"
VERSIONS_DIR = MIGRATIONS_DIR / "versions"
HEAD = "0003_detection_events"

EXPECTED_TABLES = {"users", "sessions", "api_tokens", "events", "action_runs", "alembic_version"}
REMOVED_TABLES = {"cameras", "recordings", "ingest_health", "clips"}
NEW_EVENT_COLUMNS = {
    "camera_name",
    "label",
    "confidence",
    "zone",
    "track_id",
    "ended_at",
    "thumbnail_path",
    "clip_path",
}


def _use_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> str:
    """Point WARDEN_DB_URL (and the process-wide engine) at a new SQLite file."""
    url = f"sqlite:///{tmp_path}/{name}"
    monkeypatch.setenv("WARDEN_DB_URL", url)
    reset_engine()
    return url


def _upgrade_to(url: str, revision: str) -> None:
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            command.upgrade(_alembic_config(conn), revision)
    finally:
        engine.dispose()


def _downgrade_to(url: str, revision: str) -> None:
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            command.downgrade(_alembic_config(conn), revision)
    finally:
        engine.dispose()


def _query(url: str, sql: str) -> list[tuple]:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return [tuple(row) for row in conn.execute(text(sql))]
    finally:
        engine.dispose()


def _tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _columns(url: str, table: str) -> set[str]:
    engine = create_engine(url)
    try:
        return {col["name"] for col in inspect(engine).get_columns(table)}
    finally:
        engine.dispose()


def _seed_0002(url: str) -> None:
    """Rows as a 0002 install has them: EventSink rows with camera_id NULL, plus edge cases."""
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO cameras (id, name, main_url, sub_url) "
                    "VALUES (1, 'joined', 'rtsp://u:p@h/m', '')"
                )
            )
            rows = [
                # the cameras join wins over the message
                (1, 1, "person", "person detected on other/main (confidence=0.90)", "{}"),
                # metadata_json["camera"]
                (2, None, "motion", "manual", json.dumps({"camera": "frommeta"})),
                # the EventSink message format
                (3, None, "motion", "motion detected on yard/sub (confidence=0.50)", "{}"),
                # nothing to go on, and metadata_json is not JSON
                (4, None, "ingest_lost", "ingest lost", "not json"),
            ]
            for row_id, camera_id, event_type, message, meta in rows:
                conn.execute(
                    text(
                        "INSERT INTO events (id, camera_id, event_type, message, metadata_json) "
                        "VALUES (:id, :camera_id, :event_type, :message, :meta)"
                    ),
                    {
                        "id": row_id,
                        "camera_id": camera_id,
                        "event_type": event_type,
                        "message": message,
                        "meta": meta,
                    },
                )
            conn.execute(
                text(
                    "INSERT INTO clips (event_id, camera_id, recording_id, path, duration_seconds) "
                    "VALUES (1, 1, 'joined_main', '/x/1.mp4', 20.0)"
                )
            )
    finally:
        engine.dispose()


class TestAlembicConfig:
    """Migrations ship inside the package; the repo-root alembic.ini is only for the CLI."""

    def test_alembic_ini_points_at_the_package(self) -> None:
        cfg = Config(str(ALEMBIC_INI))
        assert cfg.get_main_option("script_location") == "rtsp_warden:migrations"

    def test_migrations_ship_inside_the_package(self) -> None:
        pkg = importlib.resources.files("rtsp_warden") / "migrations"
        assert (pkg / "env.py").is_file()
        assert (pkg / "script.py.mako").is_file()
        names = sorted(p.name for p in (pkg / "versions").iterdir() if p.name.endswith(".py"))
        assert names == ["0001_initial.py", "0002_clips_table.py", "0003_detection_events.py"]
        assert ScriptDirectory.from_config(_alembic_config()).get_current_head() == HEAD

    def test_env_py_never_configures_logging(self) -> None:
        content = (MIGRATIONS_DIR / "env.py").read_text()
        assert "fileConfig(" not in content
        assert "target_metadata" in content
        assert "Base.metadata" in content

    @pytest.mark.parametrize(
        "name", ["0001_initial.py", "0002_clips_table.py", "0003_detection_events.py"]
    )
    def test_migration_has_upgrade_and_downgrade(self, name: str) -> None:
        content = (VERSIONS_DIR / name).read_text()
        assert "def upgrade()" in content
        assert "def downgrade()" in content


class TestMigration0003:
    """Upgrade from a 0002 database: new columns, backfill, dropped tables; and back."""

    def test_upgrade_from_0002_backfills_camera_name_and_drops_dead_tables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = _use_db(tmp_path, monkeypatch, "m0003.db")
        _upgrade_to(url, "0002_clips")
        _seed_0002(url)

        _upgrade_to(url, "head")

        assert _tables(url) == EXPECTED_TABLES
        columns = _columns(url, "events")
        assert NEW_EVENT_COLUMNS <= columns
        assert "camera_id" not in columns
        assert _query(url, "SELECT id, camera_name FROM events ORDER BY id") == [
            (1, "joined"),
            (2, "frommeta"),
            (3, "yard"),
            (4, None),
        ]
        engine = create_engine(url)
        try:
            insp = inspect(engine)
            assert {ix["name"] for ix in insp.get_indexes("events")} == {
                "ix_events_camera_name",
                "ix_events_created_at",
                "ix_events_event_type",
            }
            assert insp.get_foreign_keys("events") == []
            fks = insp.get_foreign_keys("action_runs")
            assert [(fk["referred_table"], fk["constrained_columns"]) for fk in fks] == [
                ("events", ["event_id"])
            ]
        finally:
            engine.dispose()
        assert _query(url, "SELECT version_num FROM alembic_version") == [(HEAD,)]

    def test_downgrade_to_0002_and_upgrade_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = _use_db(tmp_path, monkeypatch, "roundtrip.db")
        _upgrade_to(url, "head")

        _downgrade_to(url, "0002_clips")

        tables = _tables(url)
        assert REMOVED_TABLES <= tables
        assert "action_runs" not in tables
        columns = _columns(url, "events")
        assert "camera_id" in columns
        assert not NEW_EVENT_COLUMNS & columns

        _upgrade_to(url, "head")
        assert _tables(url) == EXPECTED_TABLES

    def test_models_match_the_migrated_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = _use_db(tmp_path, monkeypatch, "models.db")
        try:
            ensure_schema()
            for table in Base.metadata.sorted_tables:
                assert _columns(url, table.name) == set(table.columns.keys()), table.name
        finally:
            reset_engine()


class TestEnsureSchemaAlembic:
    """ensure_schema: create, stamp, no-op, upgrade with backup, refuse unknown revisions."""

    def test_ensure_schema_runs_on_fresh_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = _use_db(tmp_path, monkeypatch, "fresh.db")
        try:
            ensure_schema()
            assert _tables(url) == EXPECTED_TABLES
            assert _query(url, "SELECT version_num FROM alembic_version") == [(HEAD,)]
        finally:
            reset_engine()

    def test_ensure_schema_stamps_legacy_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tables made by create_all and no alembic_version: stamped as head, not migrated."""
        url = _use_db(tmp_path, monkeypatch, "legacy.db")
        engine = create_engine(url)
        try:
            Base.metadata.create_all(bind=engine)
            assert "alembic_version" not in set(inspect(engine).get_table_names())

            ensure_schema()

            assert _query(url, "SELECT version_num FROM alembic_version") == [(HEAD,)]
        finally:
            engine.dispose()
            reset_engine()

    def test_ensure_schema_does_not_downgrade(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling ensure_schema() twice: the second call changes nothing."""
        url = _use_db(tmp_path, monkeypatch, "nodowngrade.db")
        try:
            ensure_schema()
            ensure_schema()
            assert _tables(url) == EXPECTED_TABLES
            assert list(tmp_path.glob("*.bak-*")) == []
        finally:
            reset_engine()

    def test_ensure_schema_upgrades_a_0002_database_once_and_backs_it_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review focus: serve on an existing 0002 SQLite file upgrades it once, leaves a
        .bak-0002_clips copy, and never runs the backfill twice."""
        url = _use_db(tmp_path, monkeypatch, "behind.db")
        _upgrade_to(url, "0002_clips")
        _seed_0002(url)
        warnings: list[str] = []
        monkeypatch.setattr(
            schema_mod.log, "warning", lambda msg, *args: warnings.append(msg % args)
        )
        try:
            ensure_schema()

            backup = tmp_path / "behind.db.bak-0002_clips"
            assert backup.is_file()
            backup_url = f"sqlite:///{backup}"
            assert _query(backup_url, "SELECT version_num FROM alembic_version") == [
                ("0002_clips",)
            ]
            assert "cameras" in _tables(backup_url)
            assert _query(url, "SELECT version_num FROM alembic_version") == [(HEAD,)]
            assert _query(url, "SELECT camera_name FROM events WHERE id = 3") == [("yard",)]
            assert any("0002_clips" in w and HEAD in w and str(backup) in w for w in warnings)

            engine = create_engine(url)
            with engine.begin() as conn:
                conn.execute(text("UPDATE events SET camera_name = 'renamed' WHERE id = 3"))
            engine.dispose()
            reset_engine()
            ensure_schema()

            assert _query(url, "SELECT camera_name FROM events WHERE id = 3") == [("renamed",)]
            assert sorted(p.name for p in tmp_path.glob("behind.db.bak-*")) == [
                "behind.db.bak-0002_clips"
            ]
        finally:
            reset_engine()

    def test_ensure_schema_never_overwrites_an_existing_backup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = _use_db(tmp_path, monkeypatch, "again.db")
        _upgrade_to(url, "0002_clips")
        (tmp_path / "again.db.bak-0002_clips").write_bytes(b"older backup")
        try:
            ensure_schema()
            assert (tmp_path / "again.db.bak-0002_clips").read_bytes() == b"older backup"
            assert (tmp_path / "again.db.bak-0002_clips.2").is_file()
        finally:
            reset_engine()

    def test_ensure_schema_refuses_an_unknown_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = _use_db(tmp_path, monkeypatch, "ahead.db")
        try:
            ensure_schema()
            engine = create_engine(url)
            with engine.begin() as conn:
                conn.execute(text("UPDATE alembic_version SET version_num = '9999_future'"))
            engine.dispose()
            reset_engine()

            with pytest.raises(SystemExit, match="9999_future"):
                ensure_schema()

            assert _query(url, "SELECT version_num FROM alembic_version") == [("9999_future",)]
            assert list(tmp_path.glob("*.bak-*")) == []
        finally:
            reset_engine()

    def test_ensure_schema_refuses_unbacked_non_sqlite_upgrade_without_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A database that cannot be copied first (PostgreSQL) is upgraded only with
        WARDEN_DB_UPGRADE=1, because 0003 drops tables."""
        url = _use_db(tmp_path, monkeypatch, "server.db")
        _upgrade_to(url, "0002_clips")
        monkeypatch.setattr(schema_mod, "_backup_sqlite", lambda engine, revision: None)
        monkeypatch.setattr(schema_mod, "_backend_name", lambda engine: "postgresql")
        monkeypatch.delenv("WARDEN_DB_UPGRADE", raising=False)
        try:
            with pytest.raises(SystemExit, match="WARDEN_DB_UPGRADE=1"):
                ensure_schema()
            assert _query(url, "SELECT version_num FROM alembic_version") == [("0002_clips",)]

            monkeypatch.setenv("WARDEN_DB_UPGRADE", "1")
            reset_engine()
            ensure_schema()

            assert _query(url, "SELECT version_num FROM alembic_version") == [(HEAD,)]
        finally:
            reset_engine()

    def test_ensure_schema_keeps_the_app_logging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Alembic must not replace the app's log handlers or disable existing loggers
        (that hid the first-start admin password)."""
        _use_db(tmp_path, monkeypatch, "logging.db")
        root = logging.getLogger()
        sentinel = logging.NullHandler()
        root.addHandler(sentinel)
        level = root.level
        bootstrap_log = logging.getLogger("rtsp_warden.db.bootstrap")
        try:
            ensure_schema()
            assert sentinel in root.handlers
            assert root.level == level
            assert bootstrap_log.disabled is False
        finally:
            root.removeHandler(sentinel)
            reset_engine()

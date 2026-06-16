"""Tests for install.py — run_install and InstallResult."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import inspect

from rtsp_warden.db.engine import get_engine, reset_engine
from rtsp_warden.db.schema import get_user_by_username
from rtsp_warden.install import InstallResult, run_install


def test_install_sqlite_creates_files(tmp_path: Path) -> None:
    """run_install creates target/.env and target/.env.example."""
    result = run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password="test123",
    )
    assert result.env_path.exists()
    assert result.env_example_path.exists()


def test_install_sqlite_writes_real_secrets(tmp_path: Path) -> None:
    """.env contains a 16-char alphanumeric WARDEN_ADMIN_PASSWORD."""
    result = run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password=None,  # auto-generate
    )
    env_text = result.env_path.read_text(encoding="utf-8")
    assert "WARDEN_ADMIN_PASSWORD" in env_text
    # Extract the password value
    for line in env_text.splitlines():
        if "WARDEN_ADMIN_PASSWORD" in line:
            pw = line.split("=", 1)[1].strip().strip('"')
            assert len(pw) == 16
            assert pw.isalnum()
            break


def test_install_sqlite_writes_db_url(tmp_path: Path) -> None:
    """.env contains WARDEN_DB_URL=sqlite:///..."""
    result = run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password="test123",
    )
    env_text = result.env_path.read_text(encoding="utf-8")
    assert "WARDEN_DB_URL" in env_text
    assert "sqlite:///" in env_text


def test_install_sqlite_creates_schema(tmp_path: Path) -> None:
    """The sqlite file has all 7 tables after install."""
    run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password="test123",
    )
    # Reset engine to pick up the new DB URL
    reset_engine()
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    expected = {
        "users",
        "sessions",
        "api_tokens",
        "cameras",
        "recordings",
        "events",
        "ingest_health",
    }
    assert expected.issubset(tables)


def test_install_sqlite_creates_admin_user(tmp_path: Path) -> None:
    """get_user_by_username('admin') returns admin after install."""
    run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password="test123",
    )
    user = get_user_by_username("admin")
    assert user is not None
    assert user.role == "admin"
    assert user.is_active is True


def test_install_postgres_writes_db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """db_backend='postgres', pg_password='mypass' writes the right URL."""
    import rtsp_warden.install as install_mod

    # Mock _ensure_postgres_database to avoid actual PG connection
    def mock_ensure_pg(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(install_mod, "_ensure_postgres_database", mock_ensure_pg)

    # Mock reset_engine and ensure_schema in the install module directly
    # (install.py imports them at module level, so we patch the module's references)
    def mock_reset_engine() -> None:
        pass

    def mock_ensure_schema() -> None:
        pass

    def mock_create_admin_user(username: str, password_hash: str) -> None:
        pass

    monkeypatch.setattr(install_mod, "reset_engine", mock_reset_engine)
    monkeypatch.setattr(install_mod, "ensure_schema", mock_ensure_schema)
    monkeypatch.setattr(install_mod, "create_admin_user", mock_create_admin_user)

    result = run_install(
        target_dir=tmp_path,
        db_backend="postgres",
        pg_password="mypass",
        admin_password="test123",
    )
    env_text = result.env_path.read_text(encoding="utf-8")
    assert "WARDEN_DB_URL" in env_text
    assert "postgresql+psycopg2://warden:mypass@127.0.0.1:7777/warden" in env_text


def test_install_force_overwrites(tmp_path: Path) -> None:
    """Running install twice with force=True succeeds."""
    result1 = run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password="test123",
    )
    assert result1.env_path.exists()

    # Use a different admin username to avoid duplicate user error
    result2 = run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password="test456",
        admin_username="admin2",
        force=True,
    )
    assert result2.env_path.exists()
    # Verify the password was updated
    env_text = result2.env_path.read_text(encoding="utf-8")
    assert "test456" in env_text


def test_install_no_force_raises(tmp_path: Path) -> None:
    """Running install twice without force raises FileExistsError."""
    run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password="test123",
    )
    with pytest.raises(FileExistsError):
        run_install(
            target_dir=tmp_path,
            db_backend="sqlite",
            admin_password="test456",
        )


def test_install_result_dataclass(tmp_path: Path) -> None:
    """InstallResult has the right fields populated."""
    result = run_install(
        target_dir=tmp_path,
        db_backend="sqlite",
        admin_password="test123",
    )
    assert isinstance(result, InstallResult)
    assert result.env_path == tmp_path / ".env"
    assert result.env_example_path == tmp_path / ".env.example"
    assert result.admin_username == "admin"
    assert result.admin_password == "test123"
    assert result.app_secret is not None
    assert len(result.app_secret) == 64
    assert result.db_url.startswith("sqlite:///")
    assert result.db_backend == "sqlite"


def test_install_invalid_db_backend_raises(tmp_path: Path) -> None:
    """Unknown db_backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown db_backend"):
        run_install(
            target_dir=tmp_path,
            db_backend="mysql",
            admin_password="test123",
        )


def test_install_load_dotenv_parses_file(tmp_path: Path) -> None:
    """_load_dotenv parses a simple .env file correctly."""
    from rtsp_warden.install import _load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text('WARDEN_DB_URL="sqlite:///test.db"\nWARDEN_ADMIN_PASSWORD="secret123"\n')
    result = _load_dotenv(env_file)
    assert result["WARDEN_DB_URL"] == "sqlite:///test.db"
    assert result["WARDEN_ADMIN_PASSWORD"] == "secret123"


def test_install_load_dotenv_missing_file(tmp_path: Path) -> None:
    """_load_dotenv returns empty dict for missing file."""
    from rtsp_warden.install import _load_dotenv

    result = _load_dotenv(tmp_path / "nonexistent.env")
    assert result == {}


def test_install_load_dotenv_skips_comments_and_blanks(tmp_path: Path) -> None:
    """_load_dotenv skips comments and blank lines."""
    from rtsp_warden.install import _load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text('# This is a comment\n\nKEY1="value1"\n# Another comment\nKEY2="value2"\n')
    result = _load_dotenv(env_file)
    assert result["KEY1"] == "value1"
    assert result["KEY2"] == "value2"
    assert len(result) == 2


def test_install_postgres_auto_generates_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Postgres install auto-generates pg_password when not provided."""
    import rtsp_warden.install as install_mod

    def mock_ensure_pg(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(install_mod, "_ensure_postgres_database", mock_ensure_pg)

    def mock_reset_engine() -> None:
        pass

    def mock_ensure_schema() -> None:
        pass

    def mock_create_admin_user(username: str, password_hash: str) -> None:
        pass

    monkeypatch.setattr(install_mod, "reset_engine", mock_reset_engine)
    monkeypatch.setattr(install_mod, "ensure_schema", mock_ensure_schema)
    monkeypatch.setattr(install_mod, "create_admin_user", mock_create_admin_user)

    result = run_install(
        target_dir=tmp_path,
        db_backend="postgres",
        pg_password=None,  # auto-generate
        admin_password="test123",
    )
    env_text = result.env_path.read_text(encoding="utf-8")
    assert "POSTGRES_PASSWORD" in env_text
    # Extract the password
    for line in env_text.splitlines():
        if "POSTGRES_PASSWORD" in line:
            pw = line.split("=", 1)[1].strip().strip('"')
            assert len(pw) == 16
            assert pw.isalnum()
            break

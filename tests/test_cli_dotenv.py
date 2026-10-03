import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rtsp_warden import cli
from rtsp_warden.cli import _load_dotenv, app

CONFIG = """cameras:
  - name: cam
    main_url: rtsp://${CAM_USER}:${CAM_PASS}@127.0.0.1:554/x
    record:
      enabled: false
    proxy:
      enabled: false
"""


def _isolate(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Unset ``names`` and make monkeypatch restore (or delete) them after the test."""
    for name in names:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def _setup(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(CONFIG)
    (tmp_path / ".env").write_text('CAM_USER="u"\nCAM_PASS="p"\n')
    monkeypatch.chdir(tmp_path)
    _isolate(monkeypatch, "CAM_USER", "CAM_PASS")


def _setup_config_dir(tmp_path: Path, monkeypatch) -> Path:
    """config.yaml and .env in conf/, working directory run/ (no .env there)."""
    conf_dir = tmp_path / "conf"
    run_dir = tmp_path / "run"
    conf_dir.mkdir()
    run_dir.mkdir()
    (conf_dir / "config.yaml").write_text(CONFIG)
    (conf_dir / ".env").write_text('CAM_USER="u"\nCAM_PASS="p"\n')
    monkeypatch.chdir(run_dir)
    _isolate(monkeypatch, "CAM_USER", "CAM_PASS")
    return conf_dir / "config.yaml"


def test_doctor_loads_dotenv(tmp_path: Path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["doctor", "-c", "config.yaml"])
    assert "CAM_USER" not in result.output
    assert result.exit_code == 0, result.output


def test_status_loads_dotenv(tmp_path: Path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["status", "-c", "config.yaml"])
    assert "CAM_USER" not in result.output
    assert result.exit_code == 0, result.output


def test_doctor_loads_dotenv_next_to_config(tmp_path: Path, monkeypatch):
    config = _setup_config_dir(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["doctor", "-c", str(config)])
    assert "CAM_USER" not in result.output
    assert result.exit_code == 0, result.output


def test_status_loads_dotenv_next_to_config(tmp_path: Path, monkeypatch):
    config = _setup_config_dir(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["status", "-c", str(config)])
    assert "CAM_USER" not in result.output
    assert result.exit_code == 0, result.output


def test_config_dir_env_is_read_before_cwd_env(tmp_path: Path, monkeypatch):
    conf_dir = tmp_path / "conf"
    run_dir = tmp_path / "run"
    conf_dir.mkdir()
    run_dir.mkdir()
    (conf_dir / ".env").write_text('RW_T_BOTH="from-config-dir"\n')
    (run_dir / ".env").write_text('RW_T_BOTH="from-cwd"\nRW_T_CWD_ONLY="cwd"\n')
    monkeypatch.chdir(run_dir)
    _isolate(monkeypatch, "RW_T_BOTH", "RW_T_CWD_ONLY")

    _load_dotenv(conf_dir / "config.yaml")

    assert os.environ["RW_T_BOTH"] == "from-config-dir"
    assert os.environ["RW_T_CWD_ONLY"] == "cwd"


def test_dotenv_never_overrides_existing_variables(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text('RW_T_SET="from-file"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_T_SET", "from-process")

    _load_dotenv(tmp_path / "config.yaml")

    assert os.environ["RW_T_SET"] == "from-process"


def test_env_file_in_working_directory_is_read_once(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr(cli, "_load_dotenv_file", lambda path: calls.append(path.resolve()))

    _load_dotenv(Path("config.yaml"))

    assert calls == [(tmp_path / ".env").resolve()]


def test_without_config_path_only_cwd_env_is_read(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr(cli, "_load_dotenv_file", lambda path: calls.append(path))

    _load_dotenv()

    assert calls == [Path(".env")]


def test_serve_loads_dotenv_next_to_config(tmp_path: Path, monkeypatch):
    # serve is the command that must find the CAM_<SLUG>_* variables the web UI writes.
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG)
    seen: list[Path | None] = []

    def fake_load(config_path: Path | None = None) -> None:
        seen.append(config_path)
        raise SystemExit(0)  # stop serve before it starts anything

    monkeypatch.setattr(cli, "_load_dotenv", fake_load)
    result = CliRunner().invoke(app, ["serve", "-c", str(config)])

    assert seen == [config]
    assert result.exit_code == 0, result.output

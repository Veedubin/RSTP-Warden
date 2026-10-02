from pathlib import Path

from typer.testing import CliRunner

from rtsp_warden.cli import app

CONFIG = """cameras:
  - name: cam
    main_url: rtsp://${CAM_USER}:${CAM_PASS}@127.0.0.1:554/x
    record:
      enabled: false
    proxy:
      enabled: false
"""


def _setup(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(CONFIG)
    (tmp_path / ".env").write_text('CAM_USER="u"\nCAM_PASS="p"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAM_USER", raising=False)
    monkeypatch.delenv("CAM_PASS", raising=False)


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

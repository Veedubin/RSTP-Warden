"""`rtsp-warden init-config` template: env-based credentials, optional sub stream."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rtsp_warden.cli import SAMPLE_CONFIG_YAML, app
from rtsp_warden.config import load_config

MAIN_URL = "rtsp://${CAM_USER}:${CAM_PASS}@192.168.1.50:554/Streaming/Channels/101"
COMMENTED_SUB_URL = (
    "# sub_url: rtsp://${CAM_USER}:${CAM_PASS}@192.168.1.50:554/Streaming/Channels/102"
)


def _isolate_env(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Unset each name and make monkeypatch remove it again after the test.

    setenv first so monkeypatch records the original state; `_load_dotenv` writes
    os.environ directly and would otherwise leak into later tests.
    """
    for name in names:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def _write_template(out: Path) -> Path:
    result = CliRunner().invoke(app, ["init-config", "--out", str(out)])
    assert result.exit_code == 0, result.output
    return out


def test_template_has_no_inline_credentials() -> None:
    assert "user:pass" not in SAMPLE_CONFIG_YAML
    raw = yaml.safe_load(SAMPLE_CONFIG_YAML)
    assert raw["cameras"][0]["main_url"] == MAIN_URL


def test_template_sub_url_is_commented_out() -> None:
    raw = yaml.safe_load(SAMPLE_CONFIG_YAML)
    assert "sub_url" not in raw["cameras"][0]
    assert COMMENTED_SUB_URL in SAMPLE_CONFIG_YAML


def test_template_header_explains_the_env_file() -> None:
    header = SAMPLE_CONFIG_YAML.split("cameras:", 1)[0]
    assert ".env" in header
    assert "CAM_USER=" in header
    assert "CAM_PASS=" in header
    assert "Percent-encode" in header


def test_written_template_validates_with_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAM_USER", "u")
    monkeypatch.setenv("CAM_PASS", "p")
    cfg = load_config(_write_template(tmp_path / "config.yaml"))
    cam = cfg.cameras[0]
    assert cam.name == "front"
    assert cam.main_url == "rtsp://u:p@192.168.1.50:554/Streaming/Channels/101"
    assert cam.sub_url is None
    # No sub_url: the proxy (and the frame tap) fall back to the main stream.
    assert cam.proxy.stream == "main"


def test_written_template_without_env_names_the_missing_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_env(monkeypatch, "CAM_USER", "CAM_PASS")
    out = _write_template(tmp_path / "config.yaml")
    with pytest.raises(SystemExit, match="CAM_USER"):
        load_config(out)


def test_doctor_reads_env_file_next_to_generated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """README quick start: init-config, a .env beside it, doctor run from elsewhere."""
    _isolate_env(monkeypatch, "CAM_USER", "CAM_PASS")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    out = _write_template(tmp_path / "etc" / "config.yaml")
    (out.parent / ".env").write_text('CAM_USER="u"\nCAM_PASS="p"\n', encoding="utf-8")

    result = CliRunner().invoke(app, ["doctor", "-c", str(out)])

    assert result.exit_code == 0, result.output
    assert "config loads" in result.output
    assert "CAM_USER" not in result.output

"""Shipped deployments let the web UI write config.yaml and .env (ruling R9)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
UNIT = REPO / "packaging" / "systemd" / "rtsp-warden.service"
INSTALL_SH = REPO / "packaging" / "systemd" / "install.sh"


def _unit_paths(directive: str) -> list[str]:
    """Every path listed by `directive=` lines of the systemd unit."""
    paths: list[str] = []
    for line in UNIT.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{directive}="):
            paths.extend(line.split("=", 1)[1].split())
    return paths


def test_compose_mounts_config_dir_read_write() -> None:
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    volumes = compose["services"]["warden"]["volumes"]
    config_mounts = [v for v in volumes if isinstance(v, str) and v.split(":")[1] == "/app/config"]
    assert config_mounts == ["./config:/app/config"]


def test_systemd_unit_lets_the_service_write_its_config_dir() -> None:
    read_write = _unit_paths("ReadWritePaths")
    assert "/etc/rtsp-warden" in read_write
    assert "/etc/rtsp-warden" not in _unit_paths("ReadOnlyPaths")
    assert {"/var/lib/rtsp-warden", "/var/log/rtsp-warden"} <= set(read_write)


def test_install_script_makes_config_dir_group_writable() -> None:
    lines = INSTALL_SH.read_text(encoding="utf-8").splitlines()
    config_dir_lines = [
        line
        for line in lines
        if line.startswith("install -d ") and line.endswith(" /etc/rtsp-warden")
    ]
    assert config_dir_lines == ['install -d -m 0770 -o root -g "${SERVICE_GROUP}" /etc/rtsp-warden']

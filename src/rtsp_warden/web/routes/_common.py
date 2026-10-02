"""Helpers shared by route modules: config access, config path, camera lookup, templates."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request
from starlette.templating import Jinja2Templates

from ...config import AppConfig, CameraConfig
from ..paths import TEMPLATES_DIR

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_cfg(request: Request) -> AppConfig:
    """Return the loaded AppConfig, or raise 503 when the app has none."""
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not loaded")
    return cfg


def get_config_path(request: Request) -> Path | None:
    """Return the config.yaml path for write-back, or None for in-memory configs."""
    config_path = getattr(request.app.state, "config_path", None)
    return Path(config_path) if config_path else None


def find_camera(cfg: AppConfig, name: str) -> CameraConfig | None:
    """Return the CameraConfig with this name, or None."""
    for cam in cfg.cameras:
        if cam.name == name:
            return cam
    return None

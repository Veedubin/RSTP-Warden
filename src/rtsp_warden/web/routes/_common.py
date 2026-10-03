"""Helpers shared by route modules: templates, flash messages, config access, camera lookup."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request
from starlette.responses import Response
from starlette.templating import Jinja2Templates

from ... import __version__
from ...config import AppConfig, CameraConfig
from ..flash import FLASH_COOKIE, FLASH_MAX_AGE_SECONDS, FlashLevel, encode_flash
from ..paths import TEMPLATES_DIR

# The one Jinja2Templates instance. Every route module renders through it, so a global
# registered here (the footer's app_version) is visible on every page.
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["app_version"] = __version__


def set_flash(response: Response, message: str, level: FlashLevel = "info") -> None:
    """Attach a one-shot message that the next full-page GET shows (see web/flash.py)."""
    response.set_cookie(
        key=FLASH_COOKIE,
        value=encode_flash(message, level),
        max_age=FLASH_MAX_AGE_SECONDS,
        path="/",
        httponly=True,
        samesite="lax",
    )


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

"""Web UI configuration settings.

Uses pydantic-settings so values can be overridden via environment
variables with the WARDEN_WEB_ prefix (e.g. WARDEN_WEB_PORT=9090).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    """Configuration for the FastAPI-based web UI."""

    model_config = SettingsConfigDict(env_prefix="WARDEN_WEB_", extra="ignore")

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    # Future: auth settings, recording_root, etc.

"""FastAPI application factory for the rtsp-warden web UI.

Creates a configured FastAPI app with static file serving, Jinja2
templates, auth middleware, and route handlers for dashboard, cameras,
recordings, events, and health endpoints.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import AppConfig
from .config import WebSettings
from .paths import STATIC_DIR
from .routes.alerts import router as alerts_router
from .routes.api import router as api_router
from .routes.auth import router as auth_router
from .routes.cameras import router as cameras_router
from .routes.clips import router as clips_router
from .routes.dashboard import router as dashboard_router
from .routes.events import router as events_router
from .routes.health import router as health_router
from .routes.htl import router as htl_router
from .routes.onvif import router as onvif_router
from .routes.recordings import router as recordings_router
from .routes.settings import router as settings_router
from .routes.tokens import router as tokens_router
from .routes.users import router as users_router
from .session import install_security

# Type alias for the callable that provides the live AppRuntime.
# The web UI reads camera names from runtime.cfg.cameras at render time.
RuntimeProvider = Callable[[], object]


def create_app(
    settings: WebSettings | None = None,
    cfg: AppConfig | None = None,
    runtime_provider: RuntimeProvider | None = None,
) -> FastAPI:
    """Build and return a configured FastAPI application.

    Parameters
    ----------
    settings:
        Web UI settings. Defaults to ``WebSettings()`` if not provided.
    cfg:
        Application configuration (cameras, runtime, etc). When provided,
        routes can access it via ``request.app.state.cfg``.
    runtime_provider:
        Optional callable returning the live ``AppRuntime`` instance.
        When provided, the dashboard can list camera names dynamically.
    """
    if settings is None:
        settings = WebSettings()

    app = FastAPI(
        title="rtsp-warden",
        version=__version__,
        docs_url=None,  # Disable /docs in production
        redoc_url=None,  # Disable /redoc in production
    )

    # --- Application state ---
    app.state.cfg = cfg
    app.state.runtime_provider = runtime_provider or (lambda: None)

    # --- AlertManager (lazy init — populated when cfg is set or runtime starts) ---
    from ..alerts.manager import AlertManager

    if cfg is not None and cfg.alerts.enabled:
        app.state.alert_manager = AlertManager(cfg.alerts)
    else:
        app.state.alert_manager = None

    # --- Security middleware (CSRF + context) ---
    install_security(app)

    # --- Static files ---
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

    # --- Route registration ---
    app.include_router(auth_router)
    app.include_router(dashboard_router)
    app.include_router(cameras_router)
    app.include_router(recordings_router)
    app.include_router(events_router)
    app.include_router(health_router)
    app.include_router(htl_router)
    app.include_router(api_router)
    app.include_router(users_router)
    app.include_router(tokens_router)
    app.include_router(settings_router)
    app.include_router(alerts_router)
    app.include_router(clips_router)
    app.include_router(onvif_router)

    return app

"""FastAPI application factory for the rtsp-warden web UI.

Creates a configured FastAPI app with static file serving, Jinja2
templates, auth middleware, and route handlers for dashboard, cameras,
events, and health endpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import AppConfig
from .auth_depends import LoginRequired
from .config import WebSettings
from .paths import STATIC_DIR
from .routes.actions import router as actions_router
from .routes.auth import router as auth_router
from .routes.cameras import router as cameras_router
from .routes.dashboard import router as dashboard_router
from .routes.detection import router as detection_router
from .routes.events import router as events_router
from .routes.health import router as health_router
from .routes.htl import router as htl_router
from .routes.onvif import router as onvif_router
from .routes.settings import router as settings_router
from .routes.tokens import router as tokens_router
from .routes.users import router as users_router
from .routes.zones import router as zones_router
from .session import install_security

# Type alias for the callable that provides the live AppRuntime.
# The web UI reads camera names from runtime.cfg.cameras at render time.
RuntimeProvider = Callable[[], object]


def create_app(
    settings: WebSettings | None = None,
    cfg: AppConfig | None = None,
    runtime_provider: RuntimeProvider | None = None,
    config_path: str | Path | None = None,
    runtime: object | None = None,
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
    config_path:
        Path of the YAML file the config was loaded from. Routes that edit
        camera settings write back to it. None means in-memory only.
    runtime:
        The live ``AppRuntime``. Routes that hot-reload detectors need it.
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
    app.state.config_path = str(config_path) if config_path is not None else None
    app.state.runtime = runtime

    # --- Security middleware (CSRF + context) ---
    install_security(app)

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, exc: LoginRequired) -> RedirectResponse:
        return RedirectResponse(url=f"/login?next={quote(exc.next_url, safe='/')}", status_code=303)

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
    app.include_router(detection_router)
    app.include_router(events_router)
    app.include_router(health_router)
    app.include_router(htl_router)
    app.include_router(users_router)
    app.include_router(tokens_router)
    app.include_router(settings_router)
    app.include_router(actions_router)
    app.include_router(onvif_router)
    app.include_router(zones_router)

    return app

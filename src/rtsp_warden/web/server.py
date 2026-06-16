"""Web UI server lifecycle management.

Runs uvicorn inside a daemon thread so the FastAPI app coexists with
the recorder supervisor in a single process. The thread dies
automatically when the main process exits.
"""

from __future__ import annotations

import logging
import threading

import uvicorn

from ..config import AppConfig
from .app import RuntimeProvider, create_app
from .config import WebSettings

log = logging.getLogger(__name__)


class WebUIServer:
    """Manages the uvicorn ASGI server lifecycle in a daemon thread.

    Usage::

        settings = WebSettings(host="127.0.0.1", port=8080)
        server = WebUIServer(settings, cfg=cfg, runtime_provider=lambda: rt)
        server.start()   # non-blocking; returns immediately
        print(server.url)
        ...
        server.stop()    # graceful shutdown
    """

    def __init__(
        self,
        settings: WebSettings,
        runtime_provider: RuntimeProvider,
        cfg: AppConfig | None = None,
    ) -> None:
        self._settings = settings
        self._runtime_provider = runtime_provider
        self._cfg = cfg
        self._app = create_app(settings, cfg=cfg, runtime_provider=runtime_provider)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def app(self) -> object:
        """Return the FastAPI application instance."""
        return self._app

    @property
    def url(self) -> str:
        """Return the URL where the web UI is reachable."""
        return f"http://{self._settings.host}:{self._settings.port}"

    def start(self) -> None:
        """Start uvicorn in a daemon thread. Returns immediately."""
        config = uvicorn.Config(
            app=self._app,
            host=self._settings.host,
            port=self._settings.port,
            log_level=self._settings.log_level,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
            name="warden-web-ui",
        )
        self._thread.start()
        log.info("Web UI started on %s", self.url)

    def stop(self) -> None:
        """Request a graceful shutdown of the uvicorn server."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        log.info("Web UI stopped")

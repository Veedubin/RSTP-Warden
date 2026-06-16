from __future__ import annotations

"""Dependency-free health/status server.

This module is designed to be wired in later (Bot 6) without modifying runtime code
right now. It provides:

- /healthz      : minimal JSON probe
- /status.json  : full AppStatus JSON
- /metrics      : simple Prometheus-ish text format

Acceptance demo:
  python -c "from rtsp_warden.health_server import serve_demo; serve_demo()"

No third-party dependencies.
"""

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .auth import get_current_user
from .status_model import make_empty_status, normalize_status

log = logging.getLogger(__name__)


GetStatusFn = Callable[[], Any]


class HealthServer:
    """Small HTTP server exposing health, status, and metrics.

    Parameters
    ----------
    get_status:
        Callable returning an AppStatus-like mapping or dataclass. If None, a stub
        is used that returns make_empty_status().
    bind_host:
        Interface to bind to.
    port:
        TCP port.
    auth_enabled:
        When True, /healthz, /status.json, /metrics require a valid session
        or API token. /healthz may be exempt if WARDEN_AUTH_HEALTHZ_OPEN is true.
    """

    def __init__(
        self,
        get_status: GetStatusFn | None = None,
        bind_host: str = "127.0.0.1",
        port: int = 8899,
        auth_enabled: bool = False,
    ) -> None:
        self._get_status: GetStatusFn = get_status or (lambda: make_empty_status())
        self._bind_host = bind_host
        self._port = port
        self._auth_enabled = auth_enabled

        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def bind_host(self) -> str:
        return self._bind_host

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._bind_host}:{self._port}"

    def start(self) -> None:
        """Start the server in a background thread."""

        if self._httpd is not None:
            return

        handler_cls = _make_handler(self._get_status, auth_enabled=self._auth_enabled)
        self._httpd = ThreadingHTTPServer((self._bind_host, self._port), handler_cls)

        t = threading.Thread(target=self._httpd.serve_forever, name="health-server", daemon=True)
        t.start()
        self._thread = t

        log.info("HealthServer listening on %s", self.url)

    def serve_forever(self) -> None:
        """Run the server in the foreground (blocking)."""

        if self._httpd is None:
            handler_cls = _make_handler(self._get_status, auth_enabled=self._auth_enabled)
            self._httpd = ThreadingHTTPServer((self._bind_host, self._port), handler_cls)

        log.info("HealthServer listening on %s", self.url)
        try:
            self._httpd.serve_forever(poll_interval=0.5)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the server (safe to call multiple times)."""

        if self._httpd is None:
            return

        try:
            self._httpd.shutdown()
        except Exception:
            pass

        try:
            self._httpd.server_close()
        except Exception:
            pass

        self._httpd = None
        self._thread = None


def _make_handler(
    get_status: GetStatusFn, *, auth_enabled: bool = False
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "rtsp-warden-health/0.1"

        def _check_auth(self) -> tuple[bool, str | None]:
            """Check if the request is authenticated.

            Returns (ok, error_message). If ok is False, the request should be
            rejected with 401.
            """
            if not auth_enabled:
                return True, None

            # /healthz may be exempt
            if self.path in ("/healthz", "/healthz/"):
                if os.getenv("WARDEN_AUTH_HEALTHZ_OPEN", "false").lower() in ("true", "1", "yes"):
                    return True, None

            environ = {
                "HTTP_COOKIE": self.headers.get("Cookie", ""),
                "HTTP_AUTHORIZATION": self.headers.get("Authorization", ""),
            }
            cu = get_current_user(environ)
            if cu is None:
                return False, "authentication required"
            return True, None

        def do_GET(self) -> None:  # noqa: N802
            ok, err = self._check_auth()
            if not ok:
                self.send_response(HTTPStatus.UNAUTHORIZED)
                self.send_header("WWW-Authenticate", 'Bearer realm="warden"')
                self.send_header("Content-Type", "application/json; charset=utf-8")
                body = json.dumps({"ok": False, "error": err}).encode("utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path in ("/healthz", "/healthz/"):
                self._handle_healthz()
                return
            if self.path in ("/status.json", "/status.json/"):
                self._handle_status()
                return
            if self.path in ("/metrics", "/metrics/"):
                self._handle_metrics()
                return

            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "now": time.time(), "error": f"unknown path: {self.path}"},
            )

        def _handle_healthz(self) -> None:
            try:
                _ = normalize_status(get_status())
                self._send_json(HTTPStatus.OK, {"ok": True, "now": time.time()})
            except Exception as e:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "now": time.time(), "error": str(e)},
                )

        def _handle_status(self) -> None:
            try:
                status = normalize_status(get_status())
                # Ensure the response always includes a current timestamp.
                out = dict(status)
                out.setdefault("now", time.time())
                self._send_json(HTTPStatus.OK, out)
            except Exception as e:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "now": time.time(), "error": str(e)},
                )

        def _handle_metrics(self) -> None:
            try:
                status = normalize_status(get_status())
                body = _render_metrics(status)
                self._send_text(HTTPStatus.OK, body, content_type="text/plain; version=0.0.4")
            except Exception as e:
                self._send_text(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"rtsp_warden_metrics_error 1\n# {e}\n",
                    content_type="text/plain; charset=utf-8",
                )

        def _send_json(self, code: HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, default=_json_default, indent=2).encode("utf-8")
            self.send_response(int(code))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, code: HTTPStatus, body: str, content_type: str) -> None:
            b = body.encode("utf-8")
            self.send_response(int(code))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
            # Route http.server logs through the stdlib logger.
            log.debug("health_server: " + fmt, *args)

    return Handler


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        # normalize_status handles this, but keep this here for robustness.
        try:
            from dataclasses import asdict

            return asdict(obj)
        except Exception:
            return str(obj)

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return str(obj)

    return str(obj)


def _render_metrics(status: Mapping[str, Any]) -> str:
    """Render a tiny Prometheus-ish metrics body from AppStatus-like data."""

    lines: list[str] = []
    now = float(status.get("now") or time.time())

    app_ok = 1 if bool(status.get("ok", True)) else 0
    lines.append("# HELP rtsp_warden_up 1 if the status provider reports ok")
    lines.append("# TYPE rtsp_warden_up gauge")
    lines.append(f"rtsp_warden_up {app_ok}")

    cameras = status.get("cameras") or []
    if not isinstance(cameras, list):
        cameras = []

    lines.append("# HELP rtsp_warden_cameras Number of configured cameras in status")
    lines.append("# TYPE rtsp_warden_cameras gauge")
    lines.append(f"rtsp_warden_cameras {len(cameras)}")

    lines.append("# HELP rtsp_warden_camera_up 1 if the camera is healthy")
    lines.append("# TYPE rtsp_warden_camera_up gauge")
    lines.append("# HELP rtsp_warden_stream_ingest_up 1 if the ingest process is running")
    lines.append("# TYPE rtsp_warden_stream_ingest_up gauge")
    lines.append("# HELP rtsp_warden_stream_last_frame_age_seconds Age of last seen frame")
    lines.append("# TYPE rtsp_warden_stream_last_frame_age_seconds gauge")
    lines.append("# HELP rtsp_warden_stream_last_segment_age_seconds Age of last written segment")
    lines.append("# TYPE rtsp_warden_stream_last_segment_age_seconds gauge")
    lines.append("# HELP rtsp_warden_stream_mjpeg_clients Current MJPEG client count")
    lines.append("# TYPE rtsp_warden_stream_mjpeg_clients gauge")

    # Per-camera / per-stream metrics (best-effort, tolerate missing fields).
    for cam in cameras:
        if not isinstance(cam, Mapping):
            continue
        cam_name = str(cam.get("name", ""))
        cam_ok = 1 if bool(cam.get("ok", True)) else 0

        lines.append(f"rtsp_warden_camera_up{{camera={json.dumps(cam_name)}}} {cam_ok}")

        streams = cam.get("streams") or {}
        if not isinstance(streams, Mapping):
            continue

        for stream_name, s in streams.items():
            if not isinstance(s, Mapping):
                continue

            ingest = s.get("ingest") or {}
            ingest_running = (
                1 if bool(getattr(ingest, "get", lambda *_: None)("running", False)) else 0
            )
            lines.append(
                "rtsp_warden_stream_ingest_up"
                f"{{camera={json.dumps(cam_name)},stream={json.dumps(str(stream_name))}}} {ingest_running}"
            )

            # last_frame age
            last_frame = _as_float(s.get("last_frame_at"))
            if last_frame is not None:
                age = max(0.0, now - last_frame)
                lines.append(
                    "rtsp_warden_stream_last_frame_age_seconds"
                    f"{{camera={json.dumps(cam_name)},stream={json.dumps(str(stream_name))}}} {age:.3f}"
                )

            # last_segment age
            last_seg = _as_float(s.get("last_segment_at"))
            if last_seg is not None:
                age = max(0.0, now - last_seg)
                lines.append(
                    "rtsp_warden_stream_last_segment_age_seconds"
                    f"{{camera={json.dumps(cam_name)},stream={json.dumps(str(stream_name))}}} {age:.3f}"
                )

            # mjpeg clients
            clients = s.get("mjpeg_clients")
            if isinstance(clients, int):
                lines.append(
                    "rtsp_warden_stream_mjpeg_clients"
                    f"{{camera={json.dumps(cam_name)},stream={json.dumps(str(stream_name))}}} {clients}"
                )

    lines.append("")
    return "\n".join(lines)


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def serve_demo(bind_host: str = "127.0.0.1", port: int = 8899) -> None:
    """Run a demo server in the foreground.

    This keeps the module runnable in isolation as requested.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    server = HealthServer(get_status=None, bind_host=bind_host, port=port)
    log.info("Starting demo HealthServer on %s", server.url)
    log.info("Try:")
    log.info("  %s/healthz", server.url)
    log.info("  %s/status.json", server.url)
    log.info("  %s/metrics", server.url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping demo HealthServer...")
        server.stop()


if __name__ == "__main__":
    serve_demo()

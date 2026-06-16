from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..config import CameraConfig, RuntimeConfig
from ..ffmpeg import ManagedProcess

log = logging.getLogger(__name__)


class FrameHub:
    """Thread-safe 'latest frame' store.

    This is the fanout boundary between ingest (producer) and MJPEG proxy (consumers).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._frame_id: int = 0
        self._ts: float = 0.0

    def update(self, jpeg_bytes: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg_bytes
            self._frame_id += 1
            self._ts = time.time()
            self._cond.notify_all()

    def snapshot(self) -> tuple[bytes | None, int, float]:
        with self._lock:
            return self._jpeg, self._frame_id, self._ts

    def wait_for_new(self, last_id: int, timeout: float = 2.0) -> tuple[bytes | None, int, float]:
        end = time.time() + timeout
        with self._cond:
            while self._frame_id <= last_id:
                remaining = end - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            return self._jpeg, self._frame_id, self._ts


@dataclass
class MjpegProxyServer:
    """Serve-only MJPEG-over-HTTP proxy.

    In v0.2.0 this class spawned its own FFmpeg process and parsed MJPEG frames from stdout.
    In ingest-unified mode, FFmpeg is owned by the stream ingestor; this server only reads from
    a provided FrameHub.
    """

    camera: CameraConfig
    runtime: RuntimeConfig
    hub: FrameHub

    # Optional pointer to the upstream ingest process so /healthz can report details.
    ingest_proc: ManagedProcess | None = None

    _httpd: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None

    def start(self) -> None:
        if self._httpd is not None:
            return

        bind = (self.camera.proxy.bind_host, int(self.camera.proxy.port))

        server = ThreadingHTTPServer(bind, self._make_handler())
        server.daemon_threads = True
        self._httpd = server

        t = threading.Thread(
            target=server.serve_forever, name=f"mjpeg-http:{self.camera.name}", daemon=True
        )
        self._thread = t
        t.start()
        log.info(f"[mjpeg] serving {self.camera.name} at http://{bind[0]}:{bind[1]}/mjpeg")

    def stop(self) -> None:
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

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def process(self) -> ManagedProcess | None:
        # For compatibility with older status views; this is the upstream ingest process.
        return self.ingest_proc

    def _make_handler(self):
        hub = self.hub
        cam_name = self.camera.name
        proxy_port = int(self.camera.proxy.port)
        ingest_proc = self.ingest_proc
        stderr_tail_lines = int(self.runtime.stderr_tail_lines)

        class Handler(BaseHTTPRequestHandler):
            server_version = "rtsp-warden-mjpeg/0.1"

            def log_message(self, fmt: str, *args) -> None:  # pragma: no cover
                # Avoid noisy stdlib logs; we rely on rich logging.
                log.debug("[mjpeg] " + fmt, *args)

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", ""):
                    self._handle_index()
                    return
                if self.path.startswith("/mjpeg"):
                    self._handle_mjpeg()
                    return
                if self.path.startswith("/snapshot.jpg"):
                    self._handle_snapshot()
                    return
                if self.path.startswith("/healthz"):
                    self._handle_health()
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def _handle_index(self) -> None:
                html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{cam_name}</title></head>
<body style='background:#111;color:#eee;font-family:sans-serif'>
<h3>{cam_name} (MJPEG)</h3>
<p><a href='/mjpeg'>/mjpeg</a> &middot; <a href='/snapshot.jpg'>/snapshot.jpg</a> &middot; <a href='/healthz'>/healthz</a></p>
<img src='/mjpeg' style='max-width:100%;height:auto;border:1px solid #333'/>
</body></html>"""
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle_snapshot(self) -> None:
                jpeg, _fid, ts = hub.snapshot()
                if not jpeg:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame yet")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                # Best-effort cache busting.
                self.send_header("Cache-Control", "no-store")
                if ts:
                    self.send_header("X-Frame-Timestamp", str(ts))
                self.end_headers()
                self.wfile.write(jpeg)

            def _handle_mjpeg(self) -> None:
                boundary = "frame"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

                last_id = 0
                while True:
                    jpeg, fid, _ts = hub.wait_for_new(last_id, timeout=2.0)
                    if fid == last_id:
                        # keep-alive tick
                        continue
                    last_id = fid
                    if not jpeg:
                        continue
                    try:
                        self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    except Exception:
                        break

            def _handle_health(self) -> None:
                jpeg, fid, ts = hub.snapshot()
                now = time.time()
                age = (now - ts) if ts else None

                proc_status = None
                if ingest_proc is not None:
                    proc_status = {
                        "running": ingest_proc.is_running(),
                        "pid": ingest_proc.pid(),
                        "exit_code": ingest_proc.poll(),
                        "stderr_tail": ingest_proc.stderr_tail()[-min(stderr_tail_lines, 50) :],
                    }

                payload = {
                    "camera": cam_name,
                    "proxy": {"mode": "mjpeg", "port": proxy_port},
                    "frame": {
                        "has_frame": bool(jpeg),
                        "frame_id": fid,
                        "last_frame_unix": ts or None,
                        "age_s": age,
                    },
                    "ingest": proc_status,
                }
                data = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

"""rtsp_warden.web_ui

Basic, dependency-free web UI for viewing multiple MJPEG proxies.

This module is intentionally standalone and unintegrated; a later reconvergence
stage can wire it into config/CLI.

Constraints:
  - No new dependencies
  - Uses stdlib http.server
  - UI does NOT proxy camera streams; it only embeds existing MJPEG endpoints
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = [
    "PreviewTarget",
    "WebUiServer",
]


@dataclass(frozen=True, slots=True)
class PreviewTarget:
    """A single tile in the web UI grid."""

    camera: str
    label: str
    mjpeg_url: str
    snapshot_url: str
    health_url: str

    def proxy_root_url(self) -> str:
        """Best-effort "root" URL for linking out to the per-camera proxy server."""

        try:
            u = urllib.parse.urlsplit(self.mjpeg_url)
            # Keep scheme://host:port
            return urllib.parse.urlunsplit((u.scheme, u.netloc, "/", "", ""))
        except Exception:
            return self.mjpeg_url


DEFAULT_CSS = """
:root {
  color-scheme: dark;
  --bg: #0d1117;
  --panel: #161b22;
  --panel2: #0f1621;
  --text: #c9d1d9;
  --muted: #8b949e;
  --accent: #58a6ff;
  --border: #30363d;
}

* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  background: var(--bg);
  color: var(--text);
}

header {
  position: sticky;
  top: 0;
  z-index: 10;
  background: linear-gradient(180deg, var(--bg), rgba(13,17,23,0.85));
  border-bottom: 1px solid var(--border);
  padding: 12px 16px;
}

header h1 { margin: 0; font-size: 18px; font-weight: 650; }
header .sub { margin-top: 4px; color: var(--muted); font-size: 12px; }

.wrap { padding: 16px; }

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
  align-items: start;
}

.tile {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--panel);
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}

.tile .meta {
  padding: 10px 12px;
  display: flex;
  justify-content: space-between;
  gap: 10px;
  border-bottom: 1px solid var(--border);
  background: var(--panel2);
}

.tile .meta .title {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.tile .meta .title .name { font-weight: 650; font-size: 14px; }
.tile .meta .title .label { color: var(--muted); font-size: 12px; }

.tile .meta .links {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  text-align: right;
}

a {
  color: var(--accent);
  text-decoration: none;
  font-size: 12px;
}
a:hover { text-decoration: underline; }

.tile .frame {
  display: block;
  width: 100%;
  height: auto;
  background: #000;
}

.empty {
  border: 1px dashed var(--border);
  border-radius: 12px;
  padding: 16px;
  color: var(--muted);
  background: rgba(255,255,255,0.02);
}
""".strip()


DEFAULT_INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RTSP Warden — Live Previews</title>
    <link rel="stylesheet" href="/static/style.css" />
  </head>
  <body>
    <header>
      <h1>RTSP Warden — Live Previews</h1>
      <div class="sub">MJPEG embeds (this UI does not proxy streams).</div>
    </header>
    <div class="wrap">
      {{GRID}}
    </div>
  </body>
</html>
""".strip()


def _asset_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "web_assets")


def _read_text_if_exists(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _read_bytes_if_exists(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _render_grid(targets: Iterable[PreviewTarget]) -> str:
    items = list(targets)
    if not items:
        return '<div class="empty">No preview targets configured.</div>'

    tiles = []
    for t in items:
        tiles.append(
            "\n".join(
                [
                    '<div class="tile">',
                    '  <div class="meta">',
                    '    <div class="title">',
                    f'      <div class="name">{_html_escape(t.camera)}</div>',
                    f'      <div class="label">{_html_escape(t.label)}</div>',
                    "    </div>",
                    '    <div class="links">',
                    f'      <a href="{_html_escape(t.proxy_root_url())}" target="_blank" rel="noreferrer">proxy root</a>',
                    f'      <a href="{_html_escape(t.mjpeg_url)}" target="_blank" rel="noreferrer">mjpeg</a>',
                    f'      <a href="{_html_escape(t.snapshot_url)}" target="_blank" rel="noreferrer">snapshot</a>',
                    f'      <a href="{_html_escape(t.health_url)}" target="_blank" rel="noreferrer">health</a>',
                    "    </div>",
                    "  </div>",
                    f'  <img class="frame" src="{_html_escape(t.mjpeg_url)}" loading="lazy" />',
                    "</div>",
                ]
            )
        )

    return '<div class="grid">\n' + "\n".join(tiles) + "\n</div>"


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


class WebUiServer:
    """Simple HTTP server that renders an MJPEG preview grid."""

    def __init__(
        self,
        targets: list[PreviewTarget],
        bind_host: str = "127.0.0.1",
        port: int = 8080,
        title: str = "RTSP Warden — Live Previews",
    ) -> None:
        self._targets = list(targets)
        self.bind_host = bind_host
        self.port = int(port)
        self.title = title

        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def targets(self) -> list[PreviewTarget]:
        return list(self._targets)

    def set_targets(self, targets: list[PreviewTarget]) -> None:
        # Safe for later wiring: swap list reference.
        self._targets = list(targets)

    def start(self) -> None:
        if self._server is not None:
            return

        handler_cls = self._make_handler()
        httpd = ThreadingHTTPServer((self.bind_host, self.port), handler_cls)
        # Attach a backreference for handler access.
        httpd.web_ui = self  # type: ignore[attr-defined]
        self._server = httpd

        t = threading.Thread(target=httpd.serve_forever, name="web-ui", daemon=True)
        t.start()
        self._thread = t

    def serve_forever(self) -> None:
        self.start()
        assert self._server is not None
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if self._server is None:
            return

        try:
            self._server.shutdown()
            self._server.server_close()
        finally:
            self._server = None
            self._thread = None

    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def url(self) -> str:
        return f"http://{self.bind_host}:{self.port}/"

    # ----------------------------
    # Request handling
    # ----------------------------

    def _make_handler(self):
        asset_dir = _asset_dir()
        css_path = os.path.join(asset_dir, "style.css")
        index_path = os.path.join(asset_dir, "index.html")

        class Handler(BaseHTTPRequestHandler):
            server_version = "rtsp-warden-webui/0.1"

            def log_message(self, fmt: str, *args) -> None:  # noqa: N802
                # Keep noise low; integrate with package logging later.
                return

            def do_GET(self) -> None:  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                if path == "/" or path == "":
                    self._handle_index()
                    return

                if path == "/api/targets":
                    self._handle_targets_api()
                    return

                if path == "/static/style.css":
                    self._handle_css()
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def _handle_index(self) -> None:
                web_ui = self.server.web_ui  # type: ignore[attr-defined]
                targets = web_ui.targets

                tpl = _read_text_if_exists(index_path) or DEFAULT_INDEX_HTML
                # Keep template injection minimal and safe.
                grid_html = _render_grid(targets)
                html = tpl.replace("{{GRID}}", grid_html)

                data = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle_targets_api(self) -> None:
                web_ui = self.server.web_ui  # type: ignore[attr-defined]
                payload = [asdict(t) for t in web_ui.targets]
                data = json.dumps(payload, indent=2).encode("utf-8")

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle_css(self) -> None:
                data = _read_bytes_if_exists(css_path)
                if data is None:
                    data = (DEFAULT_CSS + "\n").encode("utf-8")

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/css; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


def _demo_targets() -> list[PreviewTarget]:
    # Adjust these ports/paths to match your per-camera MJPEG proxy instances.
    return [
        PreviewTarget(
            camera="front",
            label="sub (preview)",
            mjpeg_url="http://127.0.0.1:9001/mjpeg",
            snapshot_url="http://127.0.0.1:9001/snapshot.jpg",
            health_url="http://127.0.0.1:9001/healthz",
        ),
        PreviewTarget(
            camera="garage",
            label="sub (preview)",
            mjpeg_url="http://127.0.0.1:9002/mjpeg",
            snapshot_url="http://127.0.0.1:9002/snapshot.jpg",
            health_url="http://127.0.0.1:9002/healthz",
        ),
    ]


if __name__ == "__main__":
    # Standalone demo:
    #   python -m rtsp_warden.web_ui
    # Then open: http://127.0.0.1:8080/
    srv = WebUiServer(_demo_targets(), bind_host="127.0.0.1", port=8080)
    print(f"Web UI running at: {srv.url()}")
    srv.serve_forever()

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import CameraConfig, RuntimeConfig
from ..ffmpeg import ManagedProcess

log = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class MediaMTXProxyServer:
    """RTSP proxy via MediaMTX.

    This runs one MediaMTX instance per camera/port as requested.
    """

    camera: CameraConfig
    runtime: RuntimeConfig

    _proc: ManagedProcess | None = None
    _config_path: Path | None = None

    def publish_url(self) -> str:
        """Local RTSP URL that ingest FFmpeg should publish into."""
        return f"rtsp://127.0.0.1:{self.camera.proxy.port}/{self.camera.proxy.path}"

    def start(self) -> None:
        # Write per-camera config under workspace_dir
        cfg_dir = self.runtime.workspace_dir / "proxy" / self.camera.name
        _ensure_dir(cfg_dir)
        cfg_path = cfg_dir / "mediamtx.yml"
        self._config_path = cfg_path

        bind_host = self.camera.proxy.bind_host
        port = self.camera.proxy.port

        # In ingest-unified mode, MediaMTX must NOT pull from the upstream camera.
        # Instead, it exposes an RTSP path that accepts a local publisher (ffmpeg ingest)
        # and serves any connected clients.
        mtx_cfg = {
            "logLevel": "info",
            "logDestinations": ["stdout"],
            "rtsp": True,
            "rtspAddress": f"{bind_host}:{port}",
            # Prefer TCP to avoid UDP/NAT issues for home cameras.
            "rtspTransports": ["tcp"],
            "paths": {
                self.camera.proxy.path: {
                    # Accept a publisher (ffmpeg) and serve it to clients.
                    "source": "publisher",
                    "rtspTransport": "tcp",
                }
            },
        }

        cfg_path.write_text(yaml.safe_dump(mtx_cfg, sort_keys=False), encoding="utf-8")

        cmd = [self.runtime.mediamtx_path, str(cfg_path)]
        self._proc = ManagedProcess(
            name=f"proxy:rtsp:{self.camera.name}",
            args=cmd,
            stdout=None,
            stderr=subprocess.PIPE,
            stderr_tail_lines=self.runtime.stderr_tail_lines,
        )
        try:
            self._proc.start()
        except Exception as e:
            log.error(f"[mediamtx] failed to start for {self.camera.name}: {e!r}")
            self._proc = None
            return

        log.info(
            f"[mediamtx] {self.camera.name}: serving rtsp://{bind_host}:{port}/{self.camera.proxy.path}"
        )

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc = None

    def process(self) -> ManagedProcess | None:
        return self._proc

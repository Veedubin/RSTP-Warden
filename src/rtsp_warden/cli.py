from __future__ import annotations

import importlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .app import AppRuntime
from .config import AppConfig, load_config
from .ffmpeg import which_or_raise
from .frame_tap import FrameTapDispatcher
from .logging_utils import setup_logging
from .status_model import redact_rtsp_url
from .web.config import WebSettings
from .web.server import WebUIServer
from .web_ui import PreviewTarget, WebUiServer

app = typer.Typer(add_completion=False, help="rtsp-warden: record RTSP streams + proxy locally")
console = Console()


SAMPLE_CONFIG_YAML = """cameras:
  - name: front
    main_url: rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101
    sub_url: rtsp://user:pass@192.168.1.50:554/Streaming/Channels/102

    record:
      enabled: true
      output_dir: ./recordings

      # Main stream: TS container (NVR-grade default)
      main:
        enabled: true
        container: ts
        chunk_seconds: 300
        rtsp_transport: tcp

      # Sub stream: TS container (NVR-grade default)
      sub:
        enabled: true
        container: ts
        chunk_seconds: 300
        rtsp_transport: tcp

      retention:
        max_days: 7
        max_gb: 50
        keep_last_n: 10
        cleanup_interval_seconds: 300

    proxy:
      enabled: true
      mode: mjpeg         # mjpeg | rtsp
      stream: sub         # main | sub
      bind_host: 0.0.0.0
      port: 9001

      # RTSP (MediaMTX) options (mode=rtsp)
      path: live
      source_on_demand: true  # NOTE: best-effort; unified ingest publishes while running.

      # MJPEG options (mode=mjpeg)
      fps: 7
      scale_width: 0      # 0 disables scaling

runtime:
  ffmpeg_path: ffmpeg
  mediamtx_path: mediamtx
  ffmpeg_loglevel: warning
  workspace_dir: ./workspace

  auto_restart: true
  restart_backoff_min_s: 1
  restart_backoff_max_s: 60
  restart_backoff_factor: 2

  stderr_tail_lines: 200
  status_interval_s: 15
"""


def _load_cfg(config: Path) -> AppConfig:
    return load_config(config)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Parse a simple KEY="VALUE" .env file and set env vars. No python-dotenv dep."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Only set if not already defined in environment (env vars take precedence)
        if key and key not in os.environ:
            os.environ[key] = value


def _port_is_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
        return True
    except OSError:
        return False


def _require_binaries(cfg: AppConfig) -> None:
    which_or_raise(cfg.runtime.ffmpeg_path)
    needs_mediamtx = any(c.proxy.enabled and c.proxy.mode == "rtsp" for c in cfg.cameras)
    if needs_mediamtx:
        which_or_raise(cfg.runtime.mediamtx_path)


def _import_consumer(spec: str) -> Any:
    """Load a FrameConsumer instance from a spec.

    Supported:
      - "motion-demo" (built-in demo consumer)
      - "module:ClassName" (ClassName must be instantiable with no args)
    """
    spec = spec.strip()
    if spec in {"motion-demo", "motion_demo", "demo"}:
        from .consumers.motion_demo import MotionHeuristicConsumer

        return MotionHeuristicConsumer()

    if ":" not in spec:
        raise typer.BadParameter("consumer must be 'motion-demo' or 'module:ClassName'")

    mod_name, obj_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, obj_name, None)
    if obj is None:
        raise typer.BadParameter(f"cannot import {obj_name!r} from {mod_name!r}")

    try:
        return obj()  # type: ignore[misc]
    except TypeError as e:
        raise typer.BadParameter(f"cannot instantiate {spec!r} with no args: {e}") from e


def _build_frame_tap_dispatcher(consumer_specs: list[str]) -> FrameTapDispatcher | None:
    """Build a FrameTapDispatcher from consumer specs. Returns None if no specs."""
    if not consumer_specs:
        return None

    consumers = [_import_consumer(s) for s in consumer_specs]
    return FrameTapDispatcher(consumers=consumers)


def _latest_file_info(dir_path: Path) -> tuple[float, str]:
    """Return (mtime, path) for newest file in dir_path; (0.0, '') if none."""
    try:
        import os

        newest_mtime = 0.0
        newest_path = ""
        with os.scandir(dir_path) as it:
            for ent in it:
                if not ent.is_file():
                    continue
                try:
                    st = ent.stat()
                except FileNotFoundError:
                    continue
                if st.st_mtime > newest_mtime:
                    newest_mtime = float(st.st_mtime)
                    newest_path = str(Path(dir_path) / ent.name)
        return newest_mtime, newest_path
    except FileNotFoundError:
        return 0.0, ""
    except Exception:
        return 0.0, ""


def _proc_status(role: str, name: str, proc: Any) -> dict[str, Any]:
    if proc is None:
        return {"role": role, "name": name, "running": False}

    try:
        running = bool(proc.is_running())
    except Exception:
        running = False

    pid: int | None
    try:
        pid = proc.pid()
    except Exception:
        pid = None

    out: dict[str, Any] = {"role": role, "name": name, "running": running}
    if pid is not None:
        out["pid"] = pid

    # best-effort stderr tail
    try:
        tail = proc.stderr_tail()
        if tail:
            out["stderr_tail"] = tail
    except Exception:
        pass

    # args are often useful, but may contain credentials. Redact URLs opportunistically.
    try:
        argv = list(getattr(proc, "args", []) or [])
        redacted: list[str] = []
        for a in argv:
            if isinstance(a, str) and "rtsp://" in a:
                redacted.append(redact_rtsp_url(a))
            else:
                redacted.append(a)
        if redacted:
            out["argv"] = redacted
    except Exception:
        pass

    return out


def build_status(rt: AppRuntime, cfg: AppConfig, version: str = __version__) -> dict[str, Any]:
    now = time.time()
    cameras: list[dict[str, Any]] = []
    errors: list[str] = []

    for cam_rt in rt.cameras:
        cam = cam_rt.camera
        cam_ok = True
        streams: dict[str, Any] = {}

        for ing in cam_rt.recorder.processes():
            stream_name = str(ing.stream_name)
            st: dict[str, Any] = {"stream": stream_name}

            st["source_url"] = redact_rtsp_url(ing.upstream_url)

            # ingest proc
            st["ingest"] = _proc_status("ffmpeg_ingest", f"{cam.name}/{stream_name}", ing.proc)

            # recording status (best-effort)
            st["record_enabled"] = bool(ing.record_cfg and ing.record_cfg.enabled)
            if ing.record_cfg and ing.record_output_dir:
                rec_dir = Path(ing.record_output_dir) / cam.name / stream_name
                mtime, newest = _latest_file_info(rec_dir)
                if mtime > 0:
                    st["last_segment_at"] = mtime
                    st["last_segment_path"] = newest

            # MJPEG status (best-effort)
            if ing.mjpeg_hub is not None:
                st["mjpeg_enabled"] = True
                try:
                    _jpeg, _fid, ts = ing.mjpeg_hub.snapshot()
                    if ts:
                        st["last_frame_at"] = float(ts)
                except Exception:
                    pass
            else:
                st["mjpeg_enabled"] = False

            # RTSP publish status (best-effort)
            if ing.rtsp_publish_url:
                st["rtsp_publish_enabled"] = True
                st["rtsp_publish_url"] = ing.rtsp_publish_url
            else:
                st["rtsp_publish_enabled"] = False

            # derive stream ok
            if st.get("ingest", {}).get("running") is False and (
                st.get("record_enabled")
                or st.get("mjpeg_enabled")
                or st.get("rtsp_publish_enabled")
            ):
                cam_ok = False

            streams[stream_name] = st

        # proxy status (best-effort)
        proxy_obj = cam_rt.proxy
        if proxy_obj is not None:
            if cam.proxy.mode == "rtsp":
                # MediaMTX process wrapper
                proc = getattr(proxy_obj, "_proc", None)
                streams.setdefault(cam.proxy.stream, {})
                streams[cam.proxy.stream]["rtsp_server"] = _proc_status(
                    "mediamtx", f"{cam.name}/mediamtx", proc
                )
            elif cam.proxy.mode == "mjpeg":
                # MJPEG is an HTTP thread, not a process
                running = False
                try:
                    running = bool(proxy_obj.is_running())
                except Exception:
                    pass
                streams.setdefault(cam.proxy.stream, {})
                streams[cam.proxy.stream]["mjpeg_http_up"] = bool(running)

        cameras.append(
            {
                "name": cam.name,
                "ok": bool(cam_ok),
                "streams": streams,
            }
        )

    ok = all(c.get("ok", True) for c in cameras) if cameras else True

    return {
        "ok": bool(ok),
        "now": float(now),
        "version": str(version),
        "cameras": cameras,
        "errors": errors,
    }


@app.command()
def init_config(
    out: Path = typer.Option(
        Path("config.yaml"), "--out", "-o", help="Where to write the sample config"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite if the output already exists"),
) -> None:
    """Write a starter config.yaml you can edit."""
    if out.exists() and not force:
        raise typer.Exit(code=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(SAMPLE_CONFIG_YAML, encoding="utf-8")
    console.print(f"Wrote sample config to: {out}")


@app.command()
def install(
    target_dir: Path = typer.Option(
        Path("."), "--target", "-t", help="Where to write .env and .env.example"
    ),
    db: str = typer.Option("sqlite", "--db", help="Database backend: sqlite | postgres"),
    pg_host: str = typer.Option("127.0.0.1", "--pg-host", help="PostgreSQL host"),
    pg_port: int = typer.Option(7777, "--pg-port", help="PostgreSQL port"),
    pg_user: str = typer.Option("warden", "--pg-user", help="PostgreSQL user"),
    pg_password: str | None = typer.Option(
        None, "--pg-password", help="PostgreSQL password (generated if not set)"
    ),
    pg_database: str = typer.Option("warden", "--pg-database", help="PostgreSQL database name"),
    admin_username: str = typer.Option("admin", "--admin-username", help="Initial admin username"),
    admin_password: str | None = typer.Option(
        None, "--admin-password", help="Initial admin password (generated if not set)"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing .env files"),
) -> None:
    """First-run setup: writes .env, creates database schema, creates admin user."""
    from .install import run_install

    try:
        result = run_install(
            target_dir=target_dir,
            db_backend=db,
            pg_host=pg_host,
            pg_port=pg_port,
            pg_user=pg_user,
            pg_password=pg_password,
            pg_database=pg_database,
            admin_username=admin_username,
            admin_password=admin_password,
            force=force,
        )
    except FileExistsError as e:
        console.print(f"[red]✗[/red] {e}")
        console.print("Use --force to overwrite existing files.")
        raise typer.Exit(code=2) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Install failed: {e}")
        raise typer.Exit(code=1) from e

    console.print()
    console.print(f"[green]✓[/green] Database: {result.db_backend} at {result.db_url}")
    console.print("[green]✓[/green] Schema created (7 tables)")
    console.print("[green]✓[/green] Admin user created")
    console.print()
    console.print("─" * 60)
    console.print()
    console.print(f"  [bold]Username:[/bold]  {result.admin_username}")
    console.print(f"  [bold]Password:[/bold]  {result.admin_password}  [dim](stored in .env)[/dim]")
    console.print()
    console.print("[yellow]⚠ Save the password now — it will not be shown again.[/yellow]")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  rtsp-warden init-config")
    console.print("  rtsp-warden doctor -c config.yaml")
    console.print("  rtsp-warden run -c config.yaml")
    console.print()


@app.command()
def doctor(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, dir_okay=False, help="Path to YAML config"
    ),
) -> None:
    """Validate config + check binaries + check proxy ports are available."""
    cfg = _load_cfg(config)

    table = Table(title="rtsp-warden doctor")
    table.add_column("Check")
    table.add_column("Result")

    # config validation
    table.add_row("config loads", "OK")

    # binaries
    try:
        ff = which_or_raise(cfg.runtime.ffmpeg_path)
        table.add_row("ffmpeg", f"OK ({ff})")
    except Exception as e:
        table.add_row("ffmpeg", f"FAIL ({e})")

    needs_mediamtx = any(c.proxy.enabled and c.proxy.mode == "rtsp" for c in cfg.cameras)
    if needs_mediamtx:
        try:
            mt = which_or_raise(cfg.runtime.mediamtx_path)
            table.add_row("mediamtx", f"OK ({mt})")
        except Exception as e:
            table.add_row("mediamtx", f"FAIL ({e})")
    else:
        table.add_row("mediamtx", "N/A (no rtsp proxies)")

    # ports
    for cam in cfg.cameras:
        if not cam.proxy.enabled:
            continue
        host = str(cam.proxy.bind_host)
        port = int(cam.proxy.port)
        free = _port_is_free(host, port)
        table.add_row(f"port {cam.name}:{port}", "OK" if free else "IN USE")

    console.print(table)


@app.command()
def serve(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, dir_okay=False, help="Path to YAML config"
    ),
    verbosity: str = typer.Option(
        "info", "--verbosity", "-v", help="Logging verbosity: error|warning|info|debug"
    ),
    frame_consumer: list[str] = typer.Option(
        [],
        "--frame-consumer",
        help="Enable frame tap consumers. Use 'motion-demo' or 'module:ClassName'. Can be repeated.",
    ),
    detectors: bool = typer.Option(
        True, "--detectors/--no-detectors", help="Enable detector framework"
    ),
    web: bool = typer.Option(
        True, "--web/--no-web", help="Enable web UI (includes /healthz, /status.json, /metrics)"
    ),
    web_host: str = typer.Option("127.0.0.1", "--web-host", help="Web UI bind host"),
    web_port: int = typer.Option(
        8080, "--web-port", help="Web UI port (health endpoints are served here too)"
    ),
) -> None:
    """Run the supervisor: recording + proxying + web UI + health endpoints.

    The web UI (FastAPI) absorbs the standalone health server from v0.3.0.
    Health endpoints /healthz, /status.json, and /metrics are now served
    on the same port as the web UI.
    """
    _load_dotenv()
    setup_logging(verbosity=verbosity)  # type: ignore[arg-type]
    cfg = _load_cfg(config)
    _require_binaries(cfg)

    # Build a single dispatcher and let the ingest layer feed it (no monkey-patching).
    dispatcher = _build_frame_tap_dispatcher(frame_consumer)
    rt = AppRuntime(cfg=cfg, frame_tap_dispatcher=dispatcher, detectors_enabled=detectors)
    rt.build()

    ws: WebUIServer | None = None
    if web:
        if not _port_is_free(web_host, web_port):
            raise typer.Exit(code=2)
        web_settings = WebSettings(host=web_host, port=web_port)
        ws = WebUIServer(settings=web_settings, cfg=cfg, runtime_provider=lambda: rt)
        ws.start()
        console.print(f"Web UI: {ws.url}")
        console.print(f"  Health: {ws.url}/healthz, {ws.url}/status.json, {ws.url}/metrics")
        if os.getenv("WARDEN_AUTH_ENABLED", "true").lower() in ("true", "1", "yes"):
            console.print("  Auth: enabled (WARDEN_AUTH_ENABLED=true)")

    rt.start()
    try:
        rt.run_forever()
    finally:
        rt.stop_all()
        if ws is not None:
            ws.stop()


# Deprecated alias for the v0.3.0 `run` command.
# Hidden from --help but still works for backward compatibility.
@app.command(name="run", hidden=True, deprecated=True)
def run_command(  # noqa: ARG001
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, dir_okay=False, help="Path to YAML config"
    ),
    verbosity: str = typer.Option(
        "info", "--verbosity", "-v", help="Logging verbosity: error|warning|info|debug"
    ),
    health: bool = typer.Option(
        True,
        "--health/--no-health",
        help="[DEPRECATED] Health endpoints are now part of the web UI.",
    ),
    health_host: str = typer.Option(
        "127.0.0.1", "--health-host", help="[DEPRECATED] Use --web-host instead."
    ),
    health_port: int = typer.Option(
        8899, "--health-port", help="[DEPRECATED] Use --web-port instead."
    ),
    frame_consumer: list[str] = typer.Option(
        [],
        "--frame-consumer",
        help="Enable frame tap consumers. Use 'motion-demo' or 'module:ClassName'. Can be repeated.",
    ),
    web: bool = typer.Option(True, "--web/--no-web", help="Enable web UI"),
    web_host: str = typer.Option("127.0.0.1", "--web-host", help="Web UI bind host"),
    web_port: int = typer.Option(8080, "--web-port", help="Web UI port"),
) -> None:
    """[DEPRECATED] Use `serve` instead. The health server has been absorbed into the web UI."""
    console.print(
        "[yellow]Warning:[/yellow] `rtsp-warden run` is deprecated. Use `rtsp-warden serve` instead."
    )
    console.print("  The standalone health server has been absorbed into the web UI.")
    console.print("  --health, --health-host, --health-port are ignored.")
    console.print()
    # Forward to serve
    serve(
        config=config,
        verbosity=verbosity,
        frame_consumer=frame_consumer,
        web=web,
        web_host=web_host,
        web_port=web_port,
    )


@app.command()
def ui(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, dir_okay=False, help="Path to YAML config"
    ),
    bind_host: str = typer.Option("127.0.0.1", "--host", help="Web UI bind host"),
    port: int = typer.Option(8080, "--port", help="Web UI port"),
    embed_host: str = typer.Option(
        "127.0.0.1",
        "--embed-host",
        help="Host to use when building MJPEG URLs (usually where rtsp-warden is running)",
    ),
    title: str = typer.Option("RTSP Warden — Live Previews", "--title", help="Page title"),
) -> None:
    """Start the minimal MJPEG grid web UI.

    Note: this UI only embeds existing MJPEG endpoints. Run `rtsp-warden run ...` separately.
    """
    cfg = _load_cfg(config)

    targets: list[PreviewTarget] = []
    for cam in cfg.cameras:
        if not cam.proxy.enabled or cam.proxy.mode != "mjpeg":
            continue
        p = int(cam.proxy.port)
        targets.append(
            PreviewTarget(
                camera=cam.name,
                label=str(cam.proxy.stream),
                mjpeg_url=f"http://{embed_host}:{p}/mjpeg",
                snapshot_url=f"http://{embed_host}:{p}/snapshot.jpg",
                health_url=f"http://{embed_host}:{p}/healthz",
            )
        )

    if not targets:
        console.print("No MJPEG proxies found in config (proxy.mode must be 'mjpeg').")
        raise typer.Exit(code=2)

    if not _port_is_free(bind_host, port):
        raise typer.Exit(code=2)

    ui_server = WebUiServer(targets=targets, bind_host=bind_host, port=int(port), title=title)
    console.print(f"Web UI: {ui_server.url()}")
    ui_server.serve_forever()


@app.command()
def status(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, dir_okay=False, help="Path to YAML config"
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="If set, starts the runtime briefly to populate process states (may open RTSP connections).",
    ),
) -> None:
    """Print a single JSON status snapshot to stdout."""
    cfg = _load_cfg(config)
    rt = AppRuntime(cfg=cfg)
    rt.build()

    if live:
        setup_logging(verbosity="warning")
        _require_binaries(cfg)
        rt.start()
        time.sleep(0.25)

    try:
        payload = build_status(rt, cfg, version=__version__)
        console.print_json(json.dumps(payload))
    finally:
        if live:
            rt.stop_all()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)

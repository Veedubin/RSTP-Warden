# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rtsp-warden` is a self-hosted NVR for RTSP cameras: one Python process that runs
ffmpeg per camera stream, records `.ts` segments, runs OpenCV detectors in-process,
and serves a FastAPI web UI (htmx + Alpine + Pico, no JS build step). Package
version is 1.3.0 (`pyproject.toml` + `src/rtsp_warden/__init__.py`); the directory
name `rtsp-warden_v0.2.0` is historical. Remote: `Veedubin/RSTP-Warden` (public, MIT).

The README was rewritten on 2026-10-02 to match the code; when they disagree, trust the
code and fix the README. This file covers what the README does not say.

## Commands

```bash
uv sync                          # installs runtime (incl. onnxruntime CPU) + dev group (pytest, pytest-cov, pytest-asyncio, onnx)
uv run pytest                    # ~820 tests, ~70s, fully offline (no ffmpeg, cameras, or network)
uv run pytest tests/test_X.py    # one file
uv run pytest -k "pattern"       # by name
uv run ruff check src/ tests/    # 10 pre-existing E501 in v0.x files are the accepted baseline (see below)
uv run ruff format --check src/ tests/
uv run mypy src/                 # informal only: ~35 errors, not a gate
```

`ruff` and `mypy` are not in the dev group; `uv run` finds them on the global PATH.

**ONNX Runtime and the GPU venv.** `onnxruntime` (CPU) is a core dependency, `onnxruntime-gpu[cuda,cudnn]` is the
`gpu` extra, and `onnx` (test graphs only) is in the dev group. The CPU and GPU builds install the same `onnxruntime/`
package, so never let both into one venv. Switch to GPU with
`uv sync --extra gpu --no-install-package onnxruntime --reinstall-package onnxruntime-gpu`, then use
`uv run --no-sync ...` (or `UV_NO_SYNC=1`): a plain `uv run` reinstalls the CPU build over it. Back to CPU:
`uv sync --reinstall-package onnxruntime`. `uv run --no-sync pytest -m gpu` checks that CUDA really loads, and
`tests/test_onnxruntime_install.py` fails when both builds are installed. The Python floor is 3.11 while ruff
`target-version` stays `py310` on purpose (py311 adds 83 UP017/UP042 errors).

**Lint baseline.** The 10 E501 errors live in `app.py`, `cli.py`, `proxy/mjpeg.py`,
`recorder.py`, `retention.py`, and `tests/test_frame_tap_wiring.py`. The gate is "no new ruff errors"; do not reformat
those files just to clear them unless asked.

**Running the app locally** (needs `ffmpeg` on PATH; `mediamtx` only for `proxy.mode: rtsp`):

```bash
uv run rtsp-warden init-config --out config.yaml
uv run rtsp-warden doctor -c config.yaml
uv run rtsp-warden serve -c config.yaml --web --web-port 8080
```

`serve`, `doctor` and `status` load `.env` from the current working directory. `serve` runs `ensure_schema()`
and creates the first admin user when the users table is empty (`db/bootstrap.py`),
so `install` is optional. `config.yaml` strings may reference `${ENV_VAR}`; a missing
variable is a startup error. The `run` alias and the stdlib `ui` grid were removed.

**Migrations.** Alembic, `src/rtsp_warden/migrations/versions/000N_slug.py` (inside the
package, so wheels and Docker images ship them), revision ids like `0003_detection_events`
(at most 32 characters). Add a model in `db/models.py` plus a hand-written migration (SQLite
ALTERs need `op.batch_alter_table`); if it is a new table, add it to `EXPECTED_TABLES` in
`tests/test_alembic.py`. The repo-root `alembic.ini` is only for the `alembic` CLI
(`uv run alembic history`); the app builds its Alembic config in code.

**Version bump** touches three places: `pyproject.toml`, `src/rtsp_warden/__init__.py`,
and the version assertion in `tests/test_admin.py`.

**Docker.** `docker compose up -d` builds `Dockerfile.distroless` (default, no shell).
`Dockerfile` is the slim debuggable variant. Both run
`rtsp-warden serve -c /app/config/config.yaml --web --web-port 8080`; the bind host comes
from `WARDEN_WEB_HOST` (compose sets `0.0.0.0`; the default is `127.0.0.1`).

## Architecture

### Process model (`cli.py` → `app.py` → `web/server.py`)

`serve` builds one `AppRuntime` (`app.py`), starts `WebUIServer` (uvicorn in a daemon
thread), then runs `AppRuntime.run_forever()` on the main thread: a 0.5s supervisor
tick that runs retention sweeps and restarts dead ffmpeg / MediaMTX / MJPEG-proxy
processes with per-camera `ExponentialBackoff`. Health endpoints (`/healthz`,
`/status.json`, `/metrics`, plus the HTML `/health` and htmx `/health/partial`) are
FastAPI routes in `web/routes/health.py`.

### Ingest: one ffmpeg per (camera, stream)

`CameraRecorder` (`recorder.py`) owns a `StreamIngestor` for `main` and, when
`sub_url` is set, `sub`; without a sub stream the proxy and frame tap use main. Each
ingestor spawns a single multi-output ffmpeg built by `build_ffmpeg_ingest_cmd`
(`ffmpeg.py`): segment recording (`.ts`, `-c copy`), MJPEG to stdout for the proxy
`FrameHub`, RTSP publish to MediaMTX, and the **frame tap**: a low-res MJPEG stream to
`pipe:<fd>`, the write end of an `os.pipe` handed over via `pass_fds` (which keeps the
parent's fd number; the parent closes its copy right after the spawn). Only the ingestor of
`cam.proxy.stream` carries the tap. A reader thread owns the read end, splits it on JPEG
SOI/EOI markers and calls the camera's own `FrameTapDispatcher` (`CameraRuntime.dispatcher`).
`mode: event` recording is a 1s polling thread on the `events` table that starts/stops the
ingestors.

`.ts` is the deliberate container choice: MP4 segmenting failed on cameras with
missing PPS / non-monotonic timestamps (see `docs/archive/CONTEXT_RTSP_WARDEN_DROP_MP4_USE_TS.md`).
Do not reintroduce MP4 segment recording. `RTSP_WARDEN_RECORD_TRANSCODE` and friends
in `ffmpeg.py` are env-only escape hatches, not config fields.

### Detector pipeline (`frame_tap.py` → `detectors/`)

```
CameraRuntime.dispatcher → DetectorRunner.on_frame (one per camera; queue 8, drop-oldest, 1 worker)
  → cv2.imdecode → apply_masks (privacy polygons) → each Detector.process(frame_bgr, ts_unix)
  → filter_by_roi → GridMask.filter_detections → result sinks → EventSink → events table
```

The real `Detector` protocol (`detectors/base.py`) is `name`, `kind`, `setup()`,
`process(frame_bgr, ts_unix) -> list[Detection]`, `teardown()`. `DetectorType` is the literal
`motion | person | vehicle | dnn | custom`; builtins are lazily imported by
`detectors/registry.py`. `build_detectors_for_camera` is the single entry point that
applies camera-level `sensitivity` (`detectors/sensitivity.py`) and `detect_classes`
(`detectors/class_filter.py`) on top of per-detector specs, and builds grid masks
from `camera.zones`.

Hot reload is `AppRuntime.rebuild_camera_detectors(name)`: rebuilds the bundle from
the in-memory `CameraConfig`, sets the new runner up, swaps it under `_detector_lock` onto
that camera's own dispatcher (followed by any `--frame-consumer` consumers), then tears the
old runner down. Nothing else is restarted.

### Web layer (`web/`)

`create_app(settings, cfg, runtime_provider)` in `web/app.py` sets `app.state.cfg`,
`app.state.runtime_provider`, and `app.state.alert_manager`, then includes one router
per file in `web/routes/`. Routes read everything through `getattr(request.app.state, ...)`
so the app also works with no config (tests, `--no-web` paths).

**Wiring.** `cli.serve` passes `config_path` and `runtime` into `WebUIServer`, which
hands them to `create_app`, which sets `app.state.config_path` and `app.state.runtime`.
Tests pass them to `create_app` directly. Shared route helpers (`get_cfg`,
`get_config_path`, `find_camera`, `templates`) live in `web/routes/_common.py`.

**CSRF.** `CSRFMiddleware` accepts the token from the `X-CSRF-Token` header, the
`csrf_token` query parameter, or a `csrf_token` form field. `static/js/warden.js` adds
the header to every htmx request, so templates need no per-form `hx-headers`. Plain
HTML forms carry a hidden `csrf_token` input. Unauthenticated browser requests are
redirected to `/login?next=`; htmx requests get a 401 with `HX-Redirect`; API calls get
a plain 401.

**Live status and preview.** `web/services/cameras.live_status` derives
running / restarting / failed / idle from the `CameraRuntime` (the supervisor stores
`next_restart_at` and `last_error` on it). The preview is served same-origin from the
in-process `FrameHub` at `/cameras/{name}/live.mjpeg` and `/snapshot.jpg`
(`web/services/preview.py`); nothing links to the 127.0.0.1 MJPEG side-server anymore.

Config write-back uses `web/config_lock.py` (`flock` + temp file + `os.replace`).
`config.yaml` is authoritative; the DB never stores camera config.

Auth is split in two: `auth.py` is WSGI-environ based (bcrypt users, `sessions`
table, `wdt_`-prefixed API tokens, `Warden-Bearer` header) and `web/auth_bridge.py`
adapts a FastAPI `Request` into that environ. `CSRFMiddleware` is the outer
middleware, `ContextMiddleware` (sets `request.state.current_user` / `csrf_token`) is
inner; use `require_user` / `require_admin` from `web/auth_depends.py` on routes.

### Database (`db/`)

`db/engine.py` resolves `WARDEN_DB_URL` (default SQLite under `$XDG_DATA_HOME`) into
a process-wide engine; `reset_engine()` is how tests swap databases. `db/schema.py` is
both the CRUD helper layer and `ensure_schema()`, which migrates an empty DB to head,
stamps legacy `create_all` DBs, upgrades a DB that is behind head (a SQLite file is first
copied to `<db>.bak-<revision>`; a PostgreSQL DB is upgraded only when `WARDEN_DB_UPGRADE=1`
is set, after the owner took a `pg_dump`), and exits with a message on a revision it does not know.
`migrations/env.py` never calls `fileConfig` (that used to wipe the app's logging).

Tables: `users`, `sessions`, `api_tokens`, `events`, `action_runs`. Migration 0003 dropped
the never-written `cameras`, `recordings` and `ingest_health` tables and the `clips` table,
and backfilled `events.camera_name` from the old message text. Events are keyed by
`camera_name`. Every DB datetime is written as naive UTC and read back through
`schema.as_utc()` (SQLite drops tzinfo).

### Alerts and ONVIF

`AlertManager` is constructed in `create_app` when `cfg.alerts.enabled` and builds its
notifiers in `__init__`; it is used by the `/alerts` admin routes (list, edit, and the
`POST /alerts/{name}/test` button). `dispatch_event` has no caller in the runtime path:
`EventSink` does not call it, and the ONVIF subscriber callback registered by
`/onvif/cameras/{name}/events/subscribe` only logs. Wiring detections to notifiers is
sub-project 3 in `docs/superpowers/specs/2026-10-02-detection-and-automation-design.md`.

ONVIF PTZ and events use handcrafted SOAP over `httpx` (`onvif/ptz.py`,
`onvif/events.py`); there is no `zeep` dependency. Event
subscriptions are asyncio tasks on the uvicorn loop held in a module-level registry.
They are not persisted and do not survive a restart.

### Proxy modes

`proxy.mode: mjpeg` is in-process (`FrameHub` fed by the ingest's stdout, served by a
stdlib HTTP `MjpegProxyServer`). `proxy.mode: rtsp` spawns an external MediaMTX
process per camera; `AppRuntime.start` deliberately starts MediaMTX before the
ffmpeg that publishes to it.

## Testing conventions

Fixtures live in `tests/conftest.py`: `tmp_env` (points `WARDEN_DB_URL` at a temp
SQLite file), `clean_db` (adds `reset_engine()` + `ensure_schema()`), `admin_user`,
`db_with_user` (admin `admin` / `testpass123`), and `make_environ` for the WSGI auth
layer. Web tests build `create_app(settings, cfg=cfg, runtime_provider=lambda: None)`,
assign `app.state.config_path` / `app.state.runtime` as needed, and drive it with
`fastapi.testclient.TestClient`. ffmpeg tests only assert on the built argv. Detector
tests use synthetic numpy frames.

## Session docs

`HANDOFF.md` (newest block first, "START HERE") and `TASKS.md` (one card per task id, `RW-n`) at the repo root
carry session state between Claude sessions. Read the top HANDOFF block before starting work; update both at the
end of a session (`/handoff`). Design specs and implementation plans live under `docs/superpowers/`.

## Repository notes

- `SPRINT*_PLAN.md`, `Archive.zip`, `.coverage`, `recordings/`, `.env` are gitignored
  local files. Historical design notes live in `docs/archive/`; the current design
  spec and implementation plans are under `docs/superpowers/`.
- Sample real-camera configs are in `examples/configs/`; they reference
  `${CAM_USER}` / `${CAM_PASS}` and never contain credentials. `examples/config.yaml`
  is the minimal canonical one.
- The workspace rules in `~/Projects/AGENTS.md` (two levels up, not the Boomerang
  roster in `~/Projects/python/AGENTS.md`) apply: `uv` only, lint → typecheck → test
  before calling work done, never commit secrets.

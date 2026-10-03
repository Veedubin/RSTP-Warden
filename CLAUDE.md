# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rtsp-warden` is a self-hosted NVR for RTSP cameras: one Python process that runs
ffmpeg per camera stream, records `.ts` segments, runs detectors in-process (YOLOX on
ONNX Runtime, CPU or CUDA, plus OpenCV motion), turns tracked objects into events with
thumbnails, sends them through per-camera rules to ntfy / Apprise / webhook actions, and
serves a FastAPI web UI (htmx + Alpine + Pico, no JS build step). Package
version is 1.3.0 (`pyproject.toml` + `src/rtsp_warden/__init__.py`); the directory
name `rtsp-warden_v0.2.0` is historical. Remote: `Veedubin/RSTP-Warden` (public, MIT).

The README was rewritten on 2026-10-02 to match the code; when they disagree, trust the
code and fix the README. This file covers what the README does not say.

## Commands

```bash
uv sync                          # installs runtime (incl. onnxruntime CPU) + dev group (pytest, pytest-cov, pytest-asyncio, onnx)
uv run pytest                    # ~1530 tests, ~120s, fully offline (no ffmpeg, cameras, network, or model weights)
uv run pytest tests/test_X.py    # one file
uv run pytest -k "pattern"       # by name
uv run ruff check src/ tests/    # 4 pre-existing E501 in v0.x files are the accepted baseline (see below)
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

**Lint baseline.** The 4 E501 errors live in `cli.py`, `proxy/mjpeg.py`, `recorder.py`. The gate is "no new ruff errors"; do not reformat
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

**Migrations.** Alembic, packaged in `src/rtsp_warden/migrations/` (`versions/000N_slug.py`,
revision ids like `0003_detection_events`). `db/schema.py` builds the Alembic `Config()` in code
(`script_location = "rtsp_warden:migrations"`, no ini file, so `env.py` never calls
`fileConfig` and the app's logging survives); the repo-root `alembic.ini` is only for
`uv run alembic ...`. Add a model in `db/models.py` plus a hand-written migration; if it is a
new table, add it to `EXPECTED_TABLES` in `tests/test_alembic.py`. `ensure_schema()` upgrades a
database that is behind head when `serve` starts (copying a SQLite file to `<db>.bak-<revision>`
first) and refuses to start against a revision it does not know.

**Version bump** touches three places: `pyproject.toml`, `src/rtsp_warden/__init__.py`,
and the version assertion in `tests/test_admin.py`.

**Docker.** `docker compose up -d` builds `Dockerfile.distroless` (default, no shell).
`Dockerfile` is the slim debuggable variant. Both run
`rtsp-warden serve -c /app/config/config.yaml --web --web-port 8080`; the bind host comes
from `WARDEN_WEB_HOST` (compose sets `0.0.0.0`; the default is `127.0.0.1`).

**GPU image.** `Dockerfile.cuda` is `Dockerfile` with `--extra gpu --no-install-package onnxruntime`
in both `uv sync` lines plus `NVIDIA_*` and `LD_LIBRARY_PATH` in its `ENV`; `docker-compose.gpu.yml`
overlays it with an NVIDIA device reservation
(`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`). The CPU images
are unchanged. Models persist in `WARDEN_MODELS_DIR`: compose sets `/app/data/models`, the
systemd unit `/var/lib/rtsp-warden/models`. `tests/test_deploy_docs.py` pins these files; no
test builds an image.

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

### Detector pipeline (`frame_tap.py` → `detectors/` → `actions/`)

```
CameraRuntime.dispatcher → DetectorRunner.on_frame (one per camera; queue 8, drop-oldest, 1 worker)
  → cv2.imdecode → apply_masks (privacy polygons) → for each DetectorSlot due at its own fps:
      Detector.process(frame_bgr, ts_unix) → filter_by_roi → ignore-zone GridMask.filter_detections
  → motion slots:  MotionBurst → EventSink (only when CameraConfig.motion_events_enabled)
  → tracked slots: Tracker (greedy IoU per label) → EventBuilder → events row + thumbnail
      → on open:  RuleEngine.evaluate → ActionQueue (one action_runs row per run)
      → on close: ClipScheduler (when a matched rule has clip: true)
```

The `Detector` protocol (`detectors/base.py`) is `name`, `kind`, `setup()`,
`process(frame_bgr, ts_unix) -> list[Detection]` (bbox is x, y, w, h in frame pixels),
`teardown()`. `DetectorType` is `motion | person | vehicle | dnn | custom | onnx`; `person`,
`vehicle`, `dnn` and a spec's `interval_seconds` are deprecated and warn once through
`deprecations.warn_once`. `onnx` is `OnnxDetector` (`detectors/builtin/onnx.py`): YOLOX on ONNX
Runtime, providers chosen from `device` (`auto | cuda | cpu`), and the log line
`... provider: CUDAExecutionProvider` (or `CPUExecutionProvider`). It imports `onnxruntime` and
fetches its model only when it loads (in `setup()` when the file is present, else on the first
frame), never in `__init__`, so `rtsp-warden status` stays offline. A failed load sets `error`,
returns no detections and retries every 300 s of frame time; recording is never affected. Models
come from `detectors/model_registry.py`: built-in descriptors in `detectors/models/` (`yolox-s`,
`yolox-nano`, `coco.txt`), user descriptors in `runtime.models_dir` (`WARDEN_MODELS_DIR`, else the
XDG cache), SHA-256-verified downloads, and only `postprocess: yolox`. `detect_classes` and
`rules[].labels` are validated against the camera's model labels when the config loads.

`build_detectors_for_camera(cam, specs, *, models_dir)` is the single entry point: it applies
camera `sensitivity` and `detect_classes`, turns `kind: ignore` zones into `GridMask`s (normalised
by the decoded frame size, not the zone's saved size) and `kind: area` zones into the named masks
that set `event.zone`, and returns one `DetectorSlot` per spec. The slot index is the spec's list
position, which is also the detector's identity in the web UI and in config write-back
(`_persist_detector_entry`). The frame tap's rate and width come from
`config.compute_tap_settings(cam)`: `detect_fps`, and the widest enabled model input (min 320).

Hot reload is `AppRuntime.rebuild_camera_detectors(name)`: it rebuilds the bundle from the
in-memory `CameraConfig`, sets the new runner up, swaps it under `_detector_lock` onto that
camera's own dispatcher, then tears the old runner down on a background thread (`_retire_runners`; it joins
the worker and closes its open events), so a worker stuck in a model download never holds the
caller. The tracker starts empty. When the required tap settings changed (`detect_fps`, or a model
of another input width) it also calls `request_restart_camera(name)`, because they are ffmpeg
arguments; `_apply_add` / `_apply_restart` call it with `from_lifecycle=True`, which never requests
a restart from inside a supervisor drain. Web routes run every rebuild and every config write in
the threadpool (`run_in_threadpool`), and RW-3's config write-backs go through
`web/services/detection.update_config_yaml` (read, change and `_locked_write_yaml` under one
in-process lock). A per-detector `fps` change is a plain rebuild (the tap keeps `detect_fps`).

### Web layer (`web/`)

`create_app(settings, cfg, runtime_provider)` in `web/app.py` sets `app.state.cfg` and
`app.state.runtime_provider`, then includes one router
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
both the CRUD helper layer and `ensure_schema()`: an empty DB is upgraded to head, a legacy
`create_all` DB is stamped, a DB behind head is upgraded (a SQLite file is first copied to
`<db>.bak-<revision>`; a DB that cannot be copied, PostgreSQL, only with `WARDEN_DB_UPGRADE=1`),
and an unknown revision stops startup. `serve` runs it inside
`bootstrap_database()` before any thread starts.

Tables at `0003_detection_events`: `users`, `sessions`, `api_tokens`, `events`, `action_runs`
(plus `alembic_version`); `cameras`, `recordings`, `ingest_health` and `clips` were dropped.
`events` stores `camera_name` directly (0003 backfilled it from the old message text), plus
`label`, `confidence`, `zone`, `track_id`, `ended_at`, `thumbnail_path` and `clip_path` (both
relative to the camera's `record.output_dir`); `event_type = "test"` marks "Fire test event" rows.
Datetimes are written as naive UTC and read back through `schema.as_utc()`. Use the helpers
(`insert_event`, `update_event`, `close_event`, `list_events(camera_name=..., label=..., since=...,
until=...)`, `insert_action_run`, `action_stats`) rather than raw sessions. Thumbnails live at
`<output_dir>/<camera>/thumbnails/<event_id>.jpg` and clips at
`<output_dir>/<camera>/clips/<event_id>.(mp4|ts)`; retention sweeps both by age and size, and
`keep_last_n` counts only `main/` and `sub/` segments. A camera with `record.enabled: false` sweeps
only `thumbnails/` and `clips/` (`RetentionManager.only_subdirs`), never segments kept from before.

### Actions, rules and ONVIF

`actions/` replaced `alerts/` (there is no `AlertManager`). Top-level `actions:` entries (`ntfy`,
`webhook`, `apprise`) are built by `actions/factory.build_actions(cfg)`. Every action is
synchronous (`send(payload, attachment) -> ActionResult`, `test() -> ActionResult`), creates its
own `httpx.Client` per call, and never puts URLs, topics or tokens into `ActionResult.error`.
Per-camera `rules:` are evaluated by `actions/rules.RuleEngine` when an event opens (label, area
zone, `min_confidence`, local-time `between` that may wrap midnight, cooldown keyed by camera, rule
and label). Matches are enqueued on `ActionQueue` (bounded at 256, drop-oldest, one daemon worker,
one `action_runs` row per run, no retries); a matched `clip: true` rule schedules a `ClipJob` on
`ClipScheduler`, which runs after `ended_at + clips.post_seconds + 2 s`, writes an MP4, and keeps
the joined `.ts` when the remux fails. Payload URLs use `runtime.public_url`, else the web bind
address with `0.0.0.0` / `::` shown as `localhost`. A legacy `alerts.notifiers` list is mapped onto
`actions` at load with one warning. ONVIF events received through
`/onvif/cameras/{name}/events/subscribe` are still only logged; they create no events.

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
ONNX tests build tiny constant YOLOX graphs with `tests/helpers_onnx.py` (no weights, CPU
provider) and fake model downloads through `ensure_model_file(..., opener=...)`. GPU-only checks
carry `@pytest.mark.gpu` and skip unless the GPU build is installed. `tests/test_examples_load.py`
validates every file in `examples/` with and without its commented example blocks, and that the
`init-config` template loads (it gets the same blocks once RW-2's template lands);
`tests/test_deploy_docs.py` validates the README's marked config examples, its detection / events /
actions route rows against the routers, and the Docker files.

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
  is the minimal canonical one. It and `cli.SAMPLE_CONFIG_YAML` carry the same commented
  `# --- <name> example: ... ---` / `# --- end of <name> example ---` blocks (detection, actions);
  inside a block every line is commented YAML, so keep prose above the start marker.
- README config examples that the tests validate are marked `<!-- config-example: <name> -->`
  right above their fenced `yaml` block.
- The workspace rules in `~/Projects/AGENTS.md` (two levels up, not the Boomerang
  roster in `~/Projects/python/AGENTS.md`) apply: `uv` only, lint → typecheck → test
  before calling work done, never commit secrets.

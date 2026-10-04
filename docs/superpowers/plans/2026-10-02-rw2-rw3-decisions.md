# Planning decisions and file ownership for RW-0, RW-2 and RW-3

Read this before expanding any task. Every rule here is binding on every task in all three plans.
The spec is `docs/superpowers/specs/2026-10-02-detection-and-automation-design.md` (approved). Where the
codebase maps proved a spec premise false, the ruling below replaces the spec sentence and the plan says so.

## Owner decisions already made (from TASKS.md, HANDOFF.md and the resume prompt)

- ONNX Runtime + permissively licensed YOLOX only. No Ultralytics, no AGPL. Project stays MIT.
- GPU first, CPU fallback. Per-camera `detect_fps`, per-detector `fps`.
- Actions are ntfy / Apprise / webhook only. No Pi / GPIO work anywhere.
- Sub streams stay optional. The owner's Foscam uses `main_url` only (its `videoSub` sends no video).
- Live segment recording stays `.ts`. Never reintroduce MP4 live segmenting.
- `config.yaml` stays authoritative; every write-back goes through `web/config_lock._locked_write_yaml`.
- Never commit secrets. Config references `${CAM_USER}` / `${CAM_PASS}`-style env variables.
- `uv` only. Gate = `uv run pytest -q` green, `uv run ruff check src/ tests/` with no NEW errors (baseline is
  exactly 10 E501 lines), `uv run ruff format --check src/ tests/` clean. mypy is informational.
- Home-lab audience: "simple and usable for people without a ton of fluff". Workable, clean UI; not perfect.
- No push to origin without the owner's explicit OK. Merge decisions are the owner's.

## Rulings made during planning (owner may override at plan review; each plan restates the ones it uses)

R1. **Pre-fork step RW-0 on master.** Before the two worktrees are created, master gets two commits: repository
    hygiene (anchored `.gitignore` rules, the three never-tracked templates, `.worktrees` ignore) and the
    per-camera lifecycle API on `AppRuntime` (`_build_camera_runtime`, `_start_camera`, `_stop_camera`, a request
    queue drained by `run_forever` on the main thread). Both tracks consume that API; neither rewrites it.
R2. **Python floor becomes `>=3.11`** (onnxruntime >= 1.28 needs it). `[tool.ruff] target-version` stays
    `py310` so the ruff baseline stays at 10. `onnxruntime>=1.28` is a core dependency; the `gpu` extra is
    `onnxruntime-gpu[cuda,cudnn]>=1.28`; GPU installs use
    `uv sync --extra gpu --no-install-package onnxruntime --reinstall-package onnxruntime-gpu`. `onnx` is a dev
    dependency (test graphs only). `Dockerfile.cuda` is `python:3.13-slim` + the pip CUDA wheels, not an
    nvidia/cuda base image.
R3. **Credentials from the add-camera form** are stored as per-camera env references
    `${CAM_<SLUG>_USER}` / `${CAM_<SLUG>_PASS}` in `config.yaml` (percent-encoded userinfo is what the variable
    holds), and the values are written to the `.env` file that sits next to `config.yaml` (mode 0600, locked
    upsert). `serve`, `doctor` and `status` load `<config dir>/.env` as well as `./.env`. The running process also
    sets the variables in `os.environ` before expanding. `<SLUG>` is the camera name upper-cased with `-` turned
    into `_`. Nothing writes a plaintext password into `config.yaml`.
R4. **Zones gain `kind: ignore | area`** (default `ignore` = today's exclusion mask; additive, nothing breaks).
    An `area` zone never filters detections; it names a region. `event.zone` is the name of the first `area`
    zone (config order) whose active cells contain the centre of the best box, else `""`. `rules[].zones` names
    `area` zones; unknown names are a config validation error. `GridMask` normalises by the decoded frame shape
    at runtime, not by the zone's saved `frame_width/height` (that is a confirmed bug); the saved size stays as
    the fallback divisor so existing grid-mask tests keep passing.
R5. **Recordings UI, timeline, manual clip generation and the `clips` table are removed** in RW-3 (spec 9.1 drops
    `cameras`, `recordings`, `ingest_health`; `clips` has an FK to `cameras` and `events.clip_path` replaces it).
    `/recordings*`, `/api/recordings/{id}/timeline`, `web/services/{recordings,timeline,timeline_colors}.py`,
    `static/js/timeline.js`, `templates/recordings/*`, `templates/clips/*`, `partials/timeline.html`,
    `routes/clips.py`, `routes/api.py`'s timeline route, `detectors/categorize.py` and their tests go. The HLS
    player markup is extracted to `partials/hls_player.html` before `recordings/detail.html` is deleted. The
    `/htl` and `/segments` routes stay (file-system based, used by the `.ts` clip fallback).
R6. **Migrations move into the package** (`src/rtsp_warden/migrations/`, Alembic `Config()` built in code with
    `script_location = "rtsp_warden:migrations"`, no `alembic.ini` dependency at runtime). `ensure_schema()`
    auto-upgrades a database that is behind head after copying a SQLite file to `<db>.bak-<revision>`; it refuses
    to run against an unknown revision. `migrations/env.py` no longer calls `fileConfig` (it disabled the app's
    logging).
R7. **ONVIF page scope**: server-rendered htmx fragments and form-parsing routes (option A); the ONVIF routes stop
    reading `request.json()`. `PTZPresetStore._persist` is replaced by a raw-YAML patch written through
    `_locked_write_yaml` (it leaks expanded secrets today). `CameraConfig.onvif_port: int | None = None` is added;
    `_derive_onvif_xaddr` uses it (default 80). PTZ and event subscriptions keep HTTP Digest auth; WS-UsernameToken
    is used only by the new add-camera URL fill. Extending WS-Security to PTZ/events is deferred.
R8. **Clip jobs run on a dedicated `ClipScheduler` thread** with a not-before time of
    `ended_at + clips.post_seconds + 2 s`, not on the notification worker (spec 8.4 says "same worker"; a 120 s
    ffmpeg there would block notifications, and running at close yields truncated clips, verified). Clips are
    trimmed with input-side `-ss <offset> -t <duration>` before `-f concat`, use a per-event quote-escaped concat
    list, remux to `.mp4` with `-movflags +faststart`, and keep the concatenated `.ts` when the remux fails.
R9. **Deployments become writable**: `docker-compose.yml` mounts `./config` read-write; the systemd unit adds
    `/etc/rtsp-warden` to `ReadWritePaths`. Every config write-back route catches `OSError` and shows a flash
    message naming the path instead of a 500.
R10. **Edit and delete semantics**: the camera name is read-only on edit (rename = delete + add). Editable:
    `main_url`, `sub_url` (optional, blank clears it), username/password (blank keeps the current values),
    `record.enabled`, `onvif_port`. Proxy port is auto-allocated (lowest free port >= 9001 not used by another
    camera) and shown read-only. Delete removes the camera from `config.yaml` and the runtime and keeps its
    recordings, thumbnails, clips and events on disk; the confirmation says so. New camera names must match
    `^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$`, must be unique, and may not be `new`. `/cameras/{name}/settings` becomes
    a 303 redirect to `/cameras/{name}/edit`.
R11. **Motion events**: the motion detector has no debounce (spec 7.4's premise is false). RW-3 adds a per-camera
    `MotionBurst` stage: a burst opens after `min_track_frames` consecutive motion frames, one `events` row is
    inserted at open, `ended_at` is set when `track_grace_seconds` pass without motion. Motion rows are written
    only when the motion spec's `events` resolves true (explicit `true`, or unset with no enabled `onnx` spec).
R12. **Only the `yolox` postprocess ships** in this release. The spec names `yolo-anchor-free` and `ssd` without
    defining their tensor contracts; `postprocess` values other than `yolox` fail config validation naming the
    supported list, exactly as spec 5.2 requires for unknown values. Custom YOLOX-family models still drop in.
R13. **Test buttons**: the Actions page "Test" calls the action class directly in a threadpool, returns an HTML
    fragment with the result, and writes no `action_runs` row. "Fire test event" inserts a real `events` row with
    `label = "person"`, `event_type = "test"`, a generated placeholder thumbnail, `ended_at = created_at`, runs the
    real `RuleEngine` with cooldown bypassed (no cooldown stamp recorded) and the real `ActionQueue`, never queues a
    clip, and the response says which rules matched (or that none did and why). Test events show a "test" badge.
R14. **Legacy `alerts.notifiers`**: an `AppConfig` after-validator that runs only when the list is non-empty maps
    every `enabled: true` entry onto `actions` (name, type, url, topic, token, headers, method, urls carried;
    `severities`, `min_interval_seconds`, `min_severity`, `title_template` dropped with a warning naming them),
    leaves `enabled: false` entries where they are, errors on a name collision with `actions`, and logs one warning
    naming the entries moved. `alerts.enabled` is ignored. No `/alerts` redirect, no `rtsp_warden.alerts` shim:
    the package is renamed to `rtsp_warden.actions` and the tests are rewritten.
R15. **Detector identity** is the list index. The toggle route becomes
    `POST /cameras/{name}/detectors/{index}/enabled`; write-back patches that one raw YAML entry under the lock
    (`_persist_detector_entry`), so unknown keys and `${VAR}` text survive. `_persist_detectors` and
    `_detectors_to_list` are deleted.
R16. **`detect_classes`** is validated by an `AppConfig` after-validator against the union of labels of every
    `onnx` spec on that camera (enabled or not), skipped when the camera has no `onnx` spec. `rules[].labels` is
    validated against the same union plus `motion`. The classes page builds its groups from the model's labels
    and gains a `classes_mode = all | custom` field so "all" can be restored.
R17. **Actions are synchronous** (`httpx.Client` created per call; Apprise called directly). ntfy attaches the
    thumbnail as the request body with `title`, `message`, `filename` as query parameters (so non-ASCII camera
    names work) and the token in `Authorization`. Apprise uses `notify(attach=path)`; `attach=None` when the file
    is missing. Webhook posts JSON. Error text stored in `action_runs.error` is built from the status code and the
    response's error field, never from `str(exc)` (which contains the URL and topic).
R18. **Tap is always on** for the proxy stream of every camera (`cam.proxy.stream`; main when no `sub_url`), at
    `detect_fps` and the width from `compute_tap_settings(cam)` (largest `input_size[0]` among enabled detectors,
    minimum 320). Changing the required tap settings (via `detect_fps`, or enabling/disabling an `onnx` spec that
    changes the width) makes `rebuild_camera_detectors` call `request_restart_camera`. Each camera has its own
    `FrameTapDispatcher` on `CameraRuntime.dispatcher`; runners are built with `worker_count=1, queue_maxsize=8`.
R19. **Payload URL base**: `cli.serve` computes `public_url = cfg.runtime.public_url or f"http://{host}:{port}"`
    where a bind host of `0.0.0.0` or `::` is replaced by `localhost`, and passes it as `AppRuntime.public_url`.
R20. **Thumbnails and clips** live under `<record.output_dir>/<camera>/thumbnails/<event_id>.jpg` and
    `<record.output_dir>/<camera>/clips/<event_id>.(mp4|ts)`, stored in the DB as paths relative to
    `record.output_dir`. Retention `keep_last_n` counts only segment files under `main/` and `sub/`; thumbnails
    and clips are swept by `max_days` / `max_gb` only. Cameras with `record.enabled: false` get a
    `RetentionManager` anyway (so thumbnails are swept) and a rule with `clip: true` on such a camera logs a
    warning at config load and the clip job is skipped.
R21. **Event-mode recording** (`record.mode: event`) stays out of scope: with the tap fed by the same ffmpeg that
    event mode stops, detection cannot start recording. The plan documents the limitation; nothing else changes.
R22. **Manual camera tests** (Foscam at 192.0.2.72, `videoMain`, credentials in the owner's notes) are the last
    task of each plan and run only after the owner says so. They are never a prerequisite for the gate.

## File ownership between RW-2 (merged first) and RW-3 (rebased on top)

RW-2 owns and may edit freely:
- `web/templates/base.html`, `web/static/css/warden.css`, `web/static/js/warden.js`
- `web/templates/cameras/list.html`, `cameras/detail.html` (BUT the detectors `<article>` block — the one that
  loads `/cameras/{name}/detectors` — stays byte-for-byte unchanged and in place), `cameras/settings.html`
  (deleted), `cameras/form.html` (new), `cameras/zones.html`, `cameras/zones_editor.html`
- `web/templates/partials/camera_card.html`, `partials/password_reveal.html`, new `partials/flash.html`,
  new `partials/page_header.html`, new `partials/probe_result.html`
- `web/templates/login.html`, `health.html`, `partials/health_status.html`, `settings/form.html`, `users/*`,
  `tokens/*`, `onvif/index.html`, `dashboard.html` (camera-cards section and page chrome only; the "Recent events"
  block and the "Recent recordings" block stay byte-for-byte unchanged)
- `web/routes/_common.py`, `web/routes/auth.py`, `web/routes/dashboard.py`, `web/routes/settings.py`,
  `web/routes/users.py`, `web/routes/tokens.py`, `web/routes/health.py`, `web/routes/onvif.py`,
  `web/routes/zones.py` (list/editor pages only), new `web/routes/camera_edit.py`
- `web/routes/cameras.py` ONLY `cameras_list`, `camera_detail`, `camera_status`, `camera_settings` (deleted) and the
  imports they need. Nothing from `cameras_detectors_partial` (line 387) to the end of the file.
- `web/services/cameras.py` (`list_cameras`, `live_status`, `get_camera_by_name`; NOT `get_camera_detectors` or
  `_build_detector_summary`), new `web/services/camera_config.py`, new `web/env_file.py`, new `rtsp_warden/probe.py`,
  new `rtsp_warden/ports.py`
- `onvif/presets.py`, new `onvif/soap.py`, new `onvif/media.py`
- `config.py`: ONLY adds `CameraConfig.onvif_port` (right after `sub_url`) and `RuntimeConfig.ffprobe_path`
  (right after `mediamtx_path`)
- `cli.py`: ONLY `_load_dotenv` (config-dir `.env`), `_port_is_free` (moved to `ports.py`, re-imported), and the
  `init-config` template's credential placeholders
- `ffmpeg.py`: ONLY adds `redact_text(text: str) -> str` next to `redact_url`
- `app.py`: ONLY `_last_stderr_line` (wrap the result in `redact_text`)
- `docker-compose.yml` (`:ro` removal), `packaging/systemd/rtsp-warden.service` (`ReadWritePaths`), `README.md`
  sections on the web UI, adding cameras and ONVIF, `CLAUDE.md` web-layer paragraphs

RW-3 owns and may edit freely:
- everything under `detectors/`, `db/`, `migrations/` (moved into the package), `actions/` (new, replaces `alerts/`)
- `recorder.py`, `ffmpeg.py` (except `redact_text`), `frame_tap.py`, `clips.py`, `retention.py`,
  `retention_resolver.py`, `status_model.py`, `app.py` (except `_last_stderr_line`), `cli.py` (serve wiring of
  `public_url` and the ActionQueue, `status`, `doctor` detector lines; NOT `_load_dotenv` or the template
  placeholders), `config.py` (everything except the two RW-2 fields)
- `web/routes/cameras.py` from `cameras_detectors_partial` to the end (moved into new `web/routes/detection.py`),
  new `web/routes/detection.py`, `web/routes/events.py`, new `web/routes/actions.py`, `web/routes/api.py`,
  `web/routes/clips.py` (deleted), `web/routes/recordings.py` (deleted), `web/routes/alerts.py` (deleted)
- `web/services/events.py`, new `web/services/detection.py`, `web/services/recordings.py` (deleted),
  `web/services/timeline.py` and `timeline_colors.py` (deleted), `web/services/preview.py`
- `web/templates/events/*`, new `actions/*`, `alerts/*` (deleted), `recordings/*` (deleted), `clips/*` (deleted),
  `partials/event_row.html` (replaced by `partials/event_card.html`), `partials/detector_list.html`,
  `partials/timeline.html` (deleted), new `partials/hls_player.html`, `cameras/detection_classes.html`,
  `cameras/sensitivity.html`, new `cameras/_detection_panel.html`
- `web/templates/cameras/detail.html`: ONLY replaces the detectors `<article>` block with
  `{% include "cameras/_detection_panel.html" %}` and adds the "show boxes" toggle next to the preview image
- `web/templates/dashboard.html`: ONLY replaces the "Recent events" block with the event-card include and deletes
  the "Recent recordings" block
- `web/templates/base.html`: ONLY the nav items "Alerts" → "Actions" (href `/actions`) and removal of "Recordings"
- `web/static/css/warden.css`: ONLY appends rules at the end of the file under a `/* --- detection (RW-3) --- */`
  comment
- `web/app.py`: router includes only (add `actions`, `detection`; remove `alerts`, `clips`, `recordings`; drop the
  AlertManager block)
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `Dockerfile.distroless`, new `Dockerfile.cuda`, new
  `docker-compose.gpu.yml`, `docker-compose.yml` (adds `WARDEN_MODELS_DIR` and the models volume only),
  `README.md` sections on detection, actions, rules, events, GPU, Docker, configuration reference, `CLAUDE.md`
  detector/db/alerts paragraphs, `examples/config.yaml`, `examples/configs/*`

Both tracks: every route module uses `rtsp_warden.web.routes._common.templates` (one Jinja2Templates instance);
new route modules never build their own. Both tracks add their routers in `create_app` with one line each.

Expected merge conflicts (resolve mechanically, RW-2 wins on layout, RW-3 wins on content):
`config.py` (adjacent field insertions), `web/app.py` (router include lines), `base.html` (nav items),
`dashboard.html` (block replacement vs chrome), `cameras/detail.html` (block replacement vs layout),
`warden.css` (append vs restructure), `web/services/cameras.py` (RW-3 adds a `detection` key to the `list_cameras`
dict from `web/services/detection.py` in two lines), `README.md`, `CLAUDE.md`, `cli.py`, `uv.lock` (RW-2 adds no
dependencies, so RW-3's lock wins).

## Conventions every task follows

- Branches: RW-0 commits directly on `master`; RW-2 is `feat/ui-pass` in `.worktrees/rw-2`; RW-3 is
  `feat/detection` in `.worktrees/rw-3`. Both worktrees are created with `git worktree add .worktrees/rw-N -b
  <branch> master` and get their own `uv sync`. Every command in a worktree runs with
  `uv run --directory <abs worktree path>` or after `cd <abs worktree path> &&`; every git command uses
  `git -C <abs worktree path>`; every file path is absolute under the worktree.
- Commit after every task with the message given in the task, ending with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Tests are offline: no camera, no ffmpeg process, no network, no onnx weights. ffmpeg/ffprobe calls are faked by
  patching `subprocess.run` / `subprocess.Popen` looked up at call time; HTTP by `httpx.MockTransport`; time by
  passing `now` / `ts_unix` / a `clock` callable into the code under test (there is no freezegun).
- Deprecation warnings are tested by monkeypatching the module logger (`module.log.warning`), never `caplog`
  (Alembic's logging config disables loggers when `ensure_schema` runs).
- Numbers that reach JSON (`status.json`, `/metrics`) are cast to `int` / `float`; never numpy scalars.
- Never print a credential in a test name, commit message, log line or fixture. Test RTSP URLs use `rtsp://u:p@h/m`.
- Keep `ruff format` clean and keep E501 at or below the 10-line baseline: if a task touches one of the baseline
  lines, fix that line (the baseline shrinks; it never grows).

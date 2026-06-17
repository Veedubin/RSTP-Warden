# rtsp-warden

A self-hosted Network Video Recorder (NVR) for RTSP cameras. Records continuously or on event, runs motion / person / vehicle / DNN detection in-process, and exposes a web UI for live viewing, playback, and admin. No cloud, no subscriptions, no agent on each camera.

## What it does

- **Records** RTSP streams to disk as time-segmented `.ts` files (NVR-grade, recovers from bad SDPs)
- **Detects** motion, persons, vehicles, and arbitrary objects (YOLOv4-tiny, 80 COCO classes) on the recording host
- **Alerts** via [ntfy](https://ntfy.sh), generic webhooks, or [Apprise](https://github.com/caronc/apprise) (email, Discord, Telegram, Slack, Pushover, 90+ services)
- **ONVIF** camera discovery + PTZ control + event subscription (motion alarms, tampering)
- **Clips** generate MP4 from any detection event (configurable pre/post-roll)
- **Zones** are grid-based (toggle cells like your home security app) or polygon ROI; the "block the road" use case
- **Tuning** per camera: sensitivity (0-100), detection classes, enable/disable per detector
- **Multi-user** with bcrypt sessions, bearer API tokens, and admin/viewer roles
- **Timeline** with object-type colored markers (person = red, pet = blue, critter = green, vehicle = orange)

## Why this and not ZoneMinder / Shinobi / Frigate?

| | rtsp-warden | Frigate | ZoneMinder | Shinobi |
|---|---|---|---|---|
| **Stack** | Python stdlib + FastAPI | Python + Go + Coral/Google Coral | Perl + MySQL | Node.js |
| **Web UI** | Server-rendered (htmx + Alpine), no build step | React + Vite (heavy build) | Legacy jQuery | Angular (heavy build) |
| **Database** | SQLite default, Postgres optional | SQLite only | MySQL required | SQLite/MySQL |
| **Auth** | Multi-user, bcrypt, roles, API tokens, CSRF | None built-in | Basic | Basic |
| **Detection** | Motion, HOG person, Haar vehicle, YOLOv4 DNN, custom detectors | YOLO (always, needs GPU) | Zoneminder motion | Plugin-based |
| **Clips from events** | Built-in, MP4, ffmpeg concat | Built-in | Manual | Plugin |
| **ONVIF events** | Built-in (pull-point) | No | Limited | Limited |
| **Docker image** | 685 MB distroless default, 1.2 GB slim | ~1.5 GB | ~500 MB | ~400 MB |
| **Tests** | 780 | Many (Python) | Perl | Limited |
| **License** | MIT | MIT | GPLv2 | GPLv3 |

rtsp-warden targets homelab/self-hosters who want a feature-complete NVR with a sensible admin UI, no JS build step, no GPU requirement, and a small Docker footprint. It is not the fastest, and it does not have the largest community.

## Quick start

### Option A: pip (any Linux)

```bash
# 1. Install
pip install rtsp-warden
# OR for the latest: pip install git+https://github.com/Veedubin/RSTP-Warden.git

# 2. First-run setup: creates .env, admin user, database schema
rtsp-warden install
# (note the admin password it prints)

# 3. Create a starter config
rtsp-warden init-config --out config.yaml
# Edit it: add your cameras' RTSP URLs

# 4. Validate
rtsp-warden doctor -c config.yaml

# 5. Start the recorder + web UI
rtsp-warden serve -c config.yaml --web --web-port 8080
```

Open **http://127.0.0.1:8080/** and log in with the password from step 2.

### Option B: Docker (distroless, recommended)

```bash
# Clone (or copy your config into the project)
git clone https://github.com/Veedubin/RSTP-Warden.git
cd RSTP-Warden

# Copy a sample config and edit it
cp examples/configs/config-Foscam-C1-V3.yaml config.yaml
$EDITOR config.yaml

# Run
docker compose up -d
# Browse to http://localhost:8080
```

The distroless image is 685 MB. See [docker/README.md](docker/README.md) for the slim alternative, ONVIF/UDP notes, and volume-mounting gotchas.

## How it fits together

```
                        config.yaml
                            |
                            v
   rtsp-warden serve  -->  AppConfig
        |                       |
        v                       v
   StreamIngestor (1 per camera)        AlertManager
   |- ffmpeg subprocess                 |- ntfy notifier
   |- writes .ts segments               |- webhook notifier
   '- emits JPEG via frame tap          '- apprise notifier (90+ services)
                |
                v
        FrameConsumer (chain)
        |- MotionDetector (MOG2)
        |- PersonDetector (HOG)
        |- VehicleDetector (Haar)
        |- DNNDetector (YOLOv4-tiny, 80 COCO classes)
        '- EventSink (writes events table)
                            |
                            v
                 SQLite or PostgreSQL
                            |
                            v
                    FastAPI web UI
                    (htmx + Alpine + Pico)
```

Two processes per camera (`main` stream and `sub` stream), each running `ffmpeg`. Frames are tee'd via the frame-tap pipe FD to a chain of `FrameConsumer` objects. Detectors run in worker threads; one consumer is an `EventSink` that writes detection events to the database. The web UI reads from the same database.

## Configuration

### `config.yaml` (canonical example)

See [examples/config.yaml](examples/config.yaml) for a minimal single-camera config, or [examples/configs/](examples/configs/) for full real-world configs (Foscam, TP-Link, 2-camera NC230).

The config has three top-level sections:

```yaml
cameras:        # list of CameraConfig (required, at least one)
runtime:        # global runtime settings (ffmpeg path, restart policy, etc.)
alerts:         # notifier list (Sprint 5/v1.1.0+)
clips:          # clip generation settings (v1.1.0+)
onvif:          # ONVIF global config (v1.1.0+)
retention:      # global fallback retention (v1.2.0+; per-camera overrides supported)
```

### Camera config (every field)

```yaml
cameras:
  - name: front_door                       # required, unique
    main_url: rtsp://user:pass@host:554/... # required
    sub_url:  rtsp://user:pass@host:554/... # optional (used for proxy/recording)

    record:
      enabled: true
      mode: continuous                     # continuous | event
      audio: false                         # opt-in audio recording (v1.1.0+)
      output_dir: ./recordings
      main: { container: ts, chunk_seconds: 300, rtsp_transport: tcp }
      sub:  { container: ts, chunk_seconds: 300, rtsp_transport: tcp }
      retention:                            # per-camera override (v1.2.0+)
        max_days: 30
        max_gb: 50.0
        keep_last_n: 100
        cleanup_interval_seconds: 300

    proxy:
      enabled: true
      mode: mjpeg                          # mjpeg | rtsp
      stream: sub
      bind_host: 0.0.0.0
      port: 9001
      fps: 7                               # mjpeg mode
      scale_width: 0                       # mjpeg mode (0 = no scaling)

    detectors:                             # list (v0.7.0+)
      - type: motion                       # MOG2 background subtraction
        enabled: true
        interval_seconds: 1.0
        min_area: 500
      - type: person                       # HOG + linear SVM
        enabled: true
        min_confidence: 0.5
      - type: vehicle                      # Haar cascade
        enabled: false
        min_confidence: 0.7
      - type: dnn                          # YOLOv4-tiny, 80 COCO classes (v1.1.0+)
        enabled: true
        model: yolov4-tiny
        confidence: 0.5
        nms_threshold: 0.4
        classes: [car, truck, dog, cat]    # None = all 80 classes

    sensitivity: 50                        # 0-100, applied to all detectors (v1.2.0+)
    detect_classes: [person, dog, cat]     # camera-level filter (v1.2.0+; DNN only)

    zones:                                  # grid-based detection zones (v1.2.0+)
      - name: exclude_road
        grid_cols: 16
        grid_rows: 16
        frame_width: 1920
        frame_height: 1080
        blocked_cells: [[0, 0], [1, 0], [2, 0]]

    presets:                               # PTZ presets (v1.1.0+)
      - name: front_gate
        pan: 0.0
        tilt: 0.0
        zoom: 0.5

    onvif:                                 # optional per-camera ONVIF (v1.1.0+)
      host: 192.168.1.100
      port: 80
      username: admin
      password: changeme
    events:                                # ONVIF event subscriptions (v1.1.0+)
      - type: motion
        min_interval_seconds: 30
      - type: tamper
        min_interval_seconds: 60
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WARDEN_DB_URL` | `$XDG_DATA_HOME/rtsp-warden/warden.db` (sqlite) | Database URL. Set to `postgresql://user:pass@host:7777/warden` for Postgres (default port 7777) |
| `WARDEN_SECRET` | (required) | App secret for signing session cookies |
| `WARDEN_AUTH_ENABLED` | `true` | Enable auth on protected routes |
| `WARDEN_AUTH_HEALTHZ_OPEN` | `false` | Allow anonymous `/healthz` (for k8s probes) |
| `WARDEN_WEB_ENABLED` | `true` | Enable web UI |
| `WARDEN_WEB_HOST` | `127.0.0.1` | Web UI bind host |
| `WARDEN_WEB_PORT` | `8080` | Web UI bind port |
| `WARDEN_HTTPS` | `false` | Set true for secure cookies (behind a TLS-terminating reverse proxy) |
| `TZ` | `UTC` | Timezone for log timestamps and segment filenames |

## CLI reference

```bash
rtsp-warden [OPTIONS] COMMAND [ARGS]
```

| Command | Description |
|---|---|
| `install` | First-run setup: writes `.env`, creates admin user, creates DB schema |
| `init-config` | Write a starter `config.yaml` to stdout or a file |
| `doctor` | Validate config + check ffmpeg/mediamtx availability + check ports |
| `serve` | Start the recorder + web UI + health endpoints (default command) |
| `ui` | Start the legacy stdlib MJPEG grid UI (separate process, no auth) |
| `status` | Print a JSON status snapshot |
| `version` | Print the package version |

### `serve` flags

| Flag | Default | Description |
|---|---|---|
| `-c, --config PATH` | `config.yaml` | Path to config file |
| `--web` / `--no-web` | `--web` | Enable/disable the web UI |
| `--web-host HOST` | `WARDEN_WEB_HOST` (127.0.0.1) | Web UI bind host |
| `--web-port PORT` | `WARDEN_WEB_PORT` (8080) | Web UI bind port |
| `--detectors` / `--no-detectors` | `--detectors` | Enable/disable the detector framework |

The `run` command exists as a hidden alias for `serve` for backward compatibility; using it prints a deprecation warning.

## Web UI routes

| Path | Auth | Description |
|---|---|---|
| `GET /` | user | Dashboard with live status, today's detection count, and camera grid |
| `GET /login` / `POST /login` | none | Login (rate-limited to 5/min) |
| `POST /logout` | user | Logout |
| `GET /cameras` | user | Camera list |
| `GET /cameras/{name}` | user | Camera detail: live MJPEG, recent recordings, recent events, detector list, zone/sensitivity controls |
| `GET /cameras/{name}/detectors` | user | Detector list (htmx partial, auto-refresh) |
| `GET /cameras/{name}/retention` / `POST` | admin | Per-camera retention policy |
| `GET /cameras/{name}/zones` / `POST` | admin | Grid-based detection zone editor (web UI) |
| `POST /cameras/{name}/zones/{name}/delete` | admin | Delete a zone |
| `GET /cameras/{name}/sensitivity` / `POST` | admin | Per-camera sensitivity (0-100) |
| `GET /cameras/{name}/detection-classes` / `POST` | admin | Per-camera detection class list |
| `POST /cameras/{name}/detectors/{type}/enabled` | admin | Toggle a detector on/off |
| `POST /cameras/{name}/reload` | admin | Hot-reload that camera's config (no restart) |
| `GET /recordings` | user | Recording list with filter |
| `GET /recordings/{id}` | user | Recording detail with HLS player + canvas timeline |
| `GET /events` | user | Event list (auto-refresh every 10s) |
| `GET /events/{id}` | user | Event detail with "Generate Clip" button |
| `POST /events/{event_id}/clip` | user | Generate an MP4 clip for the event |
| `GET /clips/{clip_id}` | user | Clip detail page |
| `GET /clips/{clip_id}/download` | user | Download the MP4 clip |
| `GET /users` / `POST /users/new` | admin | User management |
| `POST /users/{id}/reset-password` / `delete` / `toggle-admin` | admin | User actions |
| `GET /api-tokens` / `POST` / `POST .../revoke` | user | API token management (bearer) |
| `GET /settings` | admin | System settings (read-only display) |
| `GET /alerts` / `new` / `{name}/edit` / `{name}/test` | admin | Alert notifier management (ntfy, webhook, apprise) |
| `GET /onvif` / `POST /onvif/discover` | admin | ONVIF camera discovery |
| `GET /onvif/cameras/{name}/ptz` | user | PTZ control pad (with preset management) |
| `POST /onvif/cameras/{name}/events/subscribe` / `unsubscribe` | admin | ONVIF event subscription |
| `POST /onvif/cameras/{name}/ptz/goto` / `save` / `{preset}/delete` | varies | PTZ preset management |
| `GET /htl/{cam}/{stream}/{start}/{end}.m3u8` | user | Dynamic HLS playlist for a time window |
| `GET /segments/{cam}/{stream}/{path:path}` | user | Serve a TS segment file |
| `GET /api/recordings/{id}/timeline` | user | JSON timeline data for the canvas scrubber |
| `GET /healthz` / `/health` / `/status.json` | none | Liveness / health / full status (JSON or HTML) |
| `GET /metrics` | none | Prometheus metrics |

**Auth legend:** `none` = always public; `user` = any authenticated user; `admin` = admin role required.

## Concepts

### Recording

Each camera has two streams (`main` and optional `sub`). Each stream runs an `ffmpeg` subprocess writing HLS-style segmented files. Default container is `.ts` (MPEG-TS), which is NVR-grade and survives camera stream restarts that would corrupt `.mp4` or `.mkv`.

**Modes** (per camera):
- `continuous` (default): always recording
- `event`: records only when a detection event is recent (1s polling loop reads the events table; segments start at the moment of trigger)

**Audio** is opt-in per camera (`record.audio: true`). Adds `-c:a aac -b:a 128k` to ffmpeg.

**Retention** has a three-tier resolution chain: camera override > record-level policy > global. Files older than `max_days`, or beyond `max_gb`, or older than the `keep_last_n`-th file are cleaned at `cleanup_interval_seconds`.

### Detection

The detector framework runs on a chain of `FrameConsumer` objects that receive JPEG frames from the ingest's frame-tap pipe. Each detector runs in a worker thread; results are written to the `events` table by an `EventSink` consumer.

| Detector | What | Speed | Accuracy |
|---|---|---|---|
| `motion` | MOG2 background subtraction | Very fast | Low (any motion) |
| `person` | HOG + linear SVM | Fast | Medium (upright people only) |
| `vehicle` | Haar cascade (bundled) | Fast | Low (false positives) |
| `dnn` | YOLOv4-tiny via OpenCV DNN, 80 COCO classes | Slower | High (cars, trucks, dogs, cats, deer, etc.) |
| `custom` | User-supplied detector via `import_path: module:Class` | Depends | Depends |

**Tuning per camera:**
- `sensitivity` (0-100) — single knob that scales per-detector params (motion varThreshold, person/DNN confidence, DNN NMS). Higher = more sensitive.
- `detect_classes` — list of COCO classes to detect; intersected with the detector's own `classes` list. None means "all".
- `enabled` per detector — toggle individual detectors on/off without deleting config.
- **Zones** — grid-based (N×M cells, block specific cells to ignore that area) or polygon ROI. AND semantics: a detection must pass the polygon ROI AND not be in a blocked grid cell.

**Hot reload:** `POST /cameras/{name}/reload` rebuilds that camera's detector chain from config without restarting the server. Changes to `sensitivity`, `detect_classes`, `zones`, and detector `enabled` flags take effect immediately.

### Alerts

`AlertManager` debounces events by `(camera, event_type)` and dispatches them to one or more notifiers in parallel. Severity filter: `info` (default) drops `info`-level events, `warning` and `error` always pass. `min_interval_seconds` per notifier prevents flooding.

Notifier types:
- `ntfy` — push to an ntfy topic
- `webhook` — generic HTTP POST with JSON body
- `apprise` — any of apprise's 90+ services (email via SMTP, Discord, Telegram, Slack, Pushover, etc.)

### ONVIF

- **Discovery:** WS-Discovery UDP multicast to `239.255.255.250:3702`. **Note:** this does not cross Docker bridge networks; use `network_mode: host` for Docker deployments.
- **PTZ:** absolute_move, continuous_move, stop. Presets are saved in `config.yaml` under `camera.presets`.
- **Events:** pull-point subscription over SOAP. Topics: `VideoSource/MotionAlarm`, `VideoSource/ImagingAlarm` (tamper), `RuleEngine/*` (analytic). ONVIF events feed into the same `AlertManager` as in-process detector events.

### Clips

Generate an MP4 from the HLS segments around a detection event. Default: 10 seconds before + 10 seconds after. Uses `ffmpeg -f concat -c copy` (no re-encoding, fast). Generated clips live in `clips.output_dir` (default: `{recordings_root}/../clips`) and are tracked in the `clips` table.

## Deployment

### Docker (recommended)

The default `docker-compose.yml` builds and runs the **distroless** image (~685 MB). See [docker/README.md](docker/README.md) for:
- Architecture (4-stage build with BFS `.so` dependency collector)
- The slim alternative (~1.2 GB, has a shell, useful for debugging)
- ONVIF/UDP and volume-mounting gotchas
- Bind mount vs named volume permissions

### systemd (Linux)

```bash
sudo pip install rtsp-warden
sudo packaging/systemd/install.sh
sudo cp your-config.yaml /etc/rtsp-warden/config.yaml
sudo systemctl enable --now rtsp-warden
```

The unit is hardened: `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `PrivateDevices`, `ReadWritePaths` limited to the data directories. See [packaging/systemd/README.md](packaging/systemd/README.md).

## Development

```bash
# Clone
git clone https://github.com/Veedubin/RSTP-Warden.git
cd RSTP-Warden

# Install with dev deps (uses uv)
uv sync --extra dev

# Run tests
uv run pytest                  # 780 tests, ~45s
uv run pytest tests/test_X.py  # single file
uv run pytest -k "pattern"     # by name

# Lint
uv run ruff check src/ tests/  # 0 errors in new code; 20 pre-existing E501 warnings in v0.x files
uv run ruff format src/ tests/

# Type check (informal; not in CI yet)
uv run mypy src/ || true
```

The test suite uses `asyncio_mode = "auto"` and is fully self-contained — no live cameras, no ffmpeg binary, no real network. Each test gets a tmp directory and an isolated SQLite DB.

## Architecture invariants

These are the stable contracts other code depends on. Do not break them in PRs:

1. **`FrameConsumer` protocol** — `on_frame(camera, stream, jpeg_bytes, ts_unix)`. Receivers are chained; exceptions are caught and logged (they never propagate to ingest).
2. **`Detector` protocol** — `detect(frame: np.ndarray) -> list[Detection]`. Operates on decoded BGR frames; receives an already-masked frame and a list of `Mask`/`ROI` objects.
3. **`Notifier` protocol** — `async send(event: AlertEvent) -> NotificationResult`. Async; failures are returned, not raised.
4. **Config authority:** `config.yaml` is authoritative; the DB is a read-cache. Changes to the YAML take effect on server restart OR via the per-camera hot-reload endpoint.
5. **Recording is additive** — adding new consumers, notifiers, detectors, or web routes must not destabilize the ingest path.

## License

MIT. See [LICENSE](LICENSE).

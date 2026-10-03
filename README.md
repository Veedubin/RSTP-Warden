# rtsp-warden

<p align="center">
  <img src="docs/assets/rtsp-warden-logo.png" alt="rtsp-warden logo" width="320">
</p>

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
| **Tests** | 800+ | Many (Python) | Perl | Limited |
| **License** | MIT | MIT | GPLv2 | GPLv3 |

rtsp-warden targets homelab/self-hosters who want a feature-complete NVR with a sensible admin UI, no JS build step, no GPU requirement, and a small Docker footprint. It is not the fastest, and it does not have the largest community.

## Quick start

### Option A: pip (any Linux)

```bash
# 1. Install
pip install rtsp-warden
# OR for the latest: pip install git+https://github.com/Veedubin/RSTP-Warden.git

# 2. (Optional) first-run setup: writes .env with an admin password you choose.
#    If you skip it, `serve` creates the schema and an admin user on first start
#    and prints the generated password in its log.
rtsp-warden install

# 3. Create a starter config, and put the camera login in a .env file next to it
#    (the template reads ${CAM_USER} / ${CAM_PASS}; percent-encode @ : / ? # %)
rtsp-warden init-config --out config.yaml
[ -e .env ] || (umask 077 && printf 'CAM_USER=admin\nCAM_PASS=your-camera-password\n' > .env)
#    ^ creates .env (mode 600) only if it does not exist; otherwise add the two lines to it
# Edit config.yaml: set your camera's address and stream path

# 4. Validate
rtsp-warden doctor -c config.yaml

# 5. Start the recorder + web UI
rtsp-warden serve -c config.yaml --web --web-port 8080
```

Open **http://127.0.0.1:8080/** and log in with the password from step 2. More cameras can be added from the web UI; see [Adding a camera](#adding-a-camera).

### Option B: Docker (distroless, recommended)

```bash
# Clone (or copy your config into the project)
git clone https://github.com/Veedubin/RSTP-Warden.git
cd RSTP-Warden

# Copy a sample config into ./config (compose mounts that directory) and edit it
mkdir -p config recordings data
cp examples/configs/config-Foscam-C1-V3.yaml config/config.yaml
$EDITOR config/config.yaml

# The samples read camera credentials from the environment; put them in .env
printf 'CAM_USER=admin\nCAM_PASS=your-camera-password\n' > .env

# Run
docker compose up -d
docker compose logs warden | grep "created admin"   # first-start admin password
# Browse to http://localhost:8080
```

The distroless image is 685 MB. See [docker/README.md](docker/README.md) for the slim alternative, ONVIF/UDP notes, and volume-mounting gotchas.

## Adding a camera

Admins add, edit and delete cameras in the web UI, without a restart. Open **Cameras**, then **Add camera** (`/cameras/new`).

| Field | Notes |
|---|---|
| Name | 1 to 32 letters, digits, `-` or `_`, starting with a letter or digit. Must be unique; names that differ only in upper and lower case, or in `-` versus `_`, count as the same name. `new` is reserved. It cannot be changed later: to rename a camera, delete it and add it again. |
| Host | The camera's IP address or host name. |
| User name, Password | The camera's login. They are not written to `config.yaml` (see below). |
| ONVIF port | Filled by **Find stream URLs (ONVIF)**; the ONVIF page (PTZ, presets, events) uses it (default 80). Leave it empty to let the lookup try ports 80, 8080, 888 and 2020. |
| Main stream | A path such as `/videoMain` or a full `rtsp://` URL without the login; filled by **Find stream URLs (ONVIF)**. Left blank, it becomes `rtsp://<host>:554/`. |
| Sub stream (optional) | Same format. Without it, the live view and detection use the main stream, and only the main stream is recorded. |
| Record continuously | Record this camera to disk. |

**Find stream URLs (ONVIF)** asks the camera for its stream URLs (`GetStreamUri`, WS-UsernameToken login) on ports 80, 8080, 888 and 2020 (or only the ONVIF port you typed), waiting at most 3 seconds per port, and uses the first port that answers. The URLs it returns are rewritten to the host you typed, because cameras often report an address you cannot reach. A wrong user name or password stops at the first port that answers and says so; a failed lookup names the ports it tried.

**Test main stream** / **Test sub stream** run `ffprobe` against that stream with a timeout and show the codec, resolution, frame rate and one snapshot. They use `runtime.ffprobe_path` when that is set, otherwise the `ffprobe` next to `runtime.ffmpeg_path`; both Docker images include it. Saving does not require a passing test.

**Save camera** adds the camera to `config.yaml` and starts it. Its MJPEG proxy port is the lowest free port from 9001 up that no other camera uses; the edit page shows it read-only.

### Where the credentials go

The form writes environment references into `config.yaml`, never the login itself:

```yaml
cameras:
  - name: front-door
    main_url: rtsp://${CAM_FRONT_DOOR_USER}:${CAM_FRONT_DOOR_PASS}@192.168.1.60:554/stream1
    onvif_port: 8080
    record:
      enabled: true
    proxy:
      enabled: true
      mode: mjpeg
      stream: main
      port: 9002
```

The variable names are `CAM_`, the camera name upper-cased with `-` turned into `_`, then `_USER` or `_PASS`. Their values are percent-encoded (so a password with `@`, `/`, `#` or `%` works) and saved in the `.env` file next to `config.yaml`, created with mode 0600. `serve`, `doctor` and `status` read that file, then `./.env`, at start; a variable that is already set in the environment (compose `environment:`, systemd `EnvironmentFile=`) wins. That is why **Save camera** refuses a name whose `CAM_<NAME>_USER` / `CAM_<NAME>_PASS` are already set to other values, in that `.env` or in the service environment: choose another name, or remove the old lines first. The repository's `.gitignore` ignores every `.env`.

For a camera you write by hand, use the same pattern: `rtsp://${CAM_USER}:${CAM_PASS}@host:554/path`, with `CAM_USER` and `CAM_PASS` in that `.env` and `@ : / ? # %` percent-encoded in the values.

### Editing and deleting

The camera page has **Edit camera** and **Delete camera** buttons. **Edit camera** (`/cameras/{name}/edit`; the old `/cameras/{name}/settings` page redirects there) changes the main and sub stream URLs (a blank sub stream removes it), the login (blank fields keep the current one), recording on or off, and the ONVIF port. Changing a URL, the login or the recording switch restarts only that camera's ffmpeg. **Delete camera** removes the camera from `config.yaml` and stops it, and drops its `CAM_<NAME>_*` login from the `.env` when no other camera uses it; its recordings and events are kept.

### The config directory must be writable

A save writes `config.yaml`, a `.config.yaml.lock` lock file, a temporary `config.yaml.tmp` and `.env` in the directory that holds `config.yaml`, so that directory must be writable by the user rtsp-warden runs as. The web UI rewrites `config.yaml` with PyYAML, so comments in it are not kept. When the directory is read-only, the page shows an error naming the file instead of saving.

- **Docker:** `docker-compose.yml` mounts `./config` read-write. A bind mount keeps the owner of the host directory, so give the directories to the container user before the first start: `sudo chown -R 65532:65532 config recordings data` for the default distroless image (`1000:1000` for the slim image). Do not mount `config/` read-only (`:ro`).
- **systemd:** the unit lists `/etc/rtsp-warden` in `ReadWritePaths`, and `packaging/systemd/install.sh` creates it as `0770 root:rtsp-warden` (re-run it on an older install, then restart the service). Logins of cameras added from the web UI land in `/etc/rtsp-warden/.env`; `warden.env` stays the place for everything else.

## How it fits together

```
                        config.yaml
                            |
                            v
   rtsp-warden serve  -->  AppConfig
        |                       |
        v                       v
   StreamIngestor (1 per camera stream)  AlertManager
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

One `ffmpeg` process per configured stream (`main`, plus `sub` when set). Frames are tee'd via the frame-tap pipe FD to a chain of `FrameConsumer` objects. Detectors run in worker threads; one consumer is an `EventSink` that writes detection events to the database. The web UI reads from the same database.

## Configuration

### `config.yaml` (canonical example)

See [examples/config.yaml](examples/config.yaml) for a minimal single-camera config, or [examples/configs/](examples/configs/) for full real-world configs (Foscam, TP-Link, 2-camera NC230).

The config has six top-level sections:

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
    sub_url:  rtsp://user:pass@host:554/... # optional; proxy and frame tap fall back to main, sub recording is skipped
    # Credentials can come from the environment: rtsp://${CAM_USER}:${CAM_PASS}@host:554/...

    record:
      enabled: true
      mode: continuous                     # continuous | event
      audio: false                         # opt-in audio recording (v1.1.0+)
      output_dir: ./recordings
      main: { container: ts, chunk_seconds: 300, rtsp_transport: tcp }
      sub:  { container: ts, chunk_seconds: 300, rtsp_transport: tcp }

    retention:                              # per-camera override of the global `retention:` block
      max_days: 30                          # (a `retention:` nested under `record:` is deprecated;
      max_gb: 50.0                          #  it is still honored, with a warning in the log)
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
        config:                            # DNN options live under `config:`
          confidence_threshold: 0.5
          nms_threshold: 0.4
          classes: [car, truck, dog, cat]  # omit = vehicles + animals (person is NOT included by default)

    sensitivity: 50                        # 0-100, applied to all detectors (v1.2.0+)
    detect_classes: [person, dog, cat]     # camera-level filter (v1.2.0+; DNN only)

    zones:                                  # grid-based detection zones (v1.2.0+)
      - name: exclude_road
        grid_cols: 16
        grid_rows: 16
        frame_width: 1920
        frame_height: 1080
        blocked_cells: [[0, 0], [1, 0], [2, 0]]

    presets:                               # PTZ presets (v1.1.0+); ONVIF credentials are global (`onvif:` below)
      - name: front_gate
        pan: 0.0
        tilt: 0.0
        zoom: 0.5

    events:                                # ONVIF event subscriptions (v1.1.0+)
      - type: motion
        min_interval_seconds: 30
      - type: tamper
        min_interval_seconds: 60
```

### Global sections

```yaml
runtime:
  ffmpeg_path: ffmpeg
  mediamtx_path: mediamtx           # only needed for proxy.mode: rtsp
  workspace_dir: ./workspace
  auto_restart: true
  restart_backoff_min_s: 1
  restart_backoff_max_s: 60
  restart_backoff_factor: 2
  stderr_tail_lines: 200
  status_interval_s: 15
retention:                          # global fallback for cameras without their own block
  max_days: 7
onvif:                              # discovery, PTZ and events are all OFF by default
  discovery_enabled: false
  ptz_enabled: false
  events_enabled: false
  username: admin                   # used for every camera; ONVIF is reached on the camera's
  password: ${ONVIF_PASS}           # RTSP host, port 80, /onvif/device_service
alerts:
  enabled: false
  notifiers: []                     # see Alerts below
clips:
  enabled: true
  pre_seconds: 10
  post_seconds: 10
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WARDEN_DB_URL` | `$XDG_DATA_HOME/rtsp-warden/warden.db` (sqlite) | Database URL. Use `postgresql+psycopg2://user:pass@host:7777/warden` for Postgres |
| `WARDEN_ADMIN_USERNAME` / `WARDEN_ADMIN_PASSWORD` | `admin` / generated | Used by `serve` to create the first admin when the users table is empty |
| `WARDEN_AUTH_ENABLED` | `true` | When `false`, `/login` redirects straight to the dashboard. `/healthz`, `/status.json` and `/metrics` are always public |
| `WARDEN_WEB_HOST` | `127.0.0.1` | Web UI bind host (`--web-host` overrides it) |
| `WARDEN_WEB_PORT` | `8080` | Web UI bind port (`--web-port` overrides it) |
| `CAM_USER`, `CAM_PASS`, any name | | Referenced from `config.yaml` as `${NAME}`; a missing variable is a startup error |
| `WARDEN_HTTPS` | `false` | Set true for secure cookies (behind a TLS-terminating reverse proxy) |
| `TZ` | `UTC` | Timezone for log timestamps and segment filenames |

## CLI reference

```bash
rtsp-warden [OPTIONS] COMMAND [ARGS]
```

| Command | Description |
|---|---|
| `install` | Optional first-run setup: writes `.env` (DB URL, admin credentials), creates the schema and admin user |
| `init-config` | Write a starter `config.yaml` (default `./config.yaml`; `--force` to overwrite) |
| `doctor` | Validate config + check ffmpeg/mediamtx availability + check ports (does not contact cameras) |
| `serve` | Create the schema and first admin if needed, then start the recorder + web UI + health endpoints |
| `status` | Print a JSON status snapshot |
| `version` | Print the package version |

### `serve` flags

| Flag | Default | Description |
|---|---|---|
| `-c, --config PATH` | (required) | Path to config file |
| `--web` / `--no-web` | `--web` | Enable/disable the web UI |
| `--web-host HOST` | `$WARDEN_WEB_HOST`, else 127.0.0.1 | Web UI bind host |
| `--web-port PORT` | `$WARDEN_WEB_PORT`, else 8080 | Web UI bind port |
| `--detectors` / `--no-detectors` | `--detectors` | Enable/disable the detector framework |
| `-v, --verbosity LEVEL` | `info` | error, warning, info or debug |

## Web UI routes

| Path | Auth | Description |
|---|---|---|
| `GET /` | user | Dashboard with live status, today's detection count, and camera grid |
| `GET /login` / `POST /login` | none | Login (rate-limited to 5/min) |
| `POST /logout` | user | Logout |
| `GET /cameras` | user | Camera list |
| `GET /cameras/new` | admin | Add-camera form |
| `POST /cameras/new/test` | admin | Test main or sub stream: ffprobe plus one snapshot (htmx fragment) |
| `POST /cameras/new/onvif` | admin | Find stream URLs over ONVIF `GetStreamUri` (htmx fragment) |
| `POST /cameras` | admin | Save a new camera to `config.yaml` and start it |
| `GET /cameras/{name}/edit` / `POST` | admin | Edit a camera's stream URLs, login, recording switch and ONVIF port |
| `POST /cameras/{name}/delete` | admin | Remove a camera from `config.yaml` and stop it (recordings are kept) |
| `GET /cameras/{name}/status-row` | user | Status row of the camera page (htmx partial, polled every 5 s) |
| `GET /cameras/{name}` | user | Camera detail: live status, live MJPEG, recent recordings, detector list, zone/sensitivity controls |
| `GET /cameras/{name}/live.mjpeg` / `snapshot.jpg` | user | Same-origin live MJPEG stream and latest JPEG frame |
| `GET /cameras/{name}/status` | user | Camera card (htmx partial, auto-refresh) with running / restarting / failed status |
| `GET /cameras/{name}/detectors` | user | Detector list (htmx partial, auto-refresh) |
| `POST /cameras/{name}/retention` | admin | Per-camera retention policy (`action=reset` clears it) |
| `GET /cameras/{name}/zones` / `POST` | admin | Grid-based detection zone editor (web UI) |
| `POST /cameras/{name}/zones/{name}/delete` | admin | Delete a zone |
| `GET /cameras/{name}/sensitivity` / `POST` | admin | Per-camera sensitivity (0-100) |
| `GET /cameras/{name}/detection-classes` / `POST` | admin | Per-camera detection class list |
| `POST /cameras/{name}/detectors/{type}/enabled` | admin | Toggle a detector on/off |
| `POST /cameras/{name}/reload` | admin | Rebuild that camera's detectors from the in-memory config (no restart, no YAML re-read) |
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
| `GET /alerts` / `new` / `{name}/edit`, `POST /alerts/{name}/test` | admin | Notifier list and test button; notifiers are edited in `config.yaml` |
| `GET /onvif` / `POST /onvif/discover` | admin | ONVIF camera discovery |
| `GET /onvif/cameras/{name}/ptz` | admin | Redirects (303) to `/onvif?camera={name}` |
| `POST /onvif/cameras/{name}/ptz` | admin | PTZ move or stop (form `direction`, `duration_ms`; htmx fragment) |
| `GET /onvif/cameras/{name}/presets` | admin | Presets panel (htmx fragment) |
| `GET /onvif/events` | admin | Event subscription table (htmx fragment) |
| `POST /onvif/cameras/{name}/events/subscribe` / `unsubscribe` | admin | ONVIF event subscription |
| `POST /onvif/cameras/{name}/ptz/goto` / `save` / `delete` | admin | PTZ presets (form field `preset_name`) |
| `GET /htl/{cam}/{stream}/{start}/{end}.m3u8` | user | Dynamic HLS playlist for a time window |
| `GET /segments/{cam}/{stream}/{path:path}` | user | Serve a TS segment file |
| `GET /api/recordings/{id}/timeline` | user | JSON timeline data for the canvas scrubber |
| `GET /healthz` / `/status.json` | none | Liveness / full status JSON (always public) |
| `GET /health` / `/health/partial` | user | Health page and its htmx partial |
| `GET /metrics` | none | Prometheus metrics |

**Auth legend:** `none` = always public; `user` = any authenticated user; `admin` = admin role required.

## Concepts

### Recording

Each camera has a `main` stream and an optional `sub` stream. Each stream runs an `ffmpeg` subprocess writing time-segmented files with ffmpeg's `segment` muxer; HLS playlists for playback are synthesized at request time. Default container is `.ts` (MPEG-TS), which is NVR-grade and survives camera stream restarts that would corrupt `.mp4` or `.mkv`.

**Modes** (per camera):
- `continuous` (default): always recording
- `event`: records only when a detection event is recent (1s polling loop reads the events table; segments start at the moment of trigger)

**Audio** is opt-in per camera (`record.audio: true`). Adds `-c:a aac -b:a 128k` to ffmpeg.

**Retention** resolves as `cameras[].retention` > global `retention`. A `retention:` nested under `record:` (the old sample layout) is still honored with a deprecation warning. Files older than `max_days`, or beyond `max_gb`, or older than the `keep_last_n`-th file are cleaned at `cleanup_interval_seconds`.

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

**Hot reload:** web UI saves write `config.yaml`. A detector `enabled` toggle rebuilds that camera's detectors at once; sensitivity and detection classes rebuild when saved with "Save and reload"; zones rebuild from the zones page's reload button. `POST /cameras/{name}/reload` rebuilds from the in-memory config; it does not re-read the YAML, so hand edits still need a restart.

### Alerts

`AlertManager` debounces by `(notifier, camera, event_type)` with each notifier's `min_interval_seconds`, and filters by severity (`info`, `warn`, `error`; ntfy and webhook default to `warn` and `error`). Today the only thing that reaches a notifier is the **Test** button on the Alerts page: detector events are not yet wired to notifiers. That wiring is the detection sub-project in `docs/superpowers/specs/`.

Notifier types:
- `ntfy` — push to an ntfy topic
- `webhook` — generic HTTP POST with JSON body
- `apprise` — any of apprise's 90+ services (email via SMTP, Discord, Telegram, Slack, Pushover, etc.)

### ONVIF

- **Discovery:** WS-Discovery UDP multicast to `239.255.255.250:3702`. **Note:** this does not cross Docker bridge networks; use `network_mode: host` for Docker deployments.
- **Stream URLs:** **Find stream URLs (ONVIF)** on the add-camera form calls `GetStreamUri` (WS-UsernameToken login) on ports 80, 8080, 888 and 2020 and fills in the port that answered as the camera's `onvif_port`. See [Adding a camera](#adding-a-camera).
- **PTZ:** absolute_move, continuous_move, stop, from the PTZ pad on the ONVIF page. PTZ and event calls go to `http://<host of main_url>:<onvif_port>/onvif/device_service` (`onvif_port` defaults to 80) with the global `onvif.username` / `onvif.password` over HTTP Digest auth. Presets are saved in `config.yaml` under `camera.presets`; a save changes only that camera's `presets` list in the raw YAML, so `${VAR}` references in the file survive.
- **Events:** pull-point subscription over SOAP, started manually from the ONVIF page. Received events are logged; they are not yet turned into alerts.

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

The unit is hardened: `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `PrivateDevices`, `ReadWritePaths` limited to the data and log directories plus `/etc/rtsp-warden` (the web UI saves `config.yaml` and `.env` there). See [packaging/systemd/README.md](packaging/systemd/README.md).

## Development

```bash
# Clone
git clone https://github.com/Veedubin/RSTP-Warden.git
cd RSTP-Warden

# Install runtime + dev dependency group (uses uv)
uv sync

# Run tests
uv run pytest                  # ~820 tests, ~70s, fully offline
uv run pytest tests/test_X.py  # single file
uv run pytest -k "pattern"     # by name

# Lint
uv run ruff check src/ tests/  # 0 errors in new code; 10 pre-existing E501 lines are the accepted baseline
uv run ruff format src/ tests/

# Type check (informal; not in CI yet)
uv run mypy src/ || true
```

The test suite uses `asyncio_mode = "auto"` and is fully self-contained — no live cameras, no ffmpeg binary, no real network. Each test gets a tmp directory and an isolated SQLite DB.

## Architecture invariants

These are the stable contracts other code depends on. Do not break them in PRs:

1. **`FrameConsumer` protocol** — `on_frame(camera, stream, jpeg_bytes, ts_unix)`. Receivers are chained; exceptions are caught and logged (they never propagate to ingest).
2. **`Detector` protocol** — `name`, `kind`, `setup()`, `process(frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]`, `teardown()`. Receives an already-masked BGR frame; ROI and grid-zone filtering happen in the runner afterwards.
3. **`Notifier` protocol** — `name`, `type`, `async send(event: dict) -> NotificationResult`, `async test() -> NotificationResult`. Failures are returned, not raised.
4. **Config authority:** `config.yaml` is authoritative; the DB never stores camera config. Web UI saves write the YAML through a file lock and hot-reload detectors; hand edits take effect on restart.
5. **Recording is additive** — adding new consumers, notifiers, detectors, or web routes must not destabilize the ingest path.

## License

MIT. See [LICENSE](LICENSE).

## Brand assets

The project logo lives at [`docs/assets/rtsp-warden-logo.png`](docs/assets/rtsp-warden-logo.png) (1794×1794, 3.3 MB PNG). Use this as the source for the GitHub social preview — upload a resized variant (1280×640 recommended) via **Settings → Social preview** on the repo page.

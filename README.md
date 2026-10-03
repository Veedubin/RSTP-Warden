# rtsp-warden

<p align="center">
  <img src="docs/assets/rtsp-warden-logo.png" alt="rtsp-warden logo" width="320">
</p>

A self-hosted Network Video Recorder (NVR) for RTSP cameras. Records continuously, detects people, vehicles and animals in-process (YOLOX on ONNX Runtime, on the CPU or an NVIDIA GPU), turns each object visit into one event with a thumbnail, and notifies you through per-camera rules and actions (ntfy, Apprise, webhooks). A web UI covers live viewing, events and admin. No cloud, no subscriptions, no agent on each camera.

## What it does

- **Records** RTSP streams to disk as time-segmented `.ts` files (NVR-grade, recovers from bad SDPs)
- **Detects** people, vehicles, animals and the rest of the 80 COCO classes with YOLOX on [ONNX Runtime](https://onnxruntime.ai), plus cheap motion detection; CPU by default, NVIDIA GPU optional
- **Tracks** each object across frames, so one person crossing the yard is one event, with a thumbnail of the best frame
- **Rules and actions** per camera (label, zone, confidence, time window, cooldown) send to [ntfy](https://ntfy.sh), generic webhooks, or [Apprise](https://github.com/caronc/apprise) (email, Discord, Telegram, Slack, Pushover, 90+ services), thumbnail attached
- **ONVIF** camera discovery + PTZ control + event subscription (motion alarms, tampering)
- **Clips** cut automatically from the recording around an event when a rule asks for one (MP4, `.ts` fallback)
- **Zones** are grid-based (toggle cells like your home security app): `ignore` zones drop detections (the "block the road" use case), named `area` zones let a rule ask for "car in the driveway"
- **Tuning** per camera: detection rate (`detect_fps`, per-detector `fps`), sensitivity (0-100), detection classes, enable/disable per detector
- **Multi-user** with bcrypt sessions, bearer API tokens, and admin/viewer roles

## Why this and not ZoneMinder / Shinobi / Frigate?

| | rtsp-warden | Frigate | ZoneMinder | Shinobi |
|---|---|---|---|---|
| **Stack** | Python stdlib + FastAPI | Python + Go + Coral/Google Coral | Perl + MySQL | Node.js |
| **Web UI** | Server-rendered (htmx + Alpine), no build step | React + Vite (heavy build) | Legacy jQuery | Angular (heavy build) |
| **Database** | SQLite default, Postgres optional | SQLite only | MySQL required | SQLite/MySQL |
| **Auth** | Multi-user, bcrypt, roles, API tokens, CSRF | None built-in | Basic | Basic |
| **Detection** | YOLOX on ONNX Runtime (CPU or NVIDIA GPU), motion, custom detectors | YOLO (always, needs GPU) | Zoneminder motion | Plugin-based |
| **Clips from events** | Automatic per rule, MP4 (`.ts` fallback), no re-encode | Built-in | Manual | Plugin |
| **ONVIF events** | Built-in (pull-point) | No | Limited | Limited |
| **Docker image** | 685 MB distroless default, 1.2 GB slim | ~1.5 GB | ~500 MB | ~400 MB |
| **Tests** | 1500+ | Many (Python) | Perl | Limited |
| **License** | MIT | MIT | GPLv2 | GPLv3 |

rtsp-warden targets homelab/self-hosters who want a feature-complete NVR with a sensible admin UI, no JS build step, no GPU requirement (an NVIDIA GPU is optional), and a small Docker footprint. It is not the fastest, and it does not have the largest community.

## Quick start

### Option A: pip (any Linux)

Needs Python 3.11 or newer (ONNX Runtime, a core dependency, ships no Python 3.10 wheels after 1.23).

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
[ -e .env ] || (umask 077 && printf 'CAM_USER=admin\nCAM_PASS=admin\n' > .env)
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
printf 'CAM_USER=admin\nCAM_PASS=admin\n' > .env

# Run
docker compose up -d
docker compose logs warden | grep "created admin"   # first-start admin password
# Browse to http://localhost:3333 (compose maps host port 3333; set WARDEN_HOST_PORT in .env to change it)
```

The distroless image is 685 MB. See [docker/README.md](docker/README.md) for the slim alternative, ONVIF/UDP notes, and volume-mounting gotchas.

The example config detects with YOLOX: its model (about 34 MB) is downloaded into `./data/models` on the first frame, so the first start needs internet access. Set `runtime.public_url` in `config/config.yaml` to the address your phone uses (for example `http://nvr.lan:8080`) so links in notifications work. For an NVIDIA GPU, see [GPU (NVIDIA)](#gpu-nvidia).

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
        |
        v
   StreamIngestor (1 per camera stream)
   |- ffmpeg subprocess
   |- writes .ts segments
   '- frame tap (preview stream only, at detect_fps)
                |
                v
   DetectorRunner (1 per camera, 1 worker, drops the oldest frame under load)
   |- MotionDetector (MOG2) -----------------> motion events
   '- OnnxDetector (YOLOX, CUDA or CPU)
        -> Tracker (one track per object visit)
        -> EventBuilder (events row + thumbnail)
        -> RuleEngine -> ActionQueue -> ntfy / webhook / apprise
                      '-> ClipScheduler -> <camera>/clips/<event_id>.mp4
                            |
                            v
                 SQLite or PostgreSQL
                            |
                            v
                    FastAPI web UI
                    (htmx + Alpine + Pico)
```

One `ffmpeg` process per configured stream (`main`, plus `sub` when set). The stream that feeds the live preview also writes the frame tap: JPEG frames at the camera's `detect_fps`, sent through a pipe to that camera's detector runner. Object detections are tracked across frames, each tracked object becomes one event with a thumbnail, and the camera's rules decide which actions hear about it. The web UI reads the same database.

## Configuration

### `config.yaml` (canonical example)

See [examples/config.yaml](examples/config.yaml) for a minimal single-camera config, or [examples/configs/](examples/configs/) for full real-world configs (Foscam, TP-Link, 2-camera NC230).

The config has six top-level sections:

```yaml
cameras:        # list of CameraConfig (required, at least one); detection and rules are per camera
runtime:        # global runtime settings (ffmpeg path, restart policy, models_dir, public_url)
actions:        # named notification targets that rules send to (ntfy, webhook, apprise)
clips:          # clip window for rules with `clip: true`
onvif:          # ONVIF global config (v1.1.0+)
retention:      # global fallback retention (v1.2.0+; per-camera overrides supported)
```

The old `alerts:` section still loads for one release; see [Upgrading from 1.3](#upgrading-from-13).

### Camera config (every field)

<!-- config-example: camera-reference -->
```yaml
cameras:
  - name: front_door                       # required, unique
    main_url: rtsp://admin:admin@host:554/... # required
    sub_url:  rtsp://admin:admin@host:554/... # optional; proxy and frame tap fall back to main, sub recording is skipped
    # Credentials can come from the environment: rtsp://${CAM_USER}:${CAM_PASS}@host:554/...

    record:
      enabled: true
      mode: continuous                     # continuous | event (see Recording below)
      audio: false                         # opt-in audio recording (v1.1.0+)
      output_dir: ./recordings             # also holds <camera>/thumbnails/ and <camera>/clips/
      main: { container: ts, chunk_seconds: 300, rtsp_transport: tcp }
      sub:  { container: ts, chunk_seconds: 300, rtsp_transport: tcp }

    retention:                              # per-camera override of the global `retention:` block
      max_days: 30                          # (a `retention:` nested under `record:` is deprecated;
      max_gb: 50.0                          #  it is still honored, with a warning in the log)
      keep_last_n: 100                      # counts video segments only
      cleanup_interval_seconds: 300

    proxy:
      enabled: true
      mode: mjpeg                          # mjpeg | rtsp
      stream: sub                          # this stream also feeds the detectors
      bind_host: 0.0.0.0
      port: 9001
      fps: 7                               # mjpeg mode (preview rate, not the detection rate)
      scale_width: 0                       # mjpeg mode (0 = no scaling)

    detect_fps: 5                          # frames/s sent to the detectors, 0.5-30 (a change restarts ingest)
    track_grace_seconds: 3.0               # an object may vanish this long before its event ends
    min_track_frames: 2                    # matched frames in a row before an object becomes an event
    stationary_iou: 0.6                    # an object that never moved makes no event (0 = off)

    detectors:                             # list; the web UI identifies a detector by its position
      - type: motion                       # MOG2 background subtraction
        enabled: true
        fps: 5                             # optional, at most detect_fps (default: detect_fps)
        min_area: 500
        # events: true                     # store motion events even with an onnx detector enabled
      - type: onnx                         # YOLOX on ONNX Runtime
        enabled: true
        model: yolox-s                     # yolox-s (default), yolox-nano, or a model in runtime.models_dir
        device: auto                       # auto | cuda | cpu
        fps: 2
        min_confidence: 0.5                # default: from sensitivity (0.5 at 50)

    sensitivity: 50                        # 0-100: motion threshold and the default min_confidence
    detect_classes: [person, car, dog, cat] # model labels to keep (omit = all); unknown names are an error

    zones:                                  # grid zones (v1.2.0+)
      - name: road
        kind: ignore                        # default: drop detections centred in a blocked cell
        grid_cols: 16
        grid_rows: 16
        frame_width: 1920
        frame_height: 1080
        blocked_cells: [[0, 0], [1, 0], [2, 0]]
      - name: driveway
        kind: area                          # names the cells NOT listed; never drops anything
        grid_cols: 2
        grid_rows: 2
        frame_width: 1920
        frame_height: 1080
        blocked_cells: [[1, 0], [1, 1]]     # the area is the left half

    rules: []                              # see Rules and actions below

    presets:                               # PTZ presets (v1.1.0+); ONVIF credentials are global (`onvif:` below)
      - name: front_gate
        pan: 0.0
        tilt: 0.0
        zoom: 0.5

    events:                                # ONVIF event subscriptions (v1.1.0+), not detection events
      - type: motion
        min_interval_seconds: 30
      - type: tamper
        min_interval_seconds: 60
```

### Global sections

<!-- config-example: global-sections -->
```yaml
runtime:
  ffmpeg_path: ffmpeg
  mediamtx_path: mediamtx           # only needed for proxy.mode: rtsp
  workspace_dir: ./workspace
  # models_dir: /srv/warden/models  # ONNX models; default $WARDEN_MODELS_DIR, else $XDG_CACHE_HOME/rtsp-warden/models
  public_url: http://nvr.lan:8080   # base of links in notifications (default: web bind address, 0.0.0.0 -> localhost)
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
  password: ${ONVIF_PASS}           # RTSP host, cameras[].onvif_port (default 80), /onvif/device_service
actions:                            # see Rules and actions below
  - name: phone
    type: ntfy
    url: https://ntfy.sh
    topic: ${NTFY_TOPIC}            # put NTFY_TOPIC=<long random string> in .env; ntfy.sh topics are public
    token: ${NTFY_TOKEN}            # optional
  - name: homeassistant
    type: webhook
    url: http://ha.local:8123/api/webhook/warden
    method: POST                    # POST | PUT
    headers: {}
  - name: email
    type: apprise
    urls: ["mailtos://${SMTP_USER}:${SMTP_PASS}@gmail.com"]
clips:                              # used by rules with `clip: true`
  pre_seconds: 10                   # before the event starts
  post_seconds: 10                  # after it ends
  max_duration: 120                 # longest clip, in seconds
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
| `WARDEN_MODELS_DIR` | `$XDG_CACHE_HOME/rtsp-warden/models`, else `~/.cache/rtsp-warden/models` | Where ONNX models are kept and downloaded to; `runtime.models_dir` in `config.yaml` wins. Compose sets `/app/data/models`. Under systemd the service user has no writable home, so put `WARDEN_MODELS_DIR=/var/lib/rtsp-warden/models` into `/etc/rtsp-warden/warden.env` (inside the unit's `ReadWritePaths`) |
| `WARDEN_HTTPS` | `false` | Set true for secure cookies (behind a TLS-terminating reverse proxy) |
| `TZ` | `UTC` | Timezone for log timestamps, segment filenames and rule `between` windows |

## CLI reference

```bash
rtsp-warden [OPTIONS] COMMAND [ARGS]
```

| Command | Description |
|---|---|
| `install` | Optional first-run setup: writes `.env` (DB URL, admin credentials), creates the schema and admin user |
| `init-config` | Write a starter `config.yaml` (default `./config.yaml`; `--force` to overwrite) |
| `doctor` | Validate config + check ffmpeg/mediamtx availability + check ports (does not contact cameras); warns when a detector asks for `device: cuda` but the GPU build of ONNX Runtime is not installed |
| `serve` | Create the schema and first admin if needed, then start the recorder + web UI + health endpoints |
| `status` | Print a JSON status snapshot (per camera: streams, and the detection provider, warnings and frame counts) |
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
| `GET /cameras/{name}/detectors` | user | Detection panel rows: each detector with its device and provider badge, fps, processed and dropped frames (htmx partial, auto-refresh) |
| `POST /cameras/{name}/retention` | admin | Per-camera retention policy (`action=reset` clears it) |
| `GET /cameras/{name}/zones` / `POST` | admin | Grid-based detection zone editor (web UI) |
| `POST /cameras/{name}/zones/{name}/delete` | admin | Delete a zone |
| `GET /cameras/{name}/sensitivity`, `POST /cameras/{name}/sensitivity` | admin | Per-camera sensitivity (0-100) |
| `GET /cameras/{name}/detection-classes`, `POST /cameras/{name}/detection-classes` | admin | Per-camera class filter built from the model's labels ("all" or a custom list) |
| `GET /cameras/{name}/detection` | user | Detection panel (htmx partial): `detect_fps`, tracking, detectors with their live provider, rules, "Fire test event" |
| `POST /cameras/{name}/detectors/{index}/enabled` | admin | Toggle one detector (by its position in `detectors:`) on/off; writes back only that entry |
| `POST /cameras/{name}/detectors/{index}/fps` | admin | Set one detector's own `fps` (at most the camera's `detect_fps`; empty = the camera's rate); a hot reload, no ingest restart |
| `POST /cameras/{name}/detection` | admin | Save `detect_fps`, `track_grace_seconds`, `min_track_frames` and `stationary_iou` (a new `detect_fps` restarts the camera's ingest) |
| `POST /cameras/{name}/rules/test` | admin | "Fire test event": a synthetic `person` event through the camera's real rules and actions |
| `GET /cameras/{name}/live-boxes.mjpeg` | user | Live MJPEG with the tracker's current boxes drawn (the "show boxes" switch) |
| `POST /cameras/{name}/reload` | admin | Rebuild that camera's detectors from the in-memory config (no YAML re-read; restarts ingest only when the frame-tap settings change) |
| `GET /events` | user | Event cards (thumbnail, label, confidence, camera, zone, time) with camera, label and date filters; htmx refresh |
| `GET /events/{id}` | user | Event detail: full thumbnail, clip player, action runs |
| `GET /events/{id}/thumbnail.jpg` | user | The event's thumbnail (404 once retention removed it) |
| `GET /events/{id}/clip` | user | The event's clip: the MP4 file, or an HLS player for a `.ts` clip |
| `GET /events/{id}/clip.m3u8` | user | One-entry HLS playlist for a `.ts` clip |
| `GET /users` / `POST /users/new` | admin | User management |
| `POST /users/{id}/reset-password` / `delete` / `toggle-admin` | admin | User actions |
| `GET /api-tokens` / `POST` / `POST .../revoke` | user | API token management (bearer) |
| `GET /settings` | admin | System settings (read-only display) |
| `GET /actions`, `POST /actions/{name}/test` | admin | Actions with last run and failure count, and a Test button each; actions are edited in `config.yaml` |
| `GET /onvif` / `POST /onvif/discover` | admin | ONVIF camera discovery |
| `GET /onvif/cameras/{name}/ptz` | admin | Redirects (303) to `/onvif?camera={name}` |
| `POST /onvif/cameras/{name}/ptz` | admin | PTZ move or stop (form `direction`, `duration_ms`; htmx fragment) |
| `GET /onvif/cameras/{name}/presets` | admin | Presets panel (htmx fragment) |
| `GET /onvif/events` | admin | Event subscription table (htmx fragment) |
| `POST /onvif/cameras/{name}/events/subscribe` / `unsubscribe` | admin | ONVIF event subscription |
| `POST /onvif/cameras/{name}/ptz/goto` / `save` / `delete` | admin | PTZ presets (form field `preset_name`) |
| `GET /htl/{cam}/{stream}/{start}/{end}.m3u8` | user | Dynamic HLS playlist for a time window |
| `GET /segments/{cam}/{stream}/{path:path}` | user | Serve a TS segment file |
| `GET /healthz` / `/status.json` | none | Liveness / full status JSON (always public) |
| `GET /health` / `/health/partial` | user | Health page and its htmx partial |
| `GET /metrics` | none | Prometheus metrics |

**Auth legend:** `none` = always public; `user` = any authenticated user; `admin` = admin role required.

## Concepts

### Recording

Each camera has a `main` stream and an optional `sub` stream. Each stream runs an `ffmpeg` subprocess writing time-segmented files with ffmpeg's `segment` muxer; HLS playlists for playback are synthesized at request time. Default container is `.ts` (MPEG-TS), which is NVR-grade and survives camera stream restarts that would corrupt `.mp4` or `.mkv`.

**Modes** (per camera):
- `continuous` (default): always recording
- `event`: meant to record only while a detection event is recent (1s polling loop on the events table). **Limitation:** the detectors read their frames from the same ffmpeg process that event mode stops, so detections cannot start a recording; use `continuous` with retention instead

**Audio** is opt-in per camera (`record.audio: true`). Adds `-c:a aac -b:a 128k` to ffmpeg.

**Retention** resolves as `cameras[].retention` > global `retention`. A `retention:` nested under `record:` (the old sample layout) is still honored with a deprecation warning. Files older than `max_days`, or beyond `max_gb`, or older than the `keep_last_n`-th file are cleaned at `cleanup_interval_seconds`. Event thumbnails and clips under `<output_dir>/<camera>/` follow the same `max_days` and `max_gb` rules, while `keep_last_n` counts video segments only; when a thumbnail or clip is gone, the event stays and the UI shows "expired".

### Detection

Each camera's preview stream (`proxy.stream`; the main stream when there is no `sub_url`) also feeds the detectors: ffmpeg writes JPEG frames at the camera's `detect_fps` (default 5, from 0.5 to 30), scaled to the widest model input (at least 320 px), into a pipe read by that camera's detector runner. The runner works on one thread with a small queue; when inference is slower than the frame rate it drops the oldest queued frame and counts it, so overload means fewer samples, never growing memory.

| Detector | What | Notes |
|---|---|---|
| `onnx` | YOLOX on ONNX Runtime, 80 COCO classes | The object detector. `model: yolox-s` (default, 640 px input, about 34 MB) or `yolox-nano` (416 px, about 3.5 MB, for small CPUs) |
| `motion` | MOG2 background subtraction | Very cheap. Its events are stored only when no `onnx` detector is enabled, or with `events: true` |
| `custom` | Your class via `import_path: module:Class` | Implements the `Detector` protocol (see Architecture invariants) |
| `person`, `vehicle`, `dnn` | HOG, Haar cascade, YOLOv4-tiny | Deprecated: each logs a warning naming `onnx`, and they are removed in the next release |

**Rates.** `detect_fps` is how many frames per second the camera's detectors receive. Changing it restarts that camera's ingest, because it is an ffmpeg argument (the Detection panel says so next to its Save button). A detector can run slower with its own `fps`, at most `detect_fps`; the runner skips frames for it. Expected cost at 2 fps with YOLOX-s: 10 to 15 ms per frame on an RTX 4080-class GPU, 60 to 120 ms on one modern CPU core.

**Models.** `yolox-s` and `yolox-nano` ship as descriptors inside the package. Their `.onnx` files are downloaded on first use into `runtime.models_dir` (default: `$WARDEN_MODELS_DIR`, else `$XDG_CACHE_HOME/rtsp-warden/models`, else `~/.cache/rtsp-warden/models`) and checked against a pinned SHA-256. Without internet access the camera keeps recording, its Detection panel shows the error, and the download is retried every 5 minutes; you can also copy the file to `<models_dir>/yolox-s/yolox_s.onnx` by hand. To add your own YOLOX-family model, create `<models_dir>/<name>/model.yaml` and set `model: <name>`:

```yaml
name: my-yolox
file: my_yolox.onnx          # in the same directory
labels: labels.txt           # one label per line; line number = class id
input_size: [640, 640]       # width, height (multiples of 32)
postprocess: yolox           # the only supported value
sha256: <64 hex characters>  # optional
url: https://example.com/my_yolox.onnx   # optional: downloaded when the file is missing
```

**Device.** `device: auto` (the default) uses CUDA when the GPU build of ONNX Runtime can load it, else the CPU. `cuda` asks for the GPU and falls back to the CPU with a warning badge on the camera. `cpu` never touches CUDA. The log names the provider in use once the model is loaded:

```
onnx detector onnx (model yolox-s, device auto) provider: CUDAExecutionProvider
```

**Classes and tuning per camera:**
- `detect_classes` — model labels to keep (for YOLOX, COCO names such as `person`, `car`, `dog`, `cat`); omit it for all labels. Unknown names fail config validation with the list of valid ones. The detection classes page offers "all" or a custom list.
- `min_confidence` per `onnx` detector — when unset it comes from `sensitivity` (0.5 at 50).
- `sensitivity` (0-100) — single knob that also scales the motion detector's threshold. Higher = more sensitive.
- `enabled` per detector — toggle individual detectors on/off without deleting config. The web UI identifies a detector by its position in `detectors:` and writes back only that entry, so other keys and `${VAR}` references survive.
- **Zones** — grid zones of N×M cells. `kind: ignore` (the default) drops detections whose box centre is in a blocked cell. `kind: area` never drops anything: its active (not blocked) cells form a named area, and an event's `zone` is the first `area` zone that contains the centre of its best box. Cells map onto the frame the detectors actually see, whatever the zone's saved `frame_width`/`frame_height`. A detector also accepts a polygon `roi`; a detection must pass the `roi` and every `ignore` zone.

**Hot reload:** web UI saves write `config.yaml`. A detector `enabled` toggle rebuilds that camera's detectors at once; sensitivity and detection classes rebuild when saved with "Save and reload"; zones rebuild from the zones page's reload button. `POST /cameras/{name}/reload` rebuilds from the in-memory config; it does not re-read the YAML, so hand edits still need a restart. A rebuild restarts the tracker empty, and restarts the camera's ingest only when the frame-tap settings change (`detect_fps`, or a model with another input width).

**Live boxes:** the "show boxes" switch next to the camera's live preview (off by default, remembered per browser) streams `/cameras/{name}/live-boxes.mjpeg`, with the tracker's current boxes drawn.

**Status:** the camera card and the `/health` page show the provider in use (`CUDAExecutionProvider` or `CPUExecutionProvider`), a warning when CUDA was requested but not used or a model failed to load, and processed / dropped frame counts. `/status.json` and `rtsp-warden status` carry the same values per camera.

### Events

Boxes from object detectors are matched across frames by a per-camera tracker (intersection over union, per label). A track that is matched on `min_track_frames` frames in a row (default 2) becomes one row in the `events` table: camera, label, best confidence, zone, and a JPEG thumbnail of the best frame with its box drawn (`<output_dir>/<camera>/thumbnails/<event_id>.jpg`). While the object stays in view, the row and thumbnail are updated when a better frame arrives (at most once per second). Once the tracker has not seen the object for `track_grace_seconds` (default 3), the event gets its end time. Two people walking past are two events; a person standing still for ten minutes is one.

**Stationary objects.** A track whose box still overlaps the box it was first seen at by at least `stationary_iou` (default 0.6) has not moved, so it makes no event: a parked car, a chair, or a dark shape the model mistakes for a microwave every time the light changes. It is held instead, and opens as an event the moment its box drifts away from where it started (the event then starts at that frame). A camera restart or the lights coming on therefore no longer makes an event out of everything in view. `stationary_iou: 0` turns the check off. The Detection panel edits the value and shows how many objects were held back.

Motion makes one event per burst (it opens after `min_track_frames` frames with motion and ends after `track_grace_seconds` without), with no thumbnail, and only when the camera has no enabled `onnx` detector or the motion detector sets `events: true`.

The Events page shows cards (thumbnail, label, confidence, camera, zone, time) with filters for camera, label and date; an event's page shows the full thumbnail, its clip and every action run. Events made with "Fire test event" carry a **test** badge.

### Rules and actions

Actions are named notification targets in the top-level `actions:` list. Rules are per camera and decide which events reach which actions. A complete example (the test suite validates it):

<!-- config-example: detection-example -->
```yaml
cameras:
  - name: yard
    main_url: rtsp://${CAM_USER}:${CAM_PASS}@192.168.1.60:554/stream1
    detect_fps: 5
    detectors:
      - type: motion
        fps: 5
      - type: onnx
        model: yolox-s
        device: auto
        fps: 2
        min_confidence: 0.5
    zones:
      - name: driveway
        kind: area
        grid_cols: 2
        grid_rows: 2
        frame_width: 1920
        frame_height: 1080
        blocked_cells: [[1, 0], [1, 1]]   # the driveway is the left half
      - name: road
        kind: ignore
        grid_cols: 2
        grid_rows: 2
        frame_width: 1920
        frame_height: 1080
        blocked_cells: [[0, 0], [1, 0]]   # ignore the top half (the road)
    rules:
      - name: person-any-time
        labels: [person]
        zones: []                 # empty = any zone
        min_confidence: 0.6
        between: null             # or "22:00-06:00", local time, may wrap midnight
        cooldown_seconds: 60
        clip: true
        actions: [phone]
      - name: car-in-driveway-at-night
        labels: [car]
        zones: [driveway]
        between: "22:00-06:00"
        actions: [phone, homeassistant]

actions:
  - name: phone
    type: ntfy
    url: https://ntfy.sh
    topic: ${NTFY_TOPIC}          # put NTFY_TOPIC=<long random string> in .env; ntfy.sh topics are public
    token: ${NTFY_TOKEN}          # optional
  - name: homeassistant
    type: webhook
    url: http://ha.local:8123/api/webhook/warden

runtime:
  public_url: http://nvr.lan:8080   # base of the links in notifications
```

A rule matches an event when its label is in `labels` (empty = any), its zone is in `zones` (empty = any; the names must be `area` zones of that camera), its confidence is at least `min_confidence`, and the local time is inside `between` (`"HH:MM-HH:MM"`, may wrap midnight; the server's time zone, `TZ` in Docker). Rules are evaluated when the event opens, so notifications are prompt. For `cooldown_seconds` after a rule fired for a camera and label, further matches are recorded on the event (`suppressed_by` in its metadata) but send nothing. Unknown action or zone names in a rule are config errors, and so are labels that none of the camera's `onnx` models know (`motion` is always allowed).

Every action gets the same payload: `camera`, `label`, `confidence`, `zone`, `started_at`, `ended_at` (null while the event is open), `thumbnail_url`, `clip_url` (null until a clip exists), `event_url` and `test`. The URLs are absolute. Set `runtime.public_url` to the address you open the web UI with from your phone; without it they are built from the web UI's bind address, with `0.0.0.0` shown as `localhost`, which only works on the server itself.

| Type | Fields | What it sends |
|---|---|---|
| `ntfy` | `url`, `topic`, `token` (optional) | A push with the thumbnail attached; the token goes in the `Authorization` header |
| `webhook` | `url`, `method` (`POST` or `PUT`), `headers` | The payload as JSON |
| `apprise` | `urls` (a list of [Apprise URLs](https://github.com/caronc/apprise/wiki)) | Title and text with the thumbnail attached: email, Discord, Telegram, Slack, MQTT and 90+ more |

Each run is stored in `action_runs` and shown on the event's page and, as last run and failure count, on the Actions page. Failed runs are not retried. Keep secrets in the environment: `token: ${NTFY_TOKEN}` in `config.yaml` and `NTFY_TOKEN=...` in `.env`; with Docker, also pass it to the container under `environment:` in `docker-compose.yml` (`NTFY_TOKEN: ${NTFY_TOKEN}`).

**Testing.** The Actions page has a **Test** button per action that sends a synthetic payload straight through that action and shows the result (it writes no `action_runs` row). On a camera's Detection panel, **Fire test event** creates a real `person` event (type `test`, placeholder thumbnail), runs it through that camera's rules with cooldowns ignored and through the action queue, and lists the rules that matched; it never makes a clip.

### ONVIF

- **Discovery:** WS-Discovery UDP multicast to `239.255.255.250:3702`. **Note:** this does not cross Docker bridge networks; use `network_mode: host` for Docker deployments.
- **Stream URLs:** **Find stream URLs (ONVIF)** on the add-camera form calls `GetStreamUri` (WS-UsernameToken login) on ports 80, 8080, 888 and 2020 and fills in the port that answered as the camera's `onvif_port`. See [Adding a camera](#adding-a-camera).
- **PTZ:** absolute_move, continuous_move, stop, from the PTZ pad on the ONVIF page. PTZ and event calls go to `http://<host of main_url>:<onvif_port>/onvif/device_service` (`onvif_port` defaults to 80) with the global `onvif.username` / `onvif.password` over HTTP Digest auth. Presets are saved in `config.yaml` under `camera.presets`; a save changes only that camera's `presets` list in the raw YAML, so `${VAR}` references in the file survive.
- **Events:** pull-point subscription over SOAP, started manually from the ONVIF page. Received events are logged; they are not yet turned into alerts.

### Clips

A rule with `clip: true` makes a clip when its event ends: the recorded `.ts` segments from `clips.pre_seconds` before the event started to `clips.post_seconds` after it ended (capped at `clips.max_duration` seconds) are joined without re-encoding and remuxed to `<output_dir>/<camera>/clips/<event_id>.mp4`. If the MP4 remux fails (some camera streams do not survive it), the joined `.ts` is kept and played through the HLS player instead. The job waits until `post_seconds` (plus two seconds) have passed after the end, so the last segment is complete. The camera must record (`record.enabled: true`); notifications go out when the event opens, so they never carry a clip link. The event's page plays the clip.

## Upgrading from 1.3

- **Python 3.11 or newer** is required (ONNX Runtime ships no Python 3.10 builds).
- **Database:** `serve` upgrades the schema when it starts. A SQLite database is first copied next to itself as `<file>.bak-<old revision>` (for example `warden.db.bak-0002_clips`); to go back to 1.3, stop rtsp-warden and put that copy back. A PostgreSQL database cannot be copied that way, so `serve` refuses to upgrade it and says so: take a backup (`pg_dump`), then start once with `WARDEN_DB_UPGRADE=1` in the environment. A database written by a newer release is refused with a message and left untouched. Existing events keep their data and get their camera name from the old event text; the never-used `cameras`, `recordings` and `ingest_health` tables and the `clips` table are dropped.
- **Removed pages:** the recordings list and detail pages, the timeline, and the "Generate Clip" button. Segment files stay on disk under `<output_dir>/<camera>/<stream>/`. MP4s made by the old button stay where they were written (`clips.output_dir`, by default a `clips/` directory next to the recordings directory); delete them when you no longer need them.
- **Alerts are now actions:** the Alerts page is replaced by the Actions page. A legacy `alerts.notifiers` list still loads for this release: each `enabled: true` entry becomes an action with the same name (its `severities`, `min_interval_seconds`, `min_severity` and `title_template` are dropped, with a warning), `enabled: false` entries are ignored, and `alerts.enabled` no longer does anything. Nothing is sent until a camera rule names the action. Move the entries to `actions:` and add rules; a name used in both lists is a config error.
- **Deprecated, removed in the next release** (each logs one warning when the config loads): the detector types `person`, `vehicle` and `dnn` (use `type: onnx`), a detector's `interval_seconds` (converted to `fps = 1 / interval_seconds`, capped at the camera's `detect_fps`), and the top-level `alerts:` section.
- **`record.mode: event`:** detection cannot start an event-mode recording (see Recording); use `continuous` with retention.

## Deployment

### Docker (recommended)

The default `docker-compose.yml` builds and runs the **distroless** image (~685 MB). See [docker/README.md](docker/README.md) for:
- Architecture (4-stage build with BFS `.so` dependency collector)
- The slim alternative (~1.2 GB, has a shell, useful for debugging)
- ONVIF/UDP and volume-mounting gotchas
- Bind mount vs named volume permissions

Models are stored in `./data/models` (`WARDEN_MODELS_DIR` in `docker-compose.yml`), so they are downloaded once and survive image rebuilds. For an NVIDIA GPU, add the GPU overlay below.

### GPU (NVIDIA)

Without a GPU, YOLOX runs on the CPU (see Detection for the expected cost). To use an NVIDIA GPU:

**Docker.** Install the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host (`nvidia-smi` must work and report CUDA 13 or newer), generate the toolkit's CDI spec once, then start with the GPU overlay. It builds `Dockerfile.cuda` (the slim image with `onnxruntime-gpu` and NVIDIA's CUDA 13 / cuDNN 9 Python wheels, a multi-gigabyte image) and hands the GPU to the container through CDI (Docker 25 or newer):

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose logs warden | grep "provider:"
# onnx detector onnx (model yolox-s, device auto) provider: CUDAExecutionProvider
```

The line appears once the model is loaded (on the first frame when it still had to be downloaded). `CPUExecutionProvider` there means CUDA did not load: check `nvidia-smi` on the host and `docker run --rm --device nvidia.com/gpu=all rtsp-warden:cuda nvidia-smi -L`. With `device: cuda` the camera card also shows a warning badge. If your host registers the toolkit's runtime with Docker instead (`nvidia-ctk runtime configure --runtime=docker`), the overlay's comment shows the classic `driver: nvidia` reservation to use in its place.

**uv or pip on the host.** The CPU and GPU builds of ONNX Runtime install into the same Python package, so never let both into one environment. From a source checkout:

```bash
uv sync --extra gpu --no-install-package onnxruntime --reinstall-package onnxruntime-gpu
uv run --no-sync rtsp-warden serve -c config.yaml --web --web-port 8080
# a plain `uv run` reinstalls the CPU build; back to CPU: uv sync --reinstall-package onnxruntime
```

With pip: `pip install "rtsp-warden[gpu] @ git+https://github.com/Veedubin/RSTP-Warden.git"`, then `pip uninstall -y onnxruntime onnxruntime-gpu` and `pip install "onnxruntime-gpu[cuda,cudnn]>=1.28"`.

**systemd.** The shipped unit sets `PrivateDevices=true`, which hides `/dev/nvidia*` from the service. Run `sudo systemctl edit rtsp-warden`, add `[Service]` and `PrivateDevices=false`, and restart. As for any systemd install, keep `WARDEN_MODELS_DIR=/var/lib/rtsp-warden/models` in `/etc/rtsp-warden/warden.env`.

The NVIDIA libraries are under NVIDIA's own license. They come only with the `gpu` extra or the CUDA image; rtsp-warden itself stays MIT.

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

# Install runtime + dev dependency group (uses uv; needs Python 3.11+, uv downloads one if missing)
uv sync

# Run tests
uv run pytest                  # ~1530 tests, ~120s, fully offline
uv run pytest tests/test_X.py  # single file
uv run pytest -k "pattern"     # by name
uv run --no-sync pytest -m gpu # CUDA smoke test, only after the GPU sync (see GPU)

# Lint
uv run ruff check src/ tests/  # 0 errors in new code; 4 pre-existing E501 lines are the accepted baseline
uv run ruff format src/ tests/

# Type check (informal; not in CI yet)
uv run mypy src/ || true
```

The test suite uses `asyncio_mode = "auto"` and is fully self-contained — no live cameras, no ffmpeg binary, no real network, no model weights (ONNX tests build tiny graphs). Each test gets a tmp directory and an isolated SQLite DB.

## Architecture invariants

These are the stable contracts other code depends on. Do not break them in PRs:

1. **`FrameConsumer` protocol** — `on_frame(camera, stream, jpeg_bytes, ts_unix)`. Receivers are chained; exceptions are caught and logged (they never propagate to ingest).
2. **`Detector` protocol** — `name`, `kind`, `setup()`, `process(frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]`, `teardown()`. Receives an already-masked BGR frame; ROI and grid-zone filtering happen in the runner afterwards.
3. **`Action` protocol** (`actions/base.py`) — `name`, `type`, `send(payload: ActionPayload, attachment: Path | None = None) -> ActionResult`, `test() -> ActionResult`. Synchronous (run on the action queue's worker thread); failures are returned, not raised, and error text never contains URLs, topics or tokens.
4. **Config authority:** `config.yaml` is authoritative; the DB never stores camera config. Web UI saves write the YAML through a file lock and hot-reload detectors; hand edits take effect on restart.
5. **Recording is additive** — adding new consumers, actions, detectors, or web routes must not destabilize the ingest path. A detector that cannot load its model reports the error and returns no detections; the camera keeps recording.

## License

MIT. See [LICENSE](LICENSE).

## Brand assets

The project logo lives at [`docs/assets/rtsp-warden-logo.png`](docs/assets/rtsp-warden-logo.png) (1794×1794, 3.3 MB PNG). Use this as the source for the GitHub social preview — upload a resized variant (1280×640 recommended) via **Settings → Social preview** on the repo page.

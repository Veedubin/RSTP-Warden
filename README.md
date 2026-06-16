# rtsp-warden v1.2.0


`rtsp-warden` is a self-hosted Network Video Recorder (NVR). It pulls RTSP camera streams, records them to disk in time-chunked segments, and provides a web UI for live viewing, playback, and administration.

## What's new in v1.0.0

The v1.0.0 release closes the gap to a complete self-hosted NVR:

- **Alerts** — Send detection events to ntfy.sh topics or any HTTP webhook. Per-notifier severity filter, debounce, and master switch. `/alerts` page (admin) for managing destinations.
- **Audio** opt-in per camera (`record.audio: true`). Falls back to silent if the source RTSP has no audio track.
- **Recording modes** — `continuous` (default, 24/7) or `event` (records N seconds before/after detector events). Polls the events table to start/stop ffmpeg.
- **ONVIF discovery + PTZ** — WS-Discovery finds ONVIF cameras on the LAN; PTZ control via SOAP (left/right/up/down/zoom/stop). `/onvif` page (admin).
## Timeline canvas

Recording detail page now has a visual scrubber. Click any time to load the HLS playlist at that point.

### Object-type colored markers

v1.2.0 replaces severity-based colors with object-type categorization. Markers on the timeline are now colored by the type of object detected, allowing for a quick visual scan of "who" was detected without opening the event detail.

**Color Legend:**
| Color | Category | Labels |
|-------|----------|--------|
| Red | Person | `person` |
| Blue | Pet | `cat`, `dog` |
| Green | Critter | `deer`, `raccoon`, `fox`, `squirrel`, etc. |
| Orange | Vehicle | `car`, `truck`, `bus`, `motorcycle` |
| Purple | Animal | `bird`, `horse`, `sheep`, `cow`, etc. |
| Gray | Other | `motion`, `tamper`, unknown labels |

**Severity Signal:**
To maintain visibility of alert severity, the marker line style is used:
- **Info**: Dashed line
- **Warn**: Solid line
- **Error**: Thick solid line

**Usage Notes:**
- This categorization is handled automatically by the `categorize_object()` service.
- MotionCatcher events always map to "Other" (Gray) as they lack a specific object class.

## Alembic migrations

- **Alembic migrations** — Schema is now version-controlled. Auto-stamps legacy databases, applies new migrations on `serve`.
- **Docker image** — Multi-stage Dockerfile, ~1.2GB, runs as non-root. `docker-compose.yml` with optional PostgreSQL.
- **Systemd unit** — Hardened service file, idempotent install/uninstall scripts.
- **Web UI** — 36 routes total (up from 24 in v0.7.0). New pages for `/alerts`, `/onvif`, and a JSON API at `/api/recordings/{id}/timeline` for the timeline widget.

See [SPRINT4_PLAN.md](./SPRINT4_PLAN.md) for the full design notes.

## What's new in v0.7.0

Detector framework integration with the web UI:

- **Dashboard** now shows "Detections today" stat card
- **Camera detail** page shows detector list with auto-refresh via htmx
- **Events list** page auto-refreshes every 10 seconds
- **New endpoint**: `GET /cameras/{name}/detectors` (htmx partial)
- Detector types: motion, person, vehicle, custom
- ROI and privacy mask support per detector

## What's new in v0.4.0

The full **web UI** is now shipped. A FastAPI app runs in a daemon thread inside `serve` (formerly `run`), serving:

- **Dashboard** with camera grid and live status
- **Camera detail** with live MJPEG embed, recent recordings, recent events
- **Recordings list/detail** with HLS playback via hls.js and dynamic htl: (HLS Timeline) playlists
- **Events list/detail** with filtering by camera, type, severity, date
- **Multi-user auth** — bcrypt sessions + bearer API tokens, CSRF protection, rate-limited login
- **Admin pages** — manage users, API tokens, system settings
- **Health endpoints** — `/healthz`, `/status.json`, `/metrics` (Prometheus) absorbed from the old `health_server.py`
- **htmx + Alpine.js** for auto-refresh and interactivity, no build step
- **Pico CSS v2** for styling, all assets vendored (no CDN at runtime)

CLI: `rtsp-warden run` is now deprecated in favor of `rtsp-warden serve` (alias still works). The standalone health server is gone — `/healthz`, `/status.json`, `/metrics` are served on the web UI's port (default 8080).

**780 tests pass**, 0 lint errors in new code.

## Quick start

```bash
# 1. First-run setup (creates .env, admin user, schema)
rtsp-warden install --db sqlite

# 2. Create a sample config
rtsp-warden init-config

# 3. Validate
rtsp-warden doctor -c config.yaml

# 4. Start the recorder + web UI + health endpoints
rtsp-warden serve -c config.yaml --web --web-port 8080
```

Open **http://127.0.0.1:8080/** and log in with the admin password printed during install.

## CLI commands

| Command | Description |
|---------|-------------|
| `rtsp-warden install` | First-run setup: writes `.env`, creates admin user, creates DB schema |
| `rtsp-warden init-config` | Write a starter `config.yaml` to edit |
| `rtsp-warden doctor` | Validate config + check binaries + check ports |
| `rtsp-warden serve` | **Start the recorder + web UI + health endpoints** (replaces `run`) |
| `rtsp-warden run` | **[DEPRECATED]** Alias for `serve`, with deprecation warning |
| `rtsp-warden ui` | Start the legacy stdlib MJPEG grid UI (still works, separate process) |
| `rtsp-warden status` | Print a JSON status snapshot |
| `rtsp-warden version` | Print the package version |

## Web UI routes

| Path | Auth | Description |
|------|------|-------------|
| `GET /` | user | Dashboard |
| `GET /login` | none | Login form |
| `POST /login` | none | Login (rate-limited to 5/min) |
| `POST /logout` | user | Logout |
| `GET /cameras` | user | Camera list |
| `GET /cameras/{name}` | user | Camera detail with live MJPEG |
| `GET /cameras/{name}/settings` | admin | Camera config (read-only) |
| `GET /cameras/{name}/detectors` | user | Detector list (htmx partial) |
| `GET /cameras/{name}/retention` | admin | Get current retention policy |
| `POST /cameras/{name}/retention` | admin | Update retention policy |
| `GET /cameras/{name}/zones` | admin | Grid zone editor page |
| `GET /cameras/{name}/zones/editor` | admin | Grid zone editor helper |
| `POST /cameras/{name}/zones` | admin | Save zone configuration |
| `POST /cameras/{name}/zones/{zone_name}/delete` | admin | Delete specific zone |
| `POST /cameras/{name}/zones/reload` | admin | Hot-reload zones |
| `GET /cameras/{name}/sensitivity` | admin | Get camera sensitivity |
| `POST /cameras/{name}/sensitivity` | admin | Update camera sensitivity |
| `GET /cameras/{name}/detection-classes` | admin | Get allowed detection classes |
| `POST /cameras/{name}/detection-classes` | admin | Update allowed detection classes |
| `POST /cameras/{name}/detectors/{det_type}/enabled` | admin | Toggle detector status |
| `POST /cameras/{name}/reload` | admin | Hot-reload camera configuration |
| `GET /recordings` | user | Recording list with filter |
| `GET /recordings/{id}` | user | Recording detail with HLS player |
| `GET /events` | user | Event list with filter |
| `GET /events/{id}` | user | Event detail |
| `POST /events/{event_id}/clip` | user | Generate MP4 clip for event |
| `GET /users` | admin | User list |
| `GET /users/new` | admin | New user form |
| `POST /users/new` | admin | Create user |
| `POST /users/{id}/reset-password` | admin | Reset password |
| `POST /users/{id}/delete` | admin | Delete user |
| `POST /users/{id}/toggle-admin` | admin | Toggle admin role |
| `GET /api-tokens` | user | API tokens for current user |
| `POST /api-tokens` | user | Create API token |
| `POST /api-tokens/{id}/revoke` | user | Revoke API token |
| `GET /settings` | admin | System settings (read-only) |
| `GET /htl/{cam}/{stream}/{start}/{end}.m3u8` | user | Dynamic HLS playlist for time window |
| `GET /segments/{cam}/{stream}/{path:path}` | user | Serve TS segment file |
| `GET /healthz` | none | Liveness probe |
| `GET /health` | none | Full health status (JSON or HTML) |
| `GET /status.json` | none | Full status JSON |
| `GET /metrics` | none | Prometheus metrics |
| `GET /onvif/events` | admin | List ONVIF event subscriptions |
| `POST /onvif/cameras/{name}/events/subscribe` | admin | Subscribe to ONVIF event |
| `POST /onvif/cameras/{name}/events/unsubscribe` | admin | Unsubscribe from ONVIF event |
| `POST /onvif/cameras/{name}/ptz/goto` | user | Move camera to PTZ position |
| `POST /onvif/cameras/{name}/ptz/save` | admin | Save current PTZ position as preset |
| `POST /onvif/cameras/{name}/ptz/{preset_name}/delete` | admin | Delete PTZ preset |
| `GET /clips/{clip_id}` | user | Clip detail page |
| `GET /clips/{clip_id}/download` | user | Download MP4 clip |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WARDEN_DB_URL` | `$XDG_DATA_HOME/rtsp-warden/warden.db` (sqlite) | Database URL |
| `WARDEN_SECRET` | (required) | App secret for signing |
| `WARDEN_AUTH_ENABLED` | `true` | Enable auth on protected routes |
| `WARDEN_AUTH_HEALTHZ_OPEN` | `false` | Allow anonymous `/healthz` |
| `WARDEN_WEB_ENABLED` | `true` | Enable web UI |
| `WARDEN_WEB_HOST` | `0.0.0.0` | Web UI bind host |
| `WARDEN_WEB_PORT` | `8080` | Web UI port |
| `WARDEN_HTTPS` | `false` | Set true for secure cookies (behind reverse proxy with TLS) |

## What changed in v0.4.0

- New: `src/rtsp_warden/web/` package (FastAPI app, 20+ routes, 9 templates, services, schemas)
- New: 8 new tests files (170 total tests passing)
- Changed: `rtsp-warden run` → `rtsp-warden serve` (alias preserved)
- Removed: standalone health server (`src/rtsp_warden/health_server.py` is still importable for back-compat, but no longer started by CLI)
- Added deps: `fastapi`, `uvicorn[standard]`, `jinja2`, `python-multipart`, `aiofiles`, `httpx` (dev)

## What's new in v0.3.0
- **PostgreSQL Support**: Support for PostgreSQL alongside SQLite for improved reliability and concurrency.
- **Multi-User Auth**: Session-based authentication using bcrypt and Bearer tokens.
- **Auth Gating**: Configurable access control for health and status endpoints.
- **TS Recording by Default**: Changed default container to MPEG-TS (`.ts`) to fix reliability issues with cameras providing malformed SDPs.
- **First-Class Frame Tap**: Frame-level ingest now works independently of the MJPEG proxy.
- **Expanded Configuration**: New environment variables for database, security, and admin management.
- **Dependencies**: Added hard dependency on OpenCV.

## Quick start

### 1) First-run installation
Use the new install command to set up your environment and initial admin account:
```bash
rtsp-warden install --admin-user admin --admin-pass secret123
```

### 2) Create a config
```bash
rtsp-warden init-config --out config.yaml
# edit config.yaml with your camera RTSP URLs
```

### 3) Validate your environment
```bash
rtsp-warden doctor --config config.yaml
```

### 4) Run
```bash
rtsp-warden run --config config.yaml
```

By default, `run` starts a health/status server on `127.0.0.1:8899`. Disable it with:
```bash
rtsp-warden run --config config.yaml --no-health
```

## Install command reference

The `install` command initializes the application database and security settings.

| Flag | Description |
|---|---|
| `--admin-user <name>` | Sets the initial administrator username |
| `--admin-pass <pass>` | Sets the initial administrator password |
| `--db-url <url>` | Override default database connection string |
| `--force` | Force re-initialization of the database |

## Database setup

`rtsp-warden` supports two database backends via the `WARDEN_DB_URL` environment variable:

- **SQLite (Default)**: Simple file-based storage. Ideal for small deployments.
- **PostgreSQL**: Recommended for production. 
  - Default port: `7777` (selected to avoid conflicts with standard Postgres 5432 deployments in containerized environments).

## Authentication

v0.3.0 introduces a session-based authentication system.

### Session & Tokens
- **Sessions**: User authentication uses bcrypt for password hashing and secure session cookies.
- **Bearer Tokens**: API access can be granted via Bearer tokens for headless integration.

### Auth Gating
Access to health and status endpoints is controlled via environment variables:
- `WARDEN_AUTH_ENABLED`: When `true`, all endpoints require authentication.
- `WARDEN_AUTH_HEALTHZ_OPEN`: When `true`, the `/healthz` endpoint remains public for monitoring tools (e.g., Kubernetes/Docker).

## Recording

Recording is time-segmented using FFmpeg's `segment` muxer (remux / copy).

- **Default Container**: `.ts` (MPEG-TS) is now the default. This provides NVR-grade reliability and specifically addresses "bad-SDP" camera issues where `.mp4` or `.mkv` files would fail to initialize or corrupt on stream restart.
- **Output layout**: `{output_dir}/{camera}/{stream}/...`
- **Retention**: Rules are optional and configured per camera under `record.retention`.

### Per-camera data retention

v1.2.0 introduces a three-tier resolution chain for data retention. The system checks for a policy in the following order of precedence:
1. **Camera Override**: Defined directly under the camera in `config.yaml` via the `retention` field.
2. **Recording Policy**: Defined under `record.retention` for that specific camera.
3. **Global Fallback**: Defined at the root of `config.yaml` under the global `retention` key.

This allow administrators to keep high-value footage (e.g., a front door) for 90 days while letting low-priority cameras (e.g., a hallway) expire after 7 days, all while maintaining a safe global default.

**Configuration Example:**
```yaml
# Global fallback (applies to cameras without overrides)
retention:
  max_days: 30
  max_gb: 50.0
  keep_last_n: 100

cameras:
  - name: front_door
    # Per-camera override: keep front door footage longer
    retention:
      max_days: 90
      max_gb: 100.0
  - name: hallway
    # Falls back to global policy
```

**Usage Notes:**
- Effective retention settings are visible on the camera settings page in the Web UI.
- If no policy is found in any tier, the `RetentionManager` remains a no-op and no files are deleted.

**Limitations:**
- Changes to retention policies are applied during the next scheduled cleanup interval.

### Recording Playback


The web UI provides HLS-based playback of historical recordings via the HTL (HLS Timeline) endpoint.

**Player URL format:**
```
/htl/{camera_name}/{stream_name}/{start_ts}/{end_ts}.m3u8
```

Where `start_ts` and `end_ts` are Unix timestamps defining the playback window.

**Segment URL format:**
```
/segments/{camera_name}/{stream_name}/{segment_path}
```

Both endpoints require authentication. The recording detail page (`/recordings/{id}`) automatically generates the HTL URL from the recording's timestamps and embeds an hls.js player.

## Frame tap (CV/analytics)

Frame tap is now a first-class ingest output. It delivers JPEG frames to consumers independently of whether a proxy is enabled.

- **Consumer contract**: `FrameConsumer.on_frame(camera, stream, jpeg_bytes, ts_unix)`
- **Demo Consumer**: 
  ```bash
  rtsp-warden run --config config.yaml --frame-consumer motion-demo
  ```
- **Custom Consumer**:
  ```bash
  rtsp-warden run --config config.yaml --frame-consumer your_module:YourConsumerClass
  ```

## Detectors

The detector framework runs real-time analysis on camera frames. Detectors are configured per-camera in `config.yaml` and produce events written to the database. The web UI shows detector status on camera detail pages and detection counts on the dashboard.

### Detector types

| Type | Description | Key config fields |
|------|-------------|-------------------|
| `motion` | MOG2 background subtraction motion detection | `min_area`, `sensitivity` |
| `person` | Haar cascade person detection | `min_confidence`, `scale_factor`, `min_neighbors` |
| `vehicle` | Vehicle detection | `min_confidence` |
| `custom` | User-supplied detector via `import_path` | `import_path`, `config` |
| `dnn` | Deep Neural Network (YOLO/COCO) detection | `model`, `confidence`, `nms_threshold`, `classes` |

### Advanced detection (YOLO/DNN)

Deep learning detectors replace traditional Haar cascades for significantly higher accuracy in vehicle, animal, and person detection. The `dnn` detector uses OpenCV's DNN module to run pre-trained models.

By default, `rtsp-warden` bundles the `yolov4-tiny` configuration and COCO class names. Weights are automatically downloaded to `~/.cache/rtsp-warden/modelss/` upon first use to keep the installation package lightweight. It can detect 80 standard COCO classes (e.g., car, truck, dog, cat, person).

**Configuration Example:**
```yaml
cameras:
  - name: front_door
    detectors:
      - type: dnn
        model: yolov4-tiny
        confidence: 0.5
        nms_threshold: 0.4
        classes: [car, truck, dog, cat]  # Set to null or omit for all 80 classes
```

**Usage Notes:**
- Use `dnn` detectors when accuracy is more important than CPU overhead.
- Use the `classes` list to filter for specific objects (e.g., only notify if a `dog` is detected).

**Limitations:**
- Requires a modern CPU with AVX/AVX2 support for acceptable performance.
- Initial startup is slightly delayed on the first run due to weight downloads.

### Tunable detection classes

v1.2.0 allows specifying which object classes should be detected on a per-camera basis via the `detect_classes` field. This acts as a master filter that intersects with the specific detector's configuration.

**Resolution Logic:**
- If both `camera.detect_classes` and `detector.classes` are set, only the **intersection** of the two lists is detected.
- If only `camera.detect_classes` is set, that list is used.
- If only `detector.classes` is set, the detector's list is used.
- If neither is set, all supported classes are detected.

**Configuration Example:**
```yaml
cameras:
  - name: front_door
    detect_classes: [person, dog, cat] # Only these will be detected, regardless of detector defaults
    detectors:
      - type: dnn
        enabled: true
```

**Usage Notes:**
- This filter is only applicable to `dnn` detectors. Standard `motion`, `person`, and `vehicle` detectors ignore this field as they have fixed targets.
- Matching is case-insensitive.

**Limitations:**
- An empty intersection (where the camera and detector disagree on all classes) results in zero detections being reported for that detector.

### Per-camera sensitivity

A 0-100 sensitivity knob is available per camera, providing a simplified way to tune multiple detectors simultaneously without manually editing confidence thresholds.

**Mapping Formulas:**
- **Motion**: `sensitivity = 1.0 - (camera_sensitivity / 100.0)`
- **Person/DNN Confidence**: `confidence = 1.0 - (camera_sensitivity / 100.0)`
- **DNN NMS**: `nms = 0.3 + (1 - camera_sensitivity/100) * 0.3`

**Configuration Example:**
```yaml
cameras:
  - name: front_door
    sensitivity: 75 # Higher = more sensitive (lower thresholds)
    detectors:
      - type: motion
        enabled: true
```

**Usage Notes:**
- Values can be adjusted via a slider on the camera detail page.
- **Priority**: Explicitly set fields within a `DetectorSpec` (e.g., `min_confidence: 0.8`) always override the camera-level sensitivity mapping.

**Limitations:**
- Sensitivity changes take effect upon the next detector restart or application reload.

### CLI flags


```bash
rtsp-warden serve --detectors        # enable detector processing (default)
rtsp-warden serve --no-detectors     # disable all detector processing
```

### Config example

```yaml
cameras:
  - name: driveway
    main_url: rtsp://user:pass@192.168.1.50:554/main
    sub_url: rtsp://user:pass@192.168.1.50:554/sub
    detectors:
      - type: motion
        enabled: true
        interval_seconds: 1.0
        min_area: 500
        sensitivity: 0.5
      - type: person
        enabled: true
        interval_seconds: 2.0
        min_confidence: 0.5
        scale_factor: 1.1
        min_neighbors: 3
      - type: vehicle
        enabled: false
        interval_seconds: 5.0
        min_confidence: 0.7
      - type: custom
        enabled: true
        interval_seconds: 3.0
        import_path: my_package.detectors:TrafficCounter
        config:
          lane_count: 4
```

### ROI and masks

Restrict detection to a polygon region or black out areas for privacy:

```yaml
detectors:
  - type: motion
    enabled: true
    interval_seconds: 1.0
    roi:
      - [100, 100]
      - [600, 100]
      - [600, 400]
      - [100, 400]
    masks:
      - - [0, 0]
        - [200, 0]
        - [200, 150]
        - [0, 150]
```

- **ROI**: Only detections whose center falls inside the polygon are kept.
- **Masks**: Polygon regions are blacked out before the detector sees the frame.

### Grid-based detection zones

v1.2.0 introduces "block the road" style detection zones using a customizable grid overlay. This allows for rapid definition of active zones without manually calculating polygon coordinates.

**How it works:**
The frame is divided into an NxM grid (default 8x8). You can enable or disable specific cells. A detection is kept if its bounding box center falls within any enabled cell.

**Configuration Example:**
```yaml
cameras:
  - name: driveway
    zones:
      - name: main_drive
        grid_cols: 8
        grid_rows: 8
        frame_width: 1920
        frame_height: 1080
        enabled_cells:
          - [3, 4]
          - [4, 4]
          - [5, 4]
```

**Usage Notes:**
- A visual grid editor is available at `/cameras/{name}/zones`.
- **Interaction**: Grid zones act as a secondary filter. If a primary polygon ROI is also defined, a detection must pass both the polygon ROI and at least one enabled grid cell.

**Limitations:**
- Zones are tied to the configured `frame_width` and `frame_height`. If the camera resolution changes, zones must be updated.

## Proxying


### Unified Ingest
For each `(camera, stream)`, the supervisor uses **one upstream RTSP pull** and fans out to recording, MJPEG proxying, and frame taps. This prevents multiple upstream connections per camera.

### MJPEG-over-HTTP (built-in)
Set a camera proxy to MJPEG in `config.yaml`:
```yaml
proxy:
  enabled: true
  mode: mjpeg
  stream: sub
  bind_host: 0.0.0.0
  port: 9001
```
- **Stream**: `http://HOST:9001/mjpeg`
- **Snapshot**: `http://HOST:9001/snapshot.jpg`
- **Health**: `http://HOST:9001/healthz`

### RTSP via MediaMTX (optional external binary)
```yaml
proxy:
  enabled: true
  mode: rtsp
  stream: main
  port: 9001
  path: live
```
- **Stream**: `rtsp://HOST:9001/live`

## Web UI (MJPEG grid)

A minimal dependency-free web UI module is included.

```bash
rtsp-warden ui --config config.yaml --port 8080 --embed-host 127.0.0.1
```
Open: `http://127.0.0.1:8080/`. The UI only lists cameras with `proxy.mode: mjpeg` enabled.

## Alerts

The alerts framework sends detection events to external services. Two notifier types are supported: **ntfy** (push notifications via [ntfy.sh](https://ntfy.sh) or self-hosted) and **webhook** (any HTTP endpoint that accepts JSON).

```yaml
# config.yaml
alerts:
  enabled: true
  notifiers:
    - name: Family phones
      type: ntfy
      url: https://ntfy.sh
      topic: my-warden-alerts
      token: tk_xxxxxxxxxxxx  # optional, for private topics
      priority: 4
      severities: [warn, error]
      min_interval_seconds: 30
    - name: Slack #ops
      type: webhook
      url: https://hooks.slack.com/services/T00/B00/XXXXX
      headers:
        Content-Type: application/json
      severities: [error]
      min_interval_seconds: 60
```

Manage notifiers at `/alerts` (admin). The web UI lets you test notifiers, view status, and (with `alerts.yaml` companion file) add/remove notifiers without restarting.

### Email and other channels (apprise)

The `apprise` notifier allows `rtsp-warden` to send alerts to over 90 different services, including Email (SMTP), Discord, Telegram, Slack, and Pushover, using a unified URL-based configuration.

Apprise notifications are wrapped in asynchronous threads to ensure that network latency during alert delivery does not block the main event loop or interfere with video ingest.

**Configuration Example:**
```yaml
alerts:
  notifiers:
    - name: Main Email
      type: apprise
      urls: ["mailto://user:pass@smtp.gmail.com:587"]
      min_interval_seconds: 60
    - name: Team Chat
      type: apprise
      urls: ["tgram://bottoken/ChatID"]
      min_interval_seconds: 30
```

**Usage Notes:**
- Use `apprise` for any notification channel not natively supported by the ntfy or webhook notifiers.
- You can provide multiple URLs per notifier to send a single event to multiple services simultaneously.

**Limitations:**
- Complex authentication (like OAuth2) for certain services may require the use of Apprise-specific tokens or API keys within the URL.

## Recording modes

By default, recording is **continuous** (24/7 chunked segments). Switch a camera to **event** mode to record only around detector events:

```yaml
cameras:
  - name: front_door
    record:
      mode: event
      event_record:
        pre_seconds: 5       # buffer before first event (aspirational)
        post_seconds: 10     # keep recording 10s after last event
        min_segment_seconds: 10
        max_segment_seconds: 600
```

When in event mode, the recorder polls the database every second and starts ffmpeg when an event is detected. The pre_seconds buffer is aspirational (segments always start at "now"); use post_seconds to control how long to keep recording after the last event.

### Generating clips from detections

`rtsp-warden` can now generate standalone MP4 clips that span a specific detection event. Unlike full recordings, clips are extracted using FFmpeg's `copy` codec for high-speed generation without re-encoding.

Clips are stored in a dedicated directory and are accessible via a new web UI detail page.

**Configuration Example:**
```yaml
clips:
  enabled: true
  pre_seconds: 10.0
  post_seconds: 10.0
  output_dir: "/var/lib/rtsp-warden/clips"
  max_duration: 120.0
```

**Usage Notes:**
- Clips are ideal for sharing specific events via email or chat without sending a full recording archive.
- The `pre_seconds` and `post_seconds` define the window around the event trigger.

**Limitations:**
- Clip extraction requires that the corresponding recording segments are still available on disk.
- High-frequency events may result in overlapping clips.


## Audio

Enable audio in recordings by setting `record.audio: true`. The recorder adds `-c:a aac -b:a 128k` to the ffmpeg command. If the camera's RTSP stream has no audio, the recording is still valid (silent).

```yaml
cameras:
  - name: front_door
    record:
      audio: true
```

## ONVIF

Enable ONVIF discovery and PTZ control globally:

```yaml
onvif:
  discovery_enabled: true
  ptz_enabled: true
  discovery_timeout_seconds: 5
  username: admin       # optional default credentials
  password: CHANGEME
```

Then visit `/onvif` (admin) to discover cameras on the local network. The discovery probe is a WS-Discovery UDP multicast to `239.255.255.250:3702`. PTZ controls are available on the same page (left/right/up/down/zoom/stop).

For Docker deployments, ONVIF discovery requires `network_mode: host` (UDP multicast doesn't work across bridge networks).

### ONVIF events subscription

`rtsp-warden` can now subscribe to native ONVIF events via WS-BaseNotification. Instead of relying solely on visual detectors, the system can trigger alerts based on internal camera events such as Motion Alarms, Imaging Alarms (tampering), or Rule Engine analytics.

These events are integrated directly into the `AlertManager` and can trigger any configured notifier.

**Configuration Example:**
```yaml
cameras:
  - name: front_door
    onvif:
      host: 192.168.1.100
      port: 80
      username: admin
      password: changeme
    events:
      - type: motion
        min_interval_seconds: 30
      - type: tamper
        min_interval_seconds: 60
```

**Usage Notes:**
- Use this for "instant" triggers that cameras provide internally, which are often faster and more reliable than software-based motion detection.

**Limitations:**
- Support varies by camera vendor; not all ONVIF-compliant cameras implement the `VideoSource/MotionAlarm` or `RuleEngine` topics.

### PTZ presets

Named PTZ presets allow you to save and recall specific pan, tilt, and zoom positions for each camera. Presets are persisted in the `config.yaml` file, allowing for consistent positioning across restarts.

**Configuration Example:**
```yaml
cameras:
  - name: front_door
    presets:
      - name: front_gate
        pan: 0.0
        tilt: 0.0
        zoom: 0.5
      - name: driveway
        pan: -0.5
        tilt: 0.1
        zoom: 0.3
```

**Usage Notes:**
- Use presets to quickly switch between high-traffic areas of a property.
- Presets can be managed via the web UI or API.

**Limitations:**
- PTZ precision depends on the camera hardware's supported coordinate system.

## Docker

A multi-stage Dockerfile is provided:

```bash
docker build -t rtsp-warden .
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/recordings:/app/recordings \
  -v $(pwd)/data:/app/data \
  rtsp-warden
```

Or with `docker-compose` (includes optional PostgreSQL):

```bash
docker compose up -d
```

See [docker/README.md](./docker/README.md) for the full guide.

## Systemd (Linux)

For traditional Linux deployments:

```bash
sudo pip install rtsp-warden
sudo packaging/systemd/install.sh
sudo cp your-config.yaml /etc/rtsp-warden/config.yaml
sudo systemctl enable --now rtsp-warden
```

See [packaging/systemd/README.md](./packaging/systemd/README.md).

## CLI command reference

| Command | Description |
|---|---|
| `install` | Initializes DB, admin user, and security settings |
| `init-config` | Generates a default `config.yaml` |
| `doctor` | Validates config and environment dependencies |
| `serve` | Starts the warden supervisor + ingest + web UI (formerly `run`) |
| `run` | Hidden deprecated alias for `serve` |
| `ui` | Starts the MJPEG grid web interface (legacy stdlib UI) |
| `status` | Prints a JSON snapshot of current stream health |
| `version` | Prints the current version of rtsp-warden |

### `serve` flags

| Flag | Default | Description |
|---|---|---|
| `-c / --config` | `config.yaml` | Path to config file |
| `--web` | on | Enable web UI (FastAPI) |
| `--no-web` | | Disable web UI (CLI-only mode) |
| `--web-host` | `WARDEN_WEB_HOST` (default `127.0.0.1`) | Web UI bind host |
| `--web-port` | `WARDEN_WEB_PORT` (default `8080`) | Web UI bind port |
| `--detectors` | on | Enable detector framework (motion/person/vehicle/custom) |
| `--no-detectors` | | Disable detector framework |
| `--alerts` | on | Enable alert notifiers |
| `--no-alerts` | | Disable alert notifiers |

## Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `WARDEN_DB_URL` | `sqlite:///warden.db` | Connection string for the database |
| `WARDEN_SECRET` | (None) | Secret key for session signing |
| `WARDEN_ADMIN_USER` | `admin` | Default admin username |
| `WARDEN_ADMIN_PASS` | (None) | Default admin password |
| `WARDEN_AUTH_ENABLED` | `false` | Enable authentication for all endpoints |
| `WARDEN_AUTH_HEALTHZ_OPEN` | `true` | Allow public access to `/healthz` |

## Security note

If your RTSP URLs contain credentials, logs and status output attempt to redact them. Avoid binding services to public interfaces unless you understand the risk.

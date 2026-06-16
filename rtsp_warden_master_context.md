# RTSP Warden — Master Context (Plan + Current Implementation)

**Project name:** `rtsp-warden`  
**Latest checkpoint artifact:** `rtsp-warden_v0.1.2.zip`  
**Primary goals (recap):**
- Connect to home security cameras over RTSP (main + substream).
- Record to disk in configurable chunks (per camera, per stream).
- Provide a local “proxy” so downstream viewers (VLC / future web UI) connect to your service instead of directly to the camera.
- Scale to multiple cameras with a simple operational model: **one camera per config entry** and **one proxy port per camera**.
- Keep the code modular and UV-friendly, with clean seams for future CV/recognition (OpenCV/Pillow).

---

## 0) What exists now (v0.1.2)

### Implemented capabilities
1. **Multi-camera orchestration**
   - A single YAML config can define **N cameras**.
   - Each camera can enable/disable recording and proxy independently.

2. **Recording: chunked segments, per stream**
   - Each camera has:
     - **main stream** recording config
     - **sub stream** recording config
   - Each stream records to disk as time-based segments using FFmpeg, defaulting to:
     - main: **MKV** container
     - sub: **MKV** container (but you can set sub to **MP4**)
   - Recording uses **stream copy** (`-c copy`) which preserves the camera’s native bitstream (no re-encode).

3. **Proxy: one camera per port, two modes**
   - `proxy.mode: mjpeg` (built-in)
     - Starts a small HTTP server (aiohttp) and serves:
       - `/mjpeg` (multipart MJPEG stream)
       - `/snapshot.jpg`
       - `/healthz`
       - `/` (simple HTML viewer)
   - `proxy.mode: rtsp` (external MediaMTX binary)
     - Spawns a dedicated MediaMTX instance per camera/port.
     - Publishes a path like `rtsp://HOST:PORT/live` (configurable path).

4. **Supervisor hardening**
   - Auto-restart on child-process failure (FFmpeg / MediaMTX) using exponential backoff parameters from config.
   - Captures and retains a small tail of stderr for debugging.
   - Periodic status output (Rich table).

5. **Retention cleanup (optional)**
   - Optional per-camera retention manager that can enforce:
     - `max_days`
     - `max_gb`
     - `keep_last_n`
   - Runs periodically per camera.

---

## 1) Artifact inventory (what you can hand to another window)

### Primary zip
- `rtsp-warden_v0.1.2.zip` — contains the latest code and example config.

### Earlier zips (if you need diffs)
- `rtsp-warden_v0.1.0.zip`
- `rtsp-warden_v0.1.1.zip`

In practice: treat **v0.1.2** as the canonical base.

---

## 2) Package layout (v0.1.2)

From the zip:

```
pyproject.toml
README.md
LICENSE
examples/config.yaml
src/rtsp_warden/
  __init__.py
  cli.py
  config.py
  logging_utils.py
  ffmpeg.py
  recorder.py
  retention.py
  app.py
  proxy/
    __init__.py
    mjpeg.py
    rtsp_mediamtx.py
```

### Responsibility map (file-by-file)
- `src/rtsp_warden/config.py`
  - Pydantic models and YAML loader.
  - Defines per-camera schema for:
    - stream recording config (main/sub)
    - proxy config
    - runtime config
- `src/rtsp_warden/ffmpeg.py`
  - Managed subprocess wrapper (start/poll/terminate)
  - Stderr tail capture (bounded)
  - Exponential backoff helper
  - FFmpeg command builders:
    - segment recording
    - MJPEG stdout pipe
- `src/rtsp_warden/recorder.py`
  - Builds and starts FFmpeg segmenters for main/sub (based on config)
- `src/rtsp_warden/proxy/mjpeg.py`
  - MJPEG proxy server: FFmpeg -> JPEG frames -> HTTP endpoints.
- `src/rtsp_warden/proxy/rtsp_mediamtx.py`
  - MediaMTX integration: writes a temporary config in `workspace_dir` and spawns a MediaMTX process.
- `src/rtsp_warden/retention.py`
  - Periodic retention sweeper for a camera’s recordings folder.
- `src/rtsp_warden/app.py`
  - Supervisor that wires everything together per camera and runs the main loop.
- `src/rtsp_warden/cli.py`
  - Typer CLI entrypoint: `run`, `doctor`, `init-config`
- `src/rtsp_warden/logging_utils.py`
  - Rich logging setup

---

## 3) Key design decisions (so far)

### FFmpeg as the ingest/recording engine
- RTSP is annoying in practice (timestamps, reconnects, codec quirks).
- FFmpeg is used as the source-of-truth transport/mux layer.
- Python orchestrates processes, config, and fan-out endpoints.

### Proxy modes
- MJPEG-over-HTTP:
  - Extremely simple to implement.
  - Easy to embed in a future “basic web UI” without WebRTC/HLS.
  - Works well for “preview stream” / substream use-cases.
- RTSP via MediaMTX:
  - Proper RTSP semantics for clients that prefer RTSP.
  - Reuses a known-good RTSP server rather than reimplementing one in Python.

---

## 4) Configuration model (what you edit)

### High-level shape
- `cameras: []` list
- `runtime: {}` global settings

### Camera fields (summary)
- `name`: unique camera name (used in folder naming and logs)
- `main_url`: RTSP URL for main stream
- `sub_url`: RTSP URL for sub stream
- `record`: recording block
- `proxy`: proxy block

### Recording block (current schema)
- `enabled`: bool
- `output_dir`: root dir for this camera’s recording output
- `main`: stream recording config (below)
- `sub`: stream recording config (below)
- `retention`: retention config (below)

### Stream recording config (main/sub)
- `enabled`: bool
- `container`: `mkv` or `mp4`
- `chunk_seconds`: segment duration
- `rtsp_transport`: `tcp` or `udp` (tcp recommended for Wi‑Fi cameras)
- `mode`: currently only `copy` (remux, no re-encode)

### Retention config
- `max_days`: optional int
- `max_gb`: optional float
- `keep_last_n`: int, keep newest N files regardless of other constraints
- `cleanup_interval_seconds`: how often to run retention sweeps

### Proxy block
- `enabled`: bool
- `mode`: `mjpeg` or `rtsp`
- `stream`: `main` or `sub` (which RTSP URL to proxy)
- `bind_host`: IP to bind (consider `127.0.0.1` by default if you don’t need LAN access)
- `port`: unique per camera
- `path`: RTSP path (mode=rtsp only)
- `source_on_demand`: RTSP mode only (start pulling upstream only when clients connect)
- MJPEG options:
  - `fps`: integer
  - `scale_width`: 0 disables, otherwise scales width keeping aspect ratio

### Runtime config
- `ffmpeg_path`: path to ffmpeg binary
- `mediamtx_path`: path to mediamtx binary
- `ffmpeg_loglevel`: warning/info/debug
- `workspace_dir`: where transient files go (ex: MediaMTX temp configs)
- `auto_restart`: bool
- `restart_backoff_*`: restart parameters
- `stderr_tail_lines`: lines of stderr to retain per process
- `status_interval_s`: supervisor status print interval

---

## 5) Example config (from `examples/config.yaml`)

```yaml
cameras:
  - name: front
    main_url: rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101
    sub_url: rtsp://user:pass@192.168.1.50:554/Streaming/Channels/102

    record:
      enabled: true
      output_dir: ./recordings

      # Main stream: prefer MKV container (remux / copy; can be useful when you care about maximum fidelity)
      main:
        enabled: true
        container: mkv
        chunk_seconds: 300
        rtsp_transport: tcp

      # Sub stream: MP4 container is often convenient for general playback tooling
      sub:
        enabled: true
        container: mp4
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

      # RTSP (MediaMTX) options
      path: live
      source_on_demand: true

      # MJPEG options
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
```

---

## 6) CLI: how to run it

### Install (UV)
Typical dev workflow:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Generate starter config
```bash
rtsp-warden init-config --out config.yaml
```

### Sanity check dependencies + ports
```bash
rtsp-warden doctor --config config.yaml
```

### Run
```bash
rtsp-warden run --config config.yaml
```

---

## 7) How to view streams

### MJPEG mode (built-in)
If `proxy.mode: mjpeg`, open:

- Browser:
  - `http://HOST:PORT/` (simple page)
  - `http://HOST:PORT/mjpeg`
  - `http://HOST:PORT/snapshot.jpg`
- VLC:
  - “Open Network Stream” -> `http://HOST:PORT/mjpeg`

### RTSP mode (MediaMTX)
If `proxy.mode: rtsp`, open:

- `rtsp://HOST:PORT/<path>`

Example:
- `rtsp://127.0.0.1:9001/live`

---

## 8) Operational notes (current behavior)

### Processes per camera (important)
Right now, the system can start multiple upstream RTSP connections per camera:
- Recording main -> one FFmpeg connection to `main_url` (if enabled)
- Recording sub -> one FFmpeg connection to `sub_url` (if enabled)
- Proxy mjpeg -> one FFmpeg connection to whichever stream is selected
- Proxy rtsp -> MediaMTX will pull from the selected stream (on-demand or always)

This means **the “single upstream connection” goal is not fully achieved yet**.
You *can* reduce camera load by:
- proxying the substream and disabling sub recording (or vice versa), or
- picking only one stream to proxy/record.

**The next milestone is to unify ingest so a single FFmpeg process per stream can feed both recording and proxy outputs.**

### Segment containers
- MKV is more resilient if the process is interrupted mid-segment.
- MP4 is widely compatible but can be less resilient if not finalized properly.

### Retention safety
Retention is conservative if you set `keep_last_n` (recommended).
The retention manager sweeps periodically and deletes older files according to rules.

---

## 9) Immediate backlog (do not drop the ball)

This is the prioritized “next work” list, aligning to your original plan.

### A) Unify ingest (single upstream per stream)
Target: **one FFmpeg process per camera stream** that can output:
- Output 1: segmented file recording (stream copy)
- Output 2: proxy stream source
  - For MJPEG: filtered decode + mjpeg output (re-encode) is fine
  - For RTSP: push a stream to MediaMTX as a publisher so clients connect to MediaMTX
    - This makes the camera see only **one upstream connection**

Implementation direction:
- Use FFmpeg multi-output:
  - main output: `-c copy -f segment ...`
  - secondary output: either:
    - `-f image2pipe pipe:1` (MJPEG) OR
    - `-f rtsp rtsp://127.0.0.1:<port>/<path>` (publish to MediaMTX)

### B) Add “frame tap” interface (CV-ready)
- A clean seam for OpenCV/Pillow without affecting recording/proxy stability:
  - `FrameConsumer.on_frame(camera, stream, jpeg_bytes, ts)` (or decoded ndarray later)
- Start with a no-op hook + one demo hook (e.g., motion heuristic)

### C) Web UI (basic)
- An index page listing cameras and embedding MJPEG endpoints
- Later: optionally HLS/WebRTC, but not required now

### D) Better health + metrics
- Aggregate health endpoint (single port) listing all cameras:
  - last frame time
  - ffmpeg process state
  - segment write heartbeat
  - client count (mjpeg)

### E) Packaging polish
- Add minimal tests:
  - config validation
  - command building
  - retention logic on temp dirs

---

## 10) Known constraints to keep in mind
- Python-only RTSP server is not a short task; MediaMTX is a pragmatic dependency for RTSP proxying.
- MJPEG is ideal for previews but not the most bandwidth-efficient for full-res continuous viewing.
- Achieving “single upstream connection” cleanly likely means:
  - one ingest process per stream, with multi-output,
  - and pushing RTSP to a local RTSP server (MediaMTX) rather than having the RTSP server pull from the camera separately.

---

## Appendix A — README (as packaged in v0.1.2)

```markdown
# rtsp-warden

A small, modular Python (UV-friendly) package to:

1. Pull RTSP camera streams and **record to disk in time-chunked segments**
2. Optionally **proxy** a camera stream locally so multiple clients can view it without each opening a direct RTSP session

Current proxy backends:

- **MJPEG-over-HTTP** (built-in): VLC can open `http://HOST:PORT/mjpeg`
- **RTSP via MediaMTX** (optional external binary): VLC can open `rtsp://HOST:PORT/PATH`

## Quick start

```bash
uv venv
uv sync
rtsp-warden init-config --out config.yaml
# edit config.yaml with your camera URLs
rtsp-warden doctor --config config.yaml
rtsp-warden run --config config.yaml
```

## Recording

By default, each camera records **two streams** independently (main + sub) and writes segments to:

```
{output_dir}/{camera}/{stream}/{camera}_{stream}_YYYYmmdd_HHMMSS.{mkv|mp4}
```

Segmenting is done with `ffmpeg`'s `segment` muxer (remux / `-c copy`).

## Proxying

### MJPEG-over-HTTP (built-in)

Set:

```yaml
proxy:
  enabled: true
  mode: mjpeg
  port: 9001
```

Then open in VLC:

- `http://HOST:9001/mjpeg`
- Snapshot: `http://HOST:9001/snapshot.jpg`
- Health: `http://HOST:9001/healthz`

### RTSP via MediaMTX (optional)

Install `mediamtx` and set:

```yaml
proxy:
  enabled: true
  mode: rtsp
  port: 9001
  path: live
```

Then open in VLC:

- `rtsp://HOST:9001/live`

rtsp-warden will generate a per-camera `mediamtx.yml` under:

```
{workspace_dir}/proxy/{camera}/mediamtx.yml
```

## Notes / next steps

- Add OpenCV / Pillow hooks for per-frame detection (motion, person detection, etc.)
- Add a minimal web UI for multi-camera live view (MJPEG grid first; then HLS/WebRTC as needed)
- Improve restart behavior (graceful proxy restarts without disconnecting HTTP clients)
```

---

## Appendix B — pyproject.toml (as packaged in v0.1.2)

```toml
[build-system]
requires = ["hatchling>=1.25.0"]
build-backend = "hatchling.build"

[project]
name = "rtsp-warden"
version = "0.1.2"
description = "Modular RTSP ingest (record) + local proxy for home security cameras (UV-friendly package)."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Jakob Charles" }]
dependencies = [
  "typer>=0.12.0",
  "pydantic>=2.7.0",
  "PyYAML>=6.0.1",
  "rich>=13.7.1",
]

[project.scripts]
rtsp-warden = "rtsp_warden.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/rtsp_warden"]

[tool.uv]
package = true

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

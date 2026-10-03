# Detection and automation: design spec

Date: 2026-10-02
Status: approved 2026-10-02
Scope: rtsp-warden sub-project 3 of 3 (see "Order of work")

## 1. Purpose

rtsp-warden is a self-hosted NVR for home-lab users: one process, a YAML config,
a web UI, no fluff. Today it records and previews cameras but its detection
pipeline is not usable: the built-in person and vehicle detectors are weak
classical models, the YOLOv4-tiny detector is fed 320-pixel frames, every
detection frame becomes an event row, and nothing ever reaches a notifier.

This spec makes detection work end to end: a modern object detector on GPU or
CPU, one event per object visit with a thumbnail and optional clip, and
per-camera rules that run named actions (notifications and webhooks). The
design keeps the model and its label list swappable so a custom model (for
example one trained on raccoons) drops in later without code changes.

Success looks like: add a camera, see live video, walk through the frame, get
one `person` event with a picture and one phone notification.

## 2. Goals and non-goals

Goals

- Object detection via ONNX Runtime with CUDA when available, CPU otherwise.
- Permissively licensed default model (YOLOX, Apache-2.0); project stays MIT.
- Swappable models: ONNX file + labels + descriptor, no code change.
- One event per object visit, with best-frame thumbnail, label, confidence,
  zone, start and end time, optional clip.
- Per-camera and per-detector sampling rates.
- Per-camera rules that fire named actions: ntfy, Apprise, webhook.
- Events, thumbnails, clips and action results visible in the web UI.
- Live preview served by warden itself, with optional detection boxes.
- Runs in Docker on CPU, and on GPU with an additional image.

Non-goals (explicitly out of scope for this spec)

- Raspberry Pi targets and GPIO or other hardware actions.
- Feeding ONVIF event subscriptions into rules.
- Object re-identification across cameras, or across long gaps.
- Model training or fine-tuning tooling.
- Creating or editing actions from the UI (YAML only in this version).
- Audio detection.

## 3. Order of work

The work is three sub-projects. This spec covers the third in depth. The first
two are bounded fixes to existing flows and get short in-chat designs when
started; they are listed in Appendix A so the implementation plan can sequence
them, because detection depends on them (config write-back, CSRF, status,
same-origin preview).

1. Stabilize: every defect from the 2026-10-02 audit.
2. UI pass: usable and clean, plus an add-camera flow with a connection test.
3. Detection and automation: this document.

## 4. Architecture

```
ffmpeg ingest (main stream)
  └─ frame tap (JPEG, detect_fps, width = largest model input)
       └─ FrameTapDispatcher
            └─ DetectorRunner (per camera; bounded queue, drop-oldest)
                 ├─ MotionDetector            ──► motion events (unchanged path)
                 └─ OnnxDetector (per spec)
                      └─ Tracker (per camera, per label)
                           └─ EventBuilder ──► events table + thumbnail file
                                └─ RuleEngine ──► ActionQueue ──► ntfy / Apprise / webhook
                                                  └─ ClipJob (optional) ──► clip file
```

Everything runs in the existing single process. The detector interface stays
`Detector.process(frame_bgr, ts_unix) -> list[Detection]`, so a separate
inference service could be added later behind the same interface, but is not
built now.

## 5. Model and inference

### 5.1 Runtime

- New detector type `onnx` in `detectors/builtin/onnx.py` using `onnxruntime`.
- `onnxruntime` (CPU) is a core dependency. `onnxruntime-gpu` is an optional
  extra: `uv sync --extra gpu`.
- Provider selection per detector `device` setting: `auto` (default: CUDA if
  the provider is available, else CPU), `cuda`, `cpu`. When `cuda` is requested
  and unavailable, the detector falls back to CPU, logs a warning, and the
  camera's status panel shows a warning badge.
- The detector logs the chosen provider at setup.

### 5.2 Model registry

A model is a directory containing a descriptor and its files:

```yaml
# <models_dir>/yolox-s/model.yaml
name: yolox-s
file: yolox_s.onnx
labels: coco.txt            # one label per line, index = class id
input_size: [640, 640]
postprocess: yolox          # yolox | yolo-anchor-free | ssd
sha256: <hex>               # of the .onnx file
url: https://...            # optional: downloaded on first use when file is absent
```

- Built-in descriptors ship in the package for `yolox-s` (default) and
  `yolox-nano` (CPU-friendly). Their ONNX files download on first use into
  `models_dir` with SHA-256 verification, reusing the download helper in
  `detectors/builtin/model_utils.py`.
- `models_dir` defaults to `$XDG_CACHE_HOME/rtsp-warden/models` and is
  configurable under `runtime.models_dir` for Docker volume mounts.
- A user adds a model by creating a directory with a descriptor. Unknown
  `postprocess` values fail config validation with the list of supported ones.
- Class names always come from the model's labels file. `detect_classes` on a
  camera filters against those names; its default is all labels. Unknown names
  in `detect_classes` are a config validation error that lists the model's labels.

### 5.3 Frame tap

- Tap rate is the camera's `detect_fps` (section 6).
- Tap width is the largest `input_size[0]` among the camera's enabled detectors,
  minimum 320 (motion only). Letterboxing to the model's square input happens
  in the detector, not in ffmpeg, so the tap stays a plain scale.
- The tap always comes from the stream that feeds the proxy. With `sub_url`
  absent (allowed after sub-project 1) that is the main stream.

### 5.4 Legacy detectors

- `motion` stays as the cheap always-on detector.
- `person` (HOG), `vehicle` (Haar) and `dnn` (yolov4-tiny) remain for one
  release. Using them logs a deprecation warning naming the `onnx` replacement.
  They are removed in the release after.

## 6. Sampling rates

- Camera-level `detect_fps` (float, default 5, range 0.5 to 30) sets the tap
  rate. Changing it restarts that camera's ingest, because the rate is an
  ffmpeg argument; the UI field says so.
- Detector-level `fps` (float, optional) must be at or below `detect_fps`;
  config validation rejects a larger value. The runner skips frames per
  detector to honor it. Default is the camera's `detect_fps`.
- `interval_seconds` on a detector is replaced by `fps`. It is still accepted
  for one release: it is converted (`fps = 1 / interval_seconds`) with a
  one-time deprecation warning naming the new field. It is removed after.
- Per-detector `fps` changes hot-reload through `rebuild_camera_detectors`.

## 7. Tracking and events

### 7.1 Tracker

`detectors/tracking.py`, one `Tracker` per camera, tracks kept per label.

- Match detections to open tracks by intersection-over-union, greedy, highest
  IoU first, threshold 0.3.
- A track that is not matched on a frame is kept open until it has missed for
  `track_grace_seconds` (default 3.0), then closed.
- Each track keeps: id, label, first and last seen timestamps, frame count,
  current box, best confidence, best frame (BGR copy) and best box.
- Tracker state is reset when the camera's detectors are rebuilt.

### 7.2 Event lifecycle

- A track becomes an event when it reaches `min_track_frames` consecutive
  matched frames (default 2). At that moment an `events` row is inserted with
  `label`, `confidence` (best so far), `zone`, `track_id`, `camera_name`,
  `created_at`, and `thumbnail_path`; the thumbnail is written then.
- While the track is open, the row's `confidence`, `zone` and thumbnail are
  updated whenever a better frame arrives (at most once per second).
- When the track closes, `ended_at` is set and the rule engine is notified of
  the closed event; clip jobs are queued at this point.
- Rules fire on event open (so notifications are prompt), not on close.
- Thumbnails are JPEG with the best box drawn, written to
  `<record.output_dir>/<camera>/thumbnails/<event_id>.jpg`.

### 7.3 Zones

- Zones already filter detections (`GridMask`). The event's `zone` is the name
  of the zone containing the center of the best box, or empty when none.
- Rules can require a zone (section 8).

### 7.4 Motion events

- Motion detections keep their current path (one event per motion burst as the
  motion detector already debounces). They get no tracker, no thumbnail.
- When a camera has any enabled `onnx` detector, the motion detector's events are
  not written unless the motion spec sets `events: true`. Motion can still act
  as a gate in a later version; not in this one.

### 7.5 Camera identity

- Events store `camera_name` directly. The `cameras` table is dropped
  (Appendix B). All event queries filter on `camera_name`.

## 8. Rules and actions

### 8.1 Actions

Top-level `actions` list of named entries. Types: `ntfy`, `apprise`, `webhook`,
implemented by the existing notifier classes moved under `actions/`.

```yaml
actions:
  - name: phone
    type: ntfy
    url: https://ntfy.sh
    topic: warden-home
    token: ${NTFY_TOKEN}        # env substitution, from sub-project 1
  - name: homeassistant
    type: webhook
    url: http://ha.local:8123/api/webhook/warden
```

- `alerts.notifiers` keeps loading for one release, mapped onto `actions`, with
  a deprecation warning. The `alerts.enabled` flag is gone; an action exists or
  it does not.
- Apprise's URL scheme covers MQTT, Telegram, Discord and others, so no further
  action types are added.

### 8.2 Rules

Per-camera `rules` list.

```yaml
cameras:
  - name: yard
    main_url: rtsp://...
    detect_fps: 5
    detectors:
      - type: motion
        fps: 5
      - type: onnx
        model: yolox-s
        device: auto
        fps: 2
        min_confidence: 0.5
    rules:
      - name: person-any-time
        labels: [person]
        zones: []                 # empty = any zone
        min_confidence: 0.6
        between: null             # or "22:00-06:00", local time, may wrap midnight
        cooldown_seconds: 60
        clip: true
        actions: [phone]
```

- A rule matches an event when: label in `labels` (empty = any), zone in
  `zones` (empty = any), confidence >= `min_confidence`, and now is inside
  `between` when set.
- Cooldown is keyed by (camera, rule name, label). An event within the cooldown
  window is recorded but fires nothing; the event row notes `suppressed_by`
  the rule name in its metadata.
- Unknown action names in `rules[].actions` are a config validation error.

### 8.3 Dispatch

- `EventBuilder` calls `RuleEngine.evaluate(event)` synchronously on event open.
  Evaluation is pure and cheap.
- Matching actions are enqueued onto `ActionQueue`: a bounded `queue.Queue`
  (size 256) drained by one daemon worker thread. On overflow the oldest job is
  dropped and counted.
- Every action run writes an `action_runs` row: event id, action name, status
  (`ok` or `failed`), error text, timestamp. No retries in this version.
- The payload given to every action is the same dict: `camera`, `label`,
  `confidence`, `zone`, `started_at`, `ended_at` (null while open),
  `thumbnail_url`, `clip_url` (null until a clip exists), `event_url`. URLs are
  absolute, built from `runtime.public_url` when set, else from the web
  server's bind host and port.
- ntfy and Apprise attach the thumbnail file. Webhook posts the dict as JSON.

### 8.4 Clips

- When a matched rule has `clip: true`, closing the event queues a clip job on
  the same worker.
- Clips come from the recorded `.ts` segments: `clips.pre_seconds` before
  `created_at` to `clips.post_seconds` after `ended_at`, concatenated with
  `-c copy` into `<record.output_dir>/<camera>/clips/<event_id>.mp4` with
  `-movflags +faststart`. If ffmpeg exits non-zero, the concatenated `.ts` is
  kept instead and served through the existing HLS player. (The MP4 ban in
  `CONTEXT_RTSP_WARDEN_DROP_MP4_USE_TS.md` is about live segmenting from the
  camera; a remux of already-written segments is a different operation, and
  the `.ts` fallback covers cameras where even that fails.)
- The segment filename regex in `clips.py` is fixed to match
  `{camera}_{stream}_YYYYMMDD_HHMMSS.ts`.
- The event row gets `clip_path`; actions fired on open do not include a clip
  URL. A second notification on close is not sent in this version.

### 8.5 Testing rules from the UI

- Actions page: "Test" per action sends a synthetic payload through the real
  action class.
- Camera rules panel: "Fire test event" creates a synthetic `person` event with
  a placeholder thumbnail and pushes it through `RuleEngine` and `ActionQueue`,
  so the whole path is exercised.

### 8.6 Removed

- `AlertManager` (`start`, `dispatch_event`, `status`) and the alerts routes'
  dependence on it. Notifier classes are kept and moved.

## 9. Storage and UI

### 9.1 Database

One Alembic migration (`0003_detection_events`):

- `events`: add `camera_name` (indexed), `label`, `confidence`, `zone`,
  `track_id`, `ended_at`, `thumbnail_path`, `clip_path`. Backfill
  `camera_name` from `metadata_json.camera` where present. Drop `camera_id`.
- New table `action_runs`: `id`, `event_id` (FK, indexed), `action_name`,
  `status`, `error`, `created_at`.
- Drop tables `cameras`, `recordings`, `ingest_health` (never written).
- Update `EXPECTED_TABLES` in `tests/test_alembic.py`.

### 9.2 Files

- Thumbnails: `<record.output_dir>/<camera>/thumbnails/<event_id>.jpg`.
- Clips: `<record.output_dir>/<camera>/clips/<event_id>.(mp4|ts)`.
- Both are swept by the camera's retention manager using the same age and size
  rules as segments. When a file is removed, the event row keeps its data and
  the UI shows "expired" in place of the image.
- Models: `runtime.models_dir`.

### 9.3 Routes

- `GET /events/{id}/thumbnail.jpg`, `GET /events/{id}/clip` (serves the file,
  or the HLS playlist for a `.ts` clip).
- `GET /cameras/{name}/live.mjpeg?boxes=0|1`: same-origin MJPEG from the frame
  hub. With `boxes=1`, frames are re-encoded with the tracker's current boxes
  drawn. Replaces every link to `127.0.0.1:<proxy port>` in templates. The
  stdlib MJPEG proxy server stays for external consumers.
- `POST /cameras/{name}/rules/test`: fires the synthetic event.
- `POST /actions/{name}/test`: tests one action (replaces `GET /alerts/{name}/test`).

### 9.4 Pages

- Events list: card grid (thumbnail, label, confidence, camera, zone, time);
  filters for camera, label, date range; htmx partial refresh kept.
- Event detail: full thumbnail, clip player when present, action runs table.
- Dashboard: recent events strip uses the same cards.
- Camera detail: new "Detection" panel with `detect_fps`, detector list with
  `fps`, `device` badge (which provider is live), enabled toggles, class filter,
  zones link, rules list with the fields from 8.2, "Fire test event". Changing
  `detect_fps` shows "restarts this camera's ingest" next to the save button.
- Live preview: "show boxes" toggle on the camera detail page (off by default,
  remembered per browser).
- Actions page (replaces Alerts): name, type, last run, failure count, Test
  button, and a note that actions are edited in `config.yaml`.

## 10. Deployment and performance

- `Dockerfile.cuda`: slim image plus `onnxruntime-gpu`. `docker-compose.gpu.yml`
  overlay reserves the NVIDIA device. CPU images unchanged.
- `README` gains a short GPU section: install the NVIDIA container toolkit, run
  with the overlay, confirm "provider: CUDAExecutionProvider" in the logs.
- Expected cost per camera at 2 fps with YOLOX-s at 640: 10 to 15 ms per frame
  on an RTX 4080 class GPU; 60 to 120 ms on one modern CPU core. The runner's
  bounded queue drops the oldest frame under load, so overload degrades to fewer
  samples rather than growing memory. Dropped-frame counts are shown on the
  camera status panel.

## 11. Configuration summary

New or changed keys (full example in 8.2):

| Key | Where | Default | Notes |
|---|---|---|---|
| `detect_fps` | camera | 5 | 0.5 to 30; restarts ingest on change |
| `detectors[].type: onnx` | camera | | new detector type |
| `detectors[].model` | onnx spec | `yolox-s` | name of a registry entry |
| `detectors[].device` | onnx spec | `auto` | `auto`, `cuda`, `cpu` |
| `detectors[].fps` | any spec | camera `detect_fps` | must be <= `detect_fps` |
| `detectors[].min_confidence` | onnx spec | 0.5 | |
| `detectors[].events` | motion spec | false when an onnx detector is enabled | |
| `track_grace_seconds` | camera | 3.0 | |
| `min_track_frames` | camera | 2 | |
| `stationary_iou` | camera | 0.6 | RW-4: a track whose box still overlaps its first box by this IoU opens no event until it moves; 0 = off |
| `rules[]` | camera | `[]` | see 8.2 |
| `actions[]` | top level | `[]` | replaces `alerts.notifiers` |
| `runtime.models_dir` | top level | XDG cache | |
| `runtime.public_url` | top level | null | base for URLs in payloads |

Deprecated for one release, then removed: detector types `person`, `vehicle`,
`dnn`; `detectors[].interval_seconds`; top-level `alerts`.

## 12. Testing

- Model: tests build a tiny ONNX graph with `onnx.helper` at test time that
  returns a fixed detection tensor, so pre- and post-processing are tested
  offline without real weights. One test per `postprocess` type.
- Tracker: unit tests with synthetic boxes covering match, miss, grace expiry,
  best-frame update, and reset.
- Rule engine: unit tests for each predicate, cooldown keying, and `between`
  wrapping midnight.
- Action queue: unit test with a fake action asserting `action_runs` rows for
  success and failure, and overflow counting.
- Integration: a synthetic frame sequence through dispatcher, runner, tracker,
  sink and rule engine to a recording fake action, asserting exactly one event
  row, one thumbnail file and one action run.
- Clip: test the fixed regex against real recorder filenames and the `.ts`
  fallback when the remux command fails.
- Manual: Foscam C1 V3 on `videoMain`, `detect_fps: 5`, onnx `fps: 2`, walk
  through, expect one `person` event with thumbnail and one ntfy message.

## Appendix A: prerequisite sub-projects

### A.1 Stabilize (bounded)

Defects confirmed on 2026-10-02 against v1.3.0, by reading code and by running
the app in a browser against the Foscam:

- `app.state.runtime` and `app.state.config_path` are never set in `serve`;
  reload returns 503 and all config write-back is skipped.
- CSRF middleware reads header or query only; five plain forms and three htmx
  forms are rejected with 403 (sensitivity, detection classes, zone delete and
  reload, retention, clip generation, zone editor save, alert test).
- Login posts via htmx and the 303 is swapped into the login card.
- Unauthenticated HTML requests get JSON 401 instead of a redirect to login.
- `_persist_camera_field` writes the value into every camera.
- Clip segment regex never matches recorder filenames.
- Retention under `record:` is ignored; `init-config` template and all
  examples put it there. Warn and document `cameras[].retention`.
- Docker CMD omits `--web-host`; Typer defaults override `WARDEN_WEB_HOST` and
  `WARDEN_WEB_PORT`; `serve` never runs `ensure_schema` or creates the admin
  user, so a fresh container has no login. systemd `ExecReload` sends HUP with
  no handler.
- `AlertManager.start` never called; alert test button posts to a GET route.
- Camera status always `unknown`; no ffmpeg error surfaced; MJPEG and snapshot
  links point at 127.0.0.1.
- `sub_url` required although documented optional; make it optional, default
  all sub-stream consumers to main.
- Example configs contain real credentials (rotate, then scrub).
- `uv.lock` stale against `pyproject.toml`.
- Dead code and unused dependencies per the audit (`health_server.py`,
  `web_ui.py` and `ui` command, `run` alias, `web/schemas`, `zeep`, `aiofiles`,
  and the rest of the list), plus duplicated route helpers.
- README and other docs: the ~40 discrepancies from the audit.

### A.2 UI pass (bounded)

- Consistent page layout, nav, and form styling; fix the broken htmx and Alpine
  bindings on the ONVIF page.
- Camera cards show live status (running, restarting with countdown, failed
  with last ffmpeg error line).
- Add-camera flow: form with name, host, credentials, optional RTSP URLs;
  "Test connection" runs ffprobe with a timeout and shows codec, resolution,
  fps and a snapshot; when host and credentials are given, ONVIF
  (WS-UsernameToken, trying ports 80, 8080, 888, 2020) fills the RTSP URLs
  from `GetStreamUri`, rewriting the host to the one the camera was reached on;
  save writes `config.yaml` and hot-adds the camera. Edit and delete from the
  camera detail page.
- Mobile-width layout check for every page.

## Appendix B: decisions made during design

- ONNX Runtime with a permissive model over Ultralytics (AGPL) so the project
  stays MIT.
- In-process pipeline over an MQTT-first or separate-service design; MQTT is
  reachable through Apprise.
- Events are per object visit via a simple IoU tracker; no Kalman or re-id.
- `cameras`, `recordings`, `ingest_health` tables dropped; recordings stay
  file-system discovered.
- Clips remux to MP4 with a `.ts` fallback; live segment recording stays `.ts`.
- Pi and GPIO actions removed from scope at the user's request.

# Sprint 3 Plan — Detector Framework (target v0.7.0)

## Goal

Turn rtsp-warden from "it records" into "it records and *notices*". Add a
real-time computer-vision pipeline that consumes JPEG frames from the
existing `FrameConsumer` seam and produces structured `Event` rows in the DB.

## Architecture

```
recorder (FFmpeg, ingest thread)
   |
   v
FrameTapDispatcher (existing stable contract)
   |
   +---> MotionHeuristicConsumer (existing stdlib demo)
   |
   +---> MJPEG proxy hub
   |
   +---> DetectorRunner (NEW — FrameConsumer)
              |
              v
          [thread pool, decode JPEG with cv2, call Detector.process()]
              |
              +---> MotionDetector       (MOG2)
              +---> PersonDetector       (HOG)
              +---> VehicleDetector      (Haar)
              +---> CustomDetector       (import_path)
              |
              v
          EventSink (NEW — FrameConsumer)
              |
              v
          events table
```

### Key invariants

- `FrameConsumer` Protocol UNCHANGED — `DetectorRunner` and `EventSink` BOTH
  implement it; the dispatcher is unaware of detectors.
- `Detector` Protocol is NEW — operates on `np.ndarray` (decoded BGR frame)
  and `ts_unix: float`, returns `list[Detection]`.
- Detector work happens in WORKER THREADS — ingest threads are never blocked.
- `motion_demo` (stdlib) keeps working — it is just another consumer in the chain.
- `EventSink` is itself a `FrameConsumer` — it sees the same JPEG bytes as
  everything else. (Detectors go through `DetectorRunner`, not through EventSink
  directly, so the DB is only written when detections fire.)
- All public functions get type hints; all new modules get docstrings.
- No emoji, no breaking changes to config schema, additive only.

## Configuration (additive)

```yaml
cameras:
  - name: front_door
    main_url: rtsp://...
    sub_url:  rtsp://...
    detectors:
      - type: motion
        enabled: true
        interval_seconds: 1.0
        min_area: 500
        sensitivity: 0.5       # MOG2 var_threshold
      - type: person
        enabled: true
        interval_seconds: 2.0
        min_confidence: 0.5
      - type: vehicle
        enabled: true
        interval_seconds: 2.0
        min_size: 40
      - type: custom
        enabled: true
        import_path: my_pkg.detectors.CatDetector
        config:
          breed: "tabby"
```

## Batches

| # | Title | New files | Tests | Est. lines |
|---|-------|-----------|-------|------------|
| 1 | Foundation | 7 | 3 | ~600 |
| 2 | MotionDetector | 1 | 1 | ~150 |
| 3 | Person/Vehicle/Custom | 3 | 3 | ~400 |
| 4 | ROI/Mask | 1 (extended) | 1 | ~150 |
| 5 | Web UI + README | 4 (touched) | 1 | ~200 |

Total target: 170 + ~40 = ~210 tests passing, 0 lint errors in new code.

## Batch 1 — Foundation (1 file: ~50 lines average)

**Files:**
- `src/rtsp_warden/detectors/__init__.py` — package marker + exports
- `src/rtsp_warden/detectors/base.py` — `Detector` Protocol, `Detection` dataclass, `Frame` (cv2 frame container), `apply_masks()`
- `src/rtsp_warden/detectors/runner.py` — `DetectorRunner` (FrameConsumer; thread pool, per-camera queues)
- `src/rtsp_warden/detectors/sinks.py` — `EventSink` (FrameConsumer; writes events table)
- `src/rtsp_warden/detectors/registry.py` — `DetectorSpec` pydantic + factory
- `src/rtsp_warden/db/schema.py` — add `create_event()`, `list_events()`, `count_events()`, `get_event_by_id()`
- `src/rtsp_warden/config.py` — add `DetectorSpec` model + `CameraConfig.detectors`
- `src/rtsp_warden/cli.py` — add `--detectors` / `--no-detectors` flag (default ON)
- `src/rtsp_warden/app.py` — wire `DetectorRunner` into `AppRuntime`
- `tests/test_detectors_base.py`
- `tests/test_detector_runner.py`
- `tests/test_event_sink.py`

## Batch 2 — MotionDetector

**Files:**
- `src/rtsp_warden/detectors/builtin/__init__.py`
- `src/rtsp_warden/detectors/builtin/motion.py` — `MotionDetector` (MOG2)
- `tests/test_motion_detector.py` — synthetic frames

## Batch 3 — Person/Vehicle/Custom

**Files:**
- `src/rtsp_warden/detectors/builtin/person.py` — `PersonDetector` (HOG)
- `src/rtsp_warden/detectors/builtin/vehicle.py` — `VehicleDetector` (Haar)
- `src/rtsp_warden/detectors/builtin/custom.py` — `CustomDetector` (import_path)
- `tests/test_person_detector.py`
- `tests/test_vehicle_detector.py`
- `tests/test_custom_detector.py`

## Batch 4 — ROI/Mask

**Files:**
- `src/rtsp_warden/detectors/roi.py` — `ROI`, `Mask` (polygon-based)
- Update `src/rtsp_warden/detectors/base.py` — `apply_masks()` from Batch 1 expanded
- Update `src/rtsp_warden/detectors/registry.py` — `DetectorSpec` accepts `roi` and `masks`
- `tests/test_roi.py`

## Batch 5 — Web UI + README

**Files:**
- Update `src/rtsp_warden/web/services/events.py` — `count_events_by_type()`, `get_recent_events()`
- Update `src/rtsp_warden/web/services/cameras.py` — `get_camera_detectors_status()`
- Update `src/rtsp_warden/web/services/runtime.py` — add detector count to runtime status
- Update `src/rtsp_warden/web/routes/cameras.py` — `/cameras/{name}/detectors` htmx partial
- Update `src/rtsp_warden/web/templates/dashboard.html` — "Detections today" stat card
- Update `src/rtsp_warden/web/templates/cameras/detail.html` — detector list
- Update `src/rtsp_warden/web/templates/events/list.html` — htmx auto-refresh
- Update `src/rtsp_warden/web/templates/partials/camera_card.html` — detector indicator
- Update `README.md` — detector config example, "Detectors" section, changelog
- `tests/test_detector_integration.py` — end-to-end with synthetic JPEG stream

## CLI additions

```
rtsp-warden serve -c config.yaml
    [--detectors | --no-detectors]   # default: --detectors
```

Detectors can be **enabled per-camera** in config.yaml (`detectors[i].enabled`)
and **globally toggled** with the CLI flag. The CLI flag is the master switch.

## Out of scope (Sprint 4 / v1.0.0)

- Alerts (ntfy, webhook, email/apprise)
- ONVIF discovery + PTZ
- Audio opt-in per camera
- Continuous vs event-only recording toggle
- Timeline canvas component
- Clip generation from detections
- Docker image + systemd unit
- Alembic migrations
- Tailscale / remote access / mobile

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| OpenCV weight (~50MB) | Already declared hard dep, no change |
| HOG person detector is slow on large frames | Downscale to 320px wide before detect |
| DB write contention (events) | Low-rate writes; current pattern is fine |
| First-run: no detectors in config | MotionDetector ON by default, banner on events page |
| Detector crashes must not kill ingest | Runner catches exceptions per-detector, logs debug |
| CustomDetector security: arbitrary import_path | Document warning; only load from trusted config |
| Frame backpressure: detector falls behind | Per-camera bounded queue, drop oldest on overflow |

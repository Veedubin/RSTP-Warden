# Sprint 4 Plan — Closure to v1.0.0

## Goal

Ship the remaining v1.0.0 features:
1. **Alerts** (ntfy + webhook) — the "tell me when something happens" feature
2. **Audio opt-in** per camera
3. **Recording modes** (continuous vs event-only)
4. **ONVIF discovery + PTZ** (camera control)
5. **Alembic migrations** (replace `ensure_schema()`)
6. **Timeline canvas** (recording scrubber UI)
7. **Docker image** (multi-stage)
8. **Systemd unit** (Linux service)

Post-v1 deferred: mobile PWA, Tailscale, apprise/email alerts.

## Architecture

### Phase 0 (orchestrator, before parallel)
Pre-touch `config.py` to add new top-level sections:
- `AlertsConfig` with `notifiers: list[NotifierSpec]`
- `RecordConfig.audio: bool = False` (Batch 2b)
- `RecordConfig.mode: Literal["continuous","event"] = "continuous"`
- `EventRecordConfig` (pre/post seconds for event-only)
- `OnvifConfig` (optional WS-Discovery)
- `NotifierSpec` pydantic for `type: ntfy | webhook`, url, headers, etc.

### Phase 1 (parallel — 8 sub-agents fire simultaneously)

| # | Topic | New files | Files modified |
|---|-------|-----------|---------------|
| 1 | Alerts | `src/rtsp_warden/alerts/__init__.py`, `base.py`, `ntfy.py`, `webhook.py`, `manager.py`; `src/rtsp_warden/web/routes/alerts.py`; `src/rtsp_warden/web/templates/alerts/*` | `cli.py` (add --test-alert flag) |
| 2 | Audio | `tests/test_audio.py` | `src/rtsp_warden/ffmpeg.py` (audio arg), `src/rtsp_warden/recorder.py` (audio handling) |
| 3 | Recording modes | `tests/test_recording_modes.py` | `src/rtsp_warden/recorder.py` (mode flag), `src/rtsp_warden/db/schema.py` (event-window queries) |
| 4 | ONVIF/PTZ | `src/rtsp_warden/onvif/__init__.py`, `discovery.py`, `ptz.py`; `src/rtsp_warden/web/routes/onvif.py`; `src/rtsp_warden/web/templates/onvif/*`; `src/rtsp_warden/web/services/onvif.py` | (mostly isolated) |
| 5 | Alembic | `migrations/env.py`, `migrations/versions/0001_initial.py`, `alembic.ini` | `src/rtsp_warden/db/schema.py` (call alembic upgrade on ensure_schema) |
| 6 | Timeline canvas | `src/rtsp_warden/web/static/js/timeline.js`, `src/rtsp_warden/web/templates/partials/timeline.html`; `src/rtsp_warden/web/routes/api.py`; `src/rtsp_warden/web/services/timeline.py` | (mostly isolated) |
| 7 | Docker | `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `docker/README.md` | (none) |
| 8 | Systemd | `packaging/systemd/rtsp-warden.service`, `packaging/systemd/README.md`, `packaging/systemd/install.sh` | (none) |

### Phase 2 (orchestrator, after all parallel done)
- Wire `app.py` to wire alerts manager into the event sink
- Add web routes to expose new features
- Update templates to link to new pages
- Update README to mention v1.0.0

### Phase 3 (final)
- All tests pass
- Lint clean
- Live smoke test
- Save to memini-ai

## New dependencies

- `httpx` (move from dev to runtime; needed for webhook alerts; ntfy uses simple POST)
- `zeep` (ONVIF SOAP client; required only if ONVIF enabled)

## Out of scope (post-v1)

- Mobile PWA / Tailscale / remote access
- Email alerts (apprise)
- Tailscale/Caddy/LetsEncrypt
- Clip generation from detections
- YOLO / DNN detectors (real vehicle detection)
- Multi-camera PTZ presets
- ONVIF events (motion events via ONVIF, separate from our detectors)

## CLI additions

```
rtsp-warden serve -c config.yaml
    [--alerts | --no-alerts]      # default: --alerts
    [--test-alert]                # send a test notification and exit
```

## Web UI additions

- `/alerts` — list, add, edit, test, delete notifiers
- `/onvif` — discover cameras, see list, click to add as new camera
- `/cameras/{name}` — PTZ controls (left/right/up/down/zoom)
- `/cameras/{name}/settings` — recording mode toggle (continuous/event), audio toggle
- `/recordings/{id}` — timeline canvas scrubber
- `/api/recordings/{id}/timeline` — JSON: `[{start, end, path}, ...]`

## Risks

| Risk | Mitigation |
|------|-----------|
| `httpx` was dev-only; making it runtime | Update pyproject; small dep (~50KB) |
| `zeep` is large (~3MB) | Optional; only loaded if ONVIF used |
| ONVIF on test networks (no real cameras) | Mock + test with local fixture; provide skip-on-no-network |
| 8 parallel sub-agents may hit rate limits | Use sequential when possible; Ollama Cloud has 10-slot limit |
| Alembic replacing ensure_schema | Keep ensure_schema as a fallback if alembic fails on existing DBs |
| Recording modes + continuous default | Continuous stays default; event-only is opt-in |

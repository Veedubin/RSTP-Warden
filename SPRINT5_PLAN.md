# Sprint 5 Implementation Plan — rtsp-warden v1.1.0

**Baseline:** v1.0.0 (commit `faf133c`), 325 tests passing, 36 web routes, 17 runtime deps.
**Goal:** 5 post-v1 features, additive layers, zero regressions.

---

## Feature 1: Email Alerts via Apprise

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **CREATE** | `src/rtsp_warden/alerts/apprise.py` | AppriseNotifier implementing Notifier Protocol |
| **MODIFY** | `src/rtsp_warden/config.py` | Extend `NotifierSpec.type` literal, add `apprise_url` field |
| **MODIFY** | `src/rtsp_warden/alerts/factory.py` | Add `type == "apprise"` branch |
| **MODIFY** | `src/rtsp_warden/web/templates/alerts/new.html` | Add apprise URL field to form |
| **MODIFY** | `src/rtsp_warden/web/templates/alerts/edit.html` | Add apprise URL field to form |
| **MODIFY** | `src/rtsp_warden/web/templates/alerts/list.html` | Show apprise URL in notifier card |
| **CREATE** | `tests/test_apprise_notifier.py` | Unit tests for AppriseNotifier |
| **MODIFY** | `pyproject.toml` | Add `apprise>=1.9.0` to dependencies |

### 2. New Pydantic Config Types

In `config.py`, modify `NotifierSpec`:

```python
class NotifierSpec(BaseModel):
    name: str
    type: Literal["ntfy", "webhook", "apprise"]  # extended
    url: str
    enabled: bool = True

    # ntfy-specific
    topic: str | None = None
    token: str | None = None
    priority: int | None = None

    # webhook-specific
    headers: dict[str, str] = Field(default_factory=dict)
    method: Literal["POST", "PUT"] = "POST"

    # apprise-specific
    apprise_url: str | None = None  # e.g. "mailto://user:pass@gmail.com"

    # Common
    min_interval_seconds: int = 30
    severities: list[Literal["info", "warn", "error"]] = Field(
        default_factory=lambda: ["warn", "error"]
    )

    @field_validator("apprise_url")
    @classmethod
    def _apprise_url_valid(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("apprise_url must be non-empty if provided")
        return v
```

### 3. New Runtime Deps

**`apprise>=1.9.0`** — Justification: Pure-Python library (MIT license), single dependency that unlocks 90+ notification services (email via `mailto://`, Slack, Discord, Telegram, Pushover, etc.). Well-maintained, ~0 transitive deps. The user explicitly requested email support; apprise is the standard Python solution.

### 4. Public API Surface

```python
# src/rtsp_warden/alerts/apprise.py

class AppriseNotifier:
    """Send notifications via the Apprise library (email + 90+ services).

    Args:
        name: Human-readable notifier label.
        apprise_url: Apprise URL string (e.g. "mailto://user:pass@gmail.com").
        tag: Optional Apprise tag for grouping (default "rtsp-warden").
    """

    name: str
    type: str  # "apprise"

    def __init__(self, name: str, apprise_url: str, tag: str = "rtsp-warden") -> None: ...

    async def send(self, event: dict[str, Any]) -> NotificationResult: ...
    async def test(self) -> NotificationResult: ...
    async def close(self) -> None: ...
```

The `send()` method formats the event into a title/body string and calls `apprise.Apprise().notify()`. The `test()` method sends a fixed test message. `close()` is a no-op (apprise is stateless) but required by the Notifier Protocol for `AlertManager.stop()` compatibility.

### 5. Test Strategy

**File:** `tests/test_apprise_notifier.py`

| Test | What it verifies |
|------|-----------------|
| `test_apprise_send_success` | Mock `apprise.Apprise.notify` returns True, verify `NotificationResult(success=True)` |
| `test_apprise_send_failure` | Mock returns False, verify `NotificationResult(success=False, error=...)` |
| `test_apprise_send_exception` | Mock raises, verify graceful error capture |
| `test_apprise_test` | Mock test notification, verify result |
| `test_apprise_close` | Verify close() is a no-op (no exception) |
| `test_apprise_factory` | `build_notifier(NotifierSpec(type="apprise", ...))` returns AppriseNotifier |
| `test_apprise_config_validation` | `NotifierSpec(type="apprise", apprise_url="")` raises ValidationError |

Use `unittest.mock.patch("apprise.Apprise")` to avoid real network calls. Follow existing patterns in `tests/test_alerts.py`.

### 6. Web UI Changes

**Routes:** No new routes needed. Existing `/alerts` routes work generically.

**Templates:**
- `alerts/new.html` — Add conditional field group for apprise: when type=apprise, show `apprise_url` input instead of ntfy/webhook fields.
- `alerts/edit.html` — Same conditional field group.
- `alerts/list.html` — Show `n.apprise_url` in the notifier card when type is apprise.

Use Alpine.js `x-show` for conditional form fields (already in the project's tech stack).

### 7. Migration Needs

**None.** Notifiers are config-driven (stored in `config.yaml`), not in the database.

### 8. Config Example

```yaml
alerts:
  enabled: true
  notifiers:
    - name: Gmail alerts
      type: apprise
      apprise_url: mailto://myuser:mypass@gmail.com
      severities: [warn, error]
      min_interval_seconds: 60
    - name: Family Telegram
      type: apprise
      apprise_url: tgram://BOT_TOKEN/CHAT_ID
      severities: [info, warn, error]
      min_interval_seconds: 30
```

### 9. README Updates

- **Alerts section:** Add "Apprise" subsection documenting the new type, example `mailto://` URL, and link to [Apprise wiki](https://github.com/caronc/apprise/wiki) for the full URL catalog.
- **Dependencies table:** Add `apprise` entry.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Apprise import fails at runtime | Low | Medium | `build_notifier` catches ImportError, logs warning, returns NullNotifier-like fallback (or raises clear error). Follow existing pattern in `factory.py`. |
| Email credentials in config.yaml | Medium | High | Document that `apprise_url` contains credentials. Recommend environment variable substitution (`${SMTP_PASS}`) or using apprise config file. Add note in README. |
| Apprise blocking in async context | Low | Medium | `apprise.notify()` is synchronous. Wrap in `asyncio.to_thread()` inside `send()` to avoid blocking the event loop. |

---

## Feature 2: Real YOLO/DNN Detector (Vehicles + Animals)

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **CREATE** | `src/rtsp_warden/detectors/builtin/dnn.py` | DNNDetector class using OpenCV DNN + YOLOv4-tiny |
| **CREATE** | `src/rtsp_warden/detectors/builtin/model_utils.py` | Model download/caching/verification utilities |
| **MODIFY** | `src/rtsp_warden/detectors/registry.py` | Add `"dnn"` to `DetectorType` literal, add branch in `build_detector` |
| **MODIFY** | `src/rtsp_warden/config.py` | No config changes needed (uses existing `DetectorSpec.config` dict) |
| **CREATE** | `tests/test_dnn_detector.py` | Unit tests for DNNDetector |
| **CREATE** | `tests/test_model_utils.py` | Unit tests for model download/caching |

### 2. New Pydantic Config Types

No new top-level config types. The `DetectorSpec` already has a `config: dict[str, Any]` field for type-specific parameters. The DNN detector reads from `spec.config`:

```python
# DetectorSpec.config keys for type=dnn:
#   model_path: str | None  — path to .weights file (auto-download if None)
#   config_path: str | None — path to .cfg file (bundled)
#   classes: list[str] | None — COCO classes to detect (default: vehicle + animal classes)
#   confidence_threshold: float — minimum confidence (default 0.5)
#   nms_threshold: float — NMS threshold (default 0.4)
#   input_width: int — DNN input width (default 416)
#   input_height: int — DNN input height (default 416)
```

Extend `DetectorType` in `registry.py`:

```python
DetectorType = Literal["motion", "person", "vehicle", "dnn", "custom"]
```

### 3. New Runtime Deps

**No new deps.** OpenCV's DNN module (`cv2.dnn`) is already included in `opencv-python-headless>=4.10.0`. YOLOv4-tiny `.cfg` file is bundled (~2KB text file). `.weights` file (~24MB) is downloaded on first use to `~/.cache/rtsp-warden/models/` with SHA256 verification.

### 4. Public API Surface

```python
# src/rtsp_warden/detectors/builtin/dnn.py

@dataclass(slots=True)
class DNNDetector:
    """YOLOv4-tiny DNN detector for vehicles and animals.

    Uses OpenCV's DNN module to run YOLOv4-tiny inference on CPU.
    Detects COCO classes: car, truck, bus, motorcycle, bicycle (vehicles)
    and bird, cat, dog, horse, sheep, cow, bear, zebra, giraffe (animals).

    Args:
        model_path: Path to .weights file. Auto-downloaded if None.
        config_path: Path to .cfg file. Uses bundled yolov4-tiny.cfg if None.
        classes: List of COCO class names to detect. Defaults to vehicle+animal.
        confidence_threshold: Minimum confidence to report (default 0.5).
        nms_threshold: Non-maximum suppression threshold (default 0.4).
        input_size: DNN input dimensions (default (416, 416)).
        name: Detector name for protocol compliance.
        kind: Detection kind string (default "dnn").
    """

    name: str = "dnn"
    kind: str = "dnn"
    model_path: str | None = None
    config_path: str | None = None
    classes: list[str] | None = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_width: int = 416
    input_height: int = 416

    def setup(self) -> None: ...
    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]: ...
    def teardown(self) -> None: ...


# src/rtsp_warden/detectors/builtin/model_utils.py

MODEL_CACHE_DIR: Path  # ~/.cache/rtsp-warden/models/
YOLOV4_TINY_WEIGHTS_URL: str
YOLOV4_TINY_WEIGHTS_SHA256: str

def ensure_model(
    model_path: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Return path to .weights file, downloading if necessary.

    If model_path is provided and exists, return it.
    Otherwise download yolov4-tiny.weights to cache_dir,
    verify SHA256, and return the cached path.
    """

def get_default_classes() -> list[str]:
    """Return the default COCO class list (vehicle + animal classes)."""
```

The bundled `yolov4-tiny.cfg` lives at `src/rtsp_warden/detectors/builtin/models/yolov4-tiny.cfg`.

### 5. Test Strategy

**File:** `tests/test_dnn_detector.py`

| Test | What it verifies |
|------|-----------------|
| `test_dnn_setup_loads_model` | Mock `cv2.dnn.readNet`, verify setup succeeds |
| `test_dnn_setup_missing_model` | Model file missing, verify graceful error + _loaded=False |
| `test_dnn_process_returns_detections` | Synthetic frame, mock DNN forward pass returns known boxes, verify Detection list |
| `test_dnn_process_confidence_filter` | Mock returns low-confidence box, verify it's filtered out |
| `test_dnn_process_nms` | Mock returns overlapping boxes, verify NMS reduces them |
| `test_dnn_process_empty_frame` | None/empty frame, verify returns [] |
| `test_dnn_process_not_loaded` | process() before setup(), verify returns [] |
| `test_dnn_teardown` | Verify net is released |
| `test_dnn_registry` | `build_detector(DetectorSpec(type="dnn", ...))` returns DNNDetector |

**File:** `tests/test_model_utils.py`

| Test | What it verifies |
|------|-----------------|
| `test_ensure_model_cached` | Model already in cache, returns path without download |
| `test_ensure_model_downloads` | Model not cached, mock download, verify SHA256 check |
| `test_ensure_model_sha256_mismatch` | Downloaded file has wrong hash, verify error raised |
| `test_ensure_model_explicit_path` | model_path provided and exists, returns it directly |

Use `unittest.mock.patch("cv2.dnn.readNet")` and `unittest.mock.patch("urllib.request.urlretrieve")`.

### 6. Web UI Changes

No new routes. The existing `/cameras/{name}/detectors` htmx partial already renders detector type generically. The `kind` field on Detection objects will be `"vehicle"` or `"animal"` (mapped from COCO class), which the events list already displays.

### 7. Migration Needs

**None.** Detectors are config-driven.

### 8. Config Example

```yaml
cameras:
  - name: driveway
    main_url: rtsp://user:pass@192.168.1.50:554/main
    sub_url: rtsp://user:pass@192.168.1.50:554/sub
    detectors:
      - type: dnn
        enabled: true
        interval_seconds: 2.0
        config:
          confidence_threshold: 0.6
          classes: [car, truck, person, dog]  # optional override
      - type: motion
        enabled: true
        interval_seconds: 1.0
        min_area: 500
```

### 9. README Updates

- **Detectors section:** Add "DNN (YOLOv4-tiny)" row to the detector types table. Document that it detects vehicles AND animals from COCO classes. Note the first-run model download (~24MB). Mention the `"vehicle"` Haar type is now legacy; recommend `"dnn"` for new setups.
- **Dependencies:** Note that no new pip deps are needed (OpenCV DNN is already included).

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 24MB download on first run | High | Low | Document in README. Show progress log message. Cache is persistent. |
| SHA256 mismatch on download | Low | Medium | Verify hash after download. On mismatch, delete corrupt file and retry once. If still mismatched, raise clear error with manual download instructions. |
| CPU inference too slow | Medium | Medium | YOLOv4-tiny is designed for CPU (~100-200ms/frame at 416x416). Configurable `interval_seconds` already exists. Document expected latency. |
| OpenCV DNN missing (rare build) | Low | High | `build_detector` catches ImportError, falls back to NullDetector with warning log. |
| Bundled .cfg file path resolution | Low | Low | Use `Path(__file__).parent / "models" / "yolov4-tiny.cfg"` — same pattern as existing cascade bundling. |

---

## Feature 3: ONVIF Events Subscription

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **CREATE** | `src/rtsp_warden/onvif/events.py` | OnvifEventSubscriber for PullPoint subscription |
| **MODIFY** | `src/rtsp_warden/config.py` | Extend `OnvifConfig` with events fields |
| **MODIFY** | `src/rtsp_warden/web/routes/onvif.py` | Add event subscription status to ONVIF page |
| **MODIFY** | `src/rtsp_warden/web/templates/onvif/index.html` | Show events subscription status |
| **CREATE** | `tests/test_onvif_events.py` | Unit tests for OnvifEventSubscriber |

### 2. New Pydantic Config Types

In `config.py`, extend `OnvifConfig`:

```python
class OnvifConfig(BaseModel):
    discovery_enabled: bool = False
    ptz_enabled: bool = False
    discovery_timeout_seconds: int = 5
    ptz_timeout_seconds: int = 10
    username: str | None = None
    password: str | None = None

    # NEW: event subscription
    events_enabled: bool = False
    events_poll_interval_seconds: int = 10  # how often to PullMessages

    @field_validator("events_poll_interval_seconds")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("events_poll_interval_seconds must be > 0")
        return v
```

### 3. New Runtime Deps

**No new deps.** Uses existing `httpx` for SOAP requests and `xml.etree.ElementTree` for XML parsing — same stack as `onvif/ptz.py`.

### 4. Public API Surface

```python
# src/rtsp_warden/onvif/events.py

class OnvifEventSubscriber:
    """PullPoint subscription client for ONVIF event services.

    Creates a PullPoint subscription on the camera's event service,
    then periodically calls PullMessages to retrieve motion, tampering,
    and other ONVIF events. Maps ONVIF event topics to rtsp-warden
    event types and routes them into the AlertManager / event DB.

    Args:
        device_xaddr: ONVIF device service URL.
        event_xaddr: Optional event service URL (auto-resolved if None).
        username: Optional ONVIF credentials.
        password: Optional ONVIF credentials.
        timeout_seconds: Per-request HTTP timeout.
        poll_interval_seconds: Interval between PullMessages calls.
        on_event: Callback receiving (camera_name, event_type, message, metadata).
    """

    def __init__(
        self,
        device_xaddr: str,
        event_xaddr: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout_seconds: int = 10,
        poll_interval_seconds: int = 10,
        on_event: Callable[[str, str, str, dict], Awaitable[None]] | None = None,
    ) -> None: ...

    async def subscribe(self) -> str:
        """Create a PullPoint subscription. Returns the subscription ID."""

    async def pull_messages(self) -> list[OnvifEvent]:
        """Pull pending messages from the subscription."""

    async def unsubscribe(self) -> None:
        """Terminate the subscription."""

    async def run_poll_loop(self, camera_name: str) -> None:
        """Subscribe and poll indefinitely, calling on_event for each message.
        Runs until cancelled. Designed to be launched as an asyncio Task."""

    async def close(self) -> None: ...


@dataclass(slots=True)
class OnvifEvent:
    """A single ONVIF event message."""
    topic: str           # e.g. "tns1:RuleEngine/CellMotionDetector/Motion"
    timestamp: datetime
    data: dict[str, Any]  # parsed SimpleItem name-value pairs
```

SOAP operations implemented (handcrafted envelopes, same pattern as `ptz.py`):
- `CreatePullPointSubscription` — event service action
- `PullMessages` — with optional `Timeout` and `MessageLimit`
- `Unsubscribe` — terminate subscription

ONVIF event topic mapping:
```
"Motion" → event_type="motion", severity="info"
"Tampering" → event_type="tamper", severity="warn"
"VideoLoss" → event_type="ingest_lost", severity="error"
```

### 5. Test Strategy

**File:** `tests/test_onvif_events.py`

| Test | What it verifies |
|------|-----------------|
| `test_subscribe_success` | Mock SOAP response with subscription ID, verify returned |
| `test_subscribe_auth_failure` | Mock 401 response, verify OnvifError raised |
| `test_pull_messages_returns_events` | Mock PullMessagesResponse with Motion topic, verify parsed OnvifEvent list |
| `test_pull_messages_empty` | Mock response with no messages, verify empty list |
| `test_unsubscribe` | Mock UnsubscribeResponse, verify no error |
| `test_topic_mapping` | OnvifEvent(topic="...Motion") maps to event_type="motion" |
| `test_poll_loop_calls_callback` | Mock subscriber, run one poll cycle, verify on_event called |
| `test_config_validation` | `OnvifConfig(events_poll_interval_seconds=0)` raises ValidationError |

Use `respx` (or `unittest.mock.patch("httpx.AsyncClient.post")`) to mock SOAP responses. Follow patterns in `tests/test_onvif.py`.

### 6. Web UI Changes

**Routes:** No new routes. The existing `/onvif` page gets a new section.

**Templates:** `onvif/index.html` — Add a third `<article>` section below PTZ control:

```html
<article>
  <header>Event Subscription</header>
  <p>Status: {{ "enabled" if cfg.onvif.events_enabled else "disabled" }}</p>
  <p>Poll interval: {{ cfg.onvif.events_poll_interval_seconds }}s</p>
  <footer>
    <small>ONVIF events (motion, tampering) are routed into the alert system and event database.</small>
  </footer>
</article>
```

### 7. Migration Needs

**None.** Events use the existing `events` table. No schema changes.

### 8. Config Example

```yaml
onvif:
  discovery_enabled: true
  ptz_enabled: true
  events_enabled: true
  events_poll_interval_seconds: 10
  discovery_timeout_seconds: 5
  username: admin
  password: CHANGEME
```

### 9. README Updates

- **ONVIF section:** Add "Event Subscription" subsection. Document that ONVIF motion/tampering events are polled and routed into the alert system. Note PullPoint vs. WS-BaseNotification trade-off (PullPoint is simpler, no callback server needed). Mention the `events_enabled` and `events_poll_interval_seconds` config keys.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Camera doesn't support PullPoint | Medium | Medium | `subscribe()` raises clear OnvifError. Caller catches and logs. System continues without ONVIF events. |
| Subscription times out | Medium | Low | ONVIF PullPoint subscriptions have a default lifetime. `run_poll_loop` catches timeout, re-subscribes automatically. |
| Polling adds HTTP load | Low | Low | Configurable `events_poll_interval_seconds` (default 10s). One request per camera per interval. |
| SOAP namespace variations | Medium | Medium | Use the same namespace constants as `ptz.py`. Test against real camera responses. Add fallback namespace parsing. |

---

## Feature 4: Multi-Camera PTZ Presets

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **CREATE** | `src/rtsp_warden/onvif/presets.py` | PresetManager for save/recall/delete |
| **MODIFY** | `src/rtsp_warden/config.py` | Add `PTZPreset` model, add `ptz_presets` to `CameraConfig` |
| **MODIFY** | `src/rtsp_warden/web/routes/onvif.py` | Add preset CRUD routes, make PTZ route dynamic (not hardcoded to `front_door`) |
| **MODIFY** | `src/rtsp_warden/web/templates/onvif/index.html` | Per-camera PTZ with preset management UI |
| **CREATE** | `tests/test_ptz_presets.py` | Unit tests for PresetManager and routes |

### 2. New Pydantic Config Types

In `config.py`:

```python
class PTZPreset(BaseModel):
    """A named PTZ preset position for a camera."""
    name: str
    pan: float
    tilt: float
    zoom: float

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("preset name must be non-empty")
        return v2


class CameraConfig(BaseModel):
    name: str
    main_url: str
    sub_url: str
    record: RecordConfig = Field(default_factory=RecordConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    detectors: list[DetectorSpec] = Field(default_factory=list)
    ptz_presets: dict[str, PTZPreset] = Field(default_factory=dict)  # NEW
```

### 3. New Runtime Deps

**No new deps.** Uses existing `OnvifPTZ.absolute_move()`.

### 4. Public API Surface

```python
# src/rtsp_warden/onvif/presets.py

class PresetManager:
    """Manage per-camera PTZ presets stored in config.yaml.

    Presets are keyed by camera name and preset name. Operations
    read/write the AppConfig's camera.ptz_presets dict. The config
    must be re-saved to disk by the caller (web route handler).

    Args:
        cfg: The live AppConfig instance.
    """

    def __init__(self, cfg: AppConfig) -> None: ...

    def list_presets(self, camera_name: str) -> dict[str, PTZPreset]:
        """Return all presets for a camera."""

    def get_preset(self, camera_name: str, preset_name: str) -> PTZPreset:
        """Get a single preset. Raises KeyError if not found."""

    def save_preset(
        self, camera_name: str, preset_name: str, pan: float, tilt: float, zoom: float
    ) -> PTZPreset:
        """Save a new preset or overwrite an existing one."""

    def delete_preset(self, camera_name: str, preset_name: str) -> bool:
        """Delete a preset. Returns True if it existed."""

    async def goto_preset(
        self,
        camera_name: str,
        preset_name: str,
        username: str | None = None,
        password: str | None = None,
        timeout_seconds: int = 10,
    ) -> None:
        """Move the camera to a saved preset position using OnvifPTZ.absolute_move()."""
```

### 5. Test Strategy

**File:** `tests/test_ptz_presets.py`

| Test | What it verifies |
|------|-----------------|
| `test_list_presets_empty` | New camera returns empty dict |
| `test_save_and_list_presets` | Save 2 presets, list returns both |
| `test_get_preset_exists` | Get returns correct PTZPreset |
| `test_get_preset_missing` | Get raises KeyError |
| `test_delete_preset` | Delete removes preset, returns True |
| `test_delete_preset_missing` | Delete non-existent returns False |
| `test_goto_preset_calls_absolute_move` | Mock OnvifPTZ, verify absolute_move called with correct pan/tilt/zoom |
| `test_preset_config_roundtrip` | Save preset, serialize AppConfig to YAML, reload, verify preset survives |
| `test_web_list_presets` | GET /onvif/cameras/{name}/presets returns JSON list |
| `test_web_save_preset` | POST /onvif/cameras/{name}/presets creates preset |
| `test_web_goto_preset` | POST /onvif/cameras/{name}/presets/{name}/goto calls absolute_move |
| `test_web_delete_preset` | DELETE /onvif/cameras/{name}/presets/{name} removes preset |
| `test_web_ptz_dynamic_camera` | POST /onvif/cameras/{name}/ptz works for any camera name, not just front_door |

### 6. Web UI Changes

**New routes** (in `src/rtsp_warden/web/routes/onvif.py`):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/onvif/cameras/{name}/presets` | List presets as JSON |
| `POST` | `/onvif/cameras/{name}/presets` | Save a preset (body: `{name, pan, tilt, zoom}`) |
| `POST` | `/onvif/cameras/{name}/presets/{preset_name}/goto` | Move to preset |
| `DELETE` | `/onvif/cameras/{name}/presets/{preset_name}` | Delete preset |

**Modified route:**
- `POST /onvif/cameras/{name}/ptz` — Already dynamic by `{name}` path param. Remove the hardcoded `front_door` from the template.

**Template:** `onvif/index.html` — Replace the hardcoded PTZ section with:

```html
<article x-data="{ camera: '', presets: [] }">
  <header>PTZ Control</header>
  <p>PTZ is {{ "enabled" if cfg.onvif.ptz_enabled else "disabled" }} globally.</p>

  <!-- Camera selector -->
  <select x-model="camera" @change="fetchPresets()">
    <option value="">-- Select camera --</option>
    {% for cam in cfg.cameras %}
    <option value="{{ cam.name }}">{{ cam.name }}</option>
    {% endfor %}
  </select>

  <!-- PTZ pad (shown when camera selected) -->
  <div x-show="camera" class="ptz-pad">
    <button hx-post="`/onvif/cameras/${camera}/ptz`" hx-vals='{"action":"up"}'>Up</button>
    <button hx-post="`/onvif/cameras/${camera}/ptz`" hx-vals='{"action":"down"}'>Down</button>
    <button hx-post="`/onvif/cameras/${camera}/ptz`" hx-vals='{"action":"left"}'>Left</button>
    <button hx-post="`/onvif/cameras/${camera}/ptz`" hx-vals='{"action":"right"}'>Right</button>
    <button hx-post="`/onvif/cameras/${camera}/ptz`" hx-vals='{"action":"zoom_in"}'>Zoom In</button>
    <button hx-post="`/onvif/cameras/${camera}/ptz`" hx-vals='{"action":"zoom_out"}'>Zoom Out</button>
    <button hx-post="`/onvif/cameras/${camera}/ptz`" hx-vals='{"action":"stop"}'>Stop</button>
  </div>

  <!-- Presets (shown when camera selected) -->
  <div x-show="camera">
    <h3>Presets</h3>
    <template x-for="p in presets" :key="p.name">
      <div>
        <span x-text="p.name"></span>
        <button hx-post="`/onvif/cameras/${camera}/presets/${p.name}/goto`">Go</button>
        <button hx-delete="`/onvif/cameras/${camera}/presets/${p.name}`"
                hx-target="closest div" hx-swap="outerHTML">Delete</button>
      </div>
    </template>
    <form hx-post="`/onvif/cameras/${camera}/presets`" hx-swap="beforeend">
      <input name="name" placeholder="Preset name" required>
      <button>Save current position</button>
    </form>
  </div>
</article>
```

The `fetchPresets()` Alpine.js function calls `GET /onvif/cameras/${camera}/presets` and populates the `presets` array.

### 7. Migration Needs

**None.** Presets are stored in `config.yaml` under each camera's `ptz_presets` dict. No DB changes.

### 8. Config Example

```yaml
cameras:
  - name: front_door
    main_url: rtsp://user:pass@192.168.1.50:554/main
    sub_url: rtsp://user:pass@192.168.1.50:554/sub
    ptz_presets:
      front_gate:
        name: front_gate
        pan: 0.25
        tilt: 0.30
        zoom: 0.0
      driveway:
        name: driveway
        pan: 0.75
        tilt: 0.20
        zoom: 0.5
```

### 9. README Updates

- **ONVIF section:** Add "PTZ Presets" subsection. Document the `ptz_presets` config key, the web UI for saving/recalling presets, and the new routes.
- **Config reference:** Add `ptz_presets` to the `CameraConfig` documentation.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Config.yaml save conflicts | Low | Medium | PresetManager operates on in-memory AppConfig. Web routes write back to config.yaml after each mutation. Use file locking or warn if concurrent writes. For v1, accept last-write-wins. |
| Camera doesn't support AbsoluteMove | Medium | Medium | `goto_preset` calls `OnvifPTZ.absolute_move()` which raises `OnvifError` on SOAP fault. Catch and return error to UI. |
| Preset positions are camera-specific | Low | Low | Presets are per-camera in config. No cross-camera confusion. |

---

## Feature 5: Clip Generation from Detections

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **CREATE** | `src/rtsp_warden/clips.py` | ClipGenerator for MP4 clip creation from HLS segments |
| **MODIFY** | `src/rtsp_warden/config.py` | Add `ClipConfig` to `AppConfig` |
| **MODIFY** | `src/rtsp_warden/db/models.py` | Add `Clip` ORM model |
| **MODIFY** | `src/rtsp_warden/db/schema.py` | Add `create_clip`, `get_clip`, `list_clips`, `delete_clip` functions |
| **CREATE** | `migrations/versions/0002_clips_table.py` | Alembic migration for clips table |
| **MODIFY** | `src/rtsp_warden/web/routes/events.py` | Add clip download route |
| **CREATE** | `src/rtsp_warden/web/routes/clips.py` | Clip serving routes |
| **MODIFY** | `src/rtsp_warden/web/templates/events/detail.html` | Add "Download Clip" button |
| **CREATE** | `tests/test_clips.py` | Unit tests for ClipGenerator and routes |

### 2. New Pydantic Config Types

In `config.py`:

```python
class ClipConfig(BaseModel):
    """Clip generation configuration.

    Controls the time window around an event used to generate
    downloadable MP4 clips from HLS segments.
    """

    enabled: bool = True
    pre_seconds: int = 10   # seconds before event to include
    post_seconds: int = 10  # seconds after event to include
    max_clip_age_days: int = 30  # auto-delete clips older than this
    output_dir: Path = Field(default_factory=lambda: Path("./workspace/clips"))

    @field_validator("pre_seconds", "post_seconds", "max_clip_age_days")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be > 0")
        return v


class AppConfig(BaseModel):
    cameras: list[CameraConfig]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    onvif: OnvifConfig = Field(default_factory=OnvifConfig)
    clips: ClipConfig = Field(default_factory=ClipConfig)  # NEW
```

### 3. New Runtime Deps

**No new deps.** ffmpeg is already a hard dependency. All clip generation uses `subprocess.run` with ffmpeg concat demuxer.

### 4. Public API Surface

```python
# src/rtsp_warden/clips.py

class ClipGenerator:
    """Generate downloadable MP4 clips from HLS segments around events.

    Given an event (with camera_name, created_at, recording_id),
    finds the HLS segments that overlap the time window
    [created_at - pre_seconds, created_at + post_seconds],
    concatenates them via ffmpeg concat demuxer, and produces
    a single MP4 file.

    Args:
        cfg: ClipConfig with pre/post seconds and output directory.
        recordings_dir: Base recordings directory (from RecordConfig.output_dir).
        ffmpeg_path: Path to ffmpeg binary.
    """

    def __init__(
        self,
        cfg: ClipConfig,
        recordings_dir: Path,
        ffmpeg_path: str = "ffmpeg",
    ) -> None: ...

    def find_segments(
        self,
        camera_name: str,
        stream: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Path]:
        """Find HLS .ts segments whose timestamps overlap [start_time, end_time].
        Segments are named %Y%m%d_%H%M%S.ts — parse filename to get start time,
        then check overlap with the requested window."""

    def generate(
        self,
        camera_name: str,
        stream: str,
        event_start: datetime,
        event_id: int,
        pre_seconds: int | None = None,
        post_seconds: int | None = None,
    ) -> Path:
        """Generate an MP4 clip for an event.

        Returns the path to the generated clip file.
        Raises ClipError if no segments found or ffmpeg fails."""

    def cleanup_old_clips(self) -> int:
        """Delete clips older than max_clip_age_days. Returns count deleted."""


class ClipError(Exception):
    """Raised when clip generation fails."""


# src/rtsp_warden/db/models.py — new model

class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_offset_s: Mapped[int] = mapped_column(Integer, nullable=False)  # pre_seconds used
    end_offset_s: Mapped[int] = mapped_column(Integer, nullable=False)    # post_seconds used
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    event: Mapped[Event] = relationship("Event")
    camera: Mapped[Camera | None] = relationship("Camera")


# src/rtsp_warden/db/schema.py — new functions

def create_clip(
    event_id: int,
    camera_id: int | None,
    path: str,
    size_bytes: int,
    start_offset_s: int,
    end_offset_s: int,
) -> Clip: ...

def get_clip(clip_id: int) -> Clip | None: ...

def list_clips(event_id: int | None = None, limit: int = 50) -> list[Clip]: ...

def delete_clip(clip_id: int) -> bool: ...
```

### 5. Test Strategy

**File:** `tests/test_clips.py`

| Test | What it verifies |
|------|-----------------|
| `test_find_segments_overlap` | Create fake segment files with known timestamps, verify correct segments found |
| `test_find_segments_no_overlap` | Event time outside all segments, verify empty list |
| `test_find_segments_partial_overlap` | Event window spans segment boundary, verify both included |
| `test_generate_creates_mp4` | Mock ffmpeg subprocess, verify output file path returned |
| `test_generate_no_segments` | No segments found, verify ClipError raised |
| `test_generate_ffmpeg_failure` | Mock ffmpeg non-zero exit, verify ClipError raised |
| `test_cleanup_old_clips` | Create old clip files, verify deleted |
| `test_clip_config_validation` | `ClipConfig(pre_seconds=0)` raises ValidationError |
| `test_db_create_clip` | Create Clip row, verify fields |
| `test_db_get_clip` | Get by ID, verify returned |
| `test_db_list_clips` | List by event_id, verify filtered |
| `test_db_delete_clip` | Delete, verify returns True |
| `test_web_generate_clip` | POST /events/{id}/clip, verify returns clip path |
| `test_web_download_clip` | GET /clips/{id}/download, verify file response |
| `test_event_detail_shows_clip_button` | GET /events/{id}, verify "Download Clip" button present |

Use `unittest.mock.patch("subprocess.run")` for ffmpeg mocking. Use temporary directories for segment files.

### 6. Web UI Changes

**New routes:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/events/{event_id}/clip` | user | Generate clip for event. Returns JSON `{clip_id, path, size_bytes}` |
| `GET` | `/clips/{clip_id}/download` | user | Stream the MP4 file as a download |
| `GET` | `/clips` | user | List generated clips (optional admin page) |

**Modified template:** `events/detail.html` — Add below the metadata section:

```html
<article>
  <header>Clip</header>
  <button id="generate-clip-btn"
          hx-post="/events/{{ event.id }}/clip"
          hx-swap="outerHTML">
    Generate Clip
  </button>
  <div id="clip-result"></div>
</article>
```

The button triggers clip generation. On success, the response replaces the button with a download link. Use htmx `hx-swap` to show progress/result.

### 7. Migration Needs

**Yes — Alembic migration required.**

Create `migrations/versions/0002_clips_table.py`:

```python
"""add clips table

Revision ID: 0002_clips
Revises: 0001_initial
Create Date: 2026-06-16 00:00:00.000000
"""

def upgrade() -> None:
    op.create_table(
        "clips",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=True),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("start_offset_s", sa.Integer(), nullable=False),
        sa.Column("end_offset_s", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_clips_event_id", "clips", ["event_id"])

def downgrade() -> None:
    op.drop_table("clips")
```

### 8. Config Example

```yaml
clips:
  enabled: true
  pre_seconds: 10
  post_seconds: 10
  max_clip_age_days: 30
  output_dir: ./workspace/clips
```

### 9. README Updates

- **New "Clips" section** after Recording/Events: Document clip generation feature. Explain the time window, the "Generate Clip" button on event detail pages, and the download endpoint. Note that clips are MP4 files concatenated from HLS segments via ffmpeg.
- **Config reference:** Add `clips` section to the config documentation.
- **Web UI routes table:** Add the 3 new clip routes.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| No HLS segments for event time window | Medium | Medium | `find_segments` returns empty list → `generate` raises `ClipError` with clear message. UI shows "No recording available for this event." |
| ffmpeg concat fails (codec mismatch) | Low | Medium | Catch non-zero exit, log stderr, raise `ClipError`. Segments are all from same camera/stream, so codec mismatch is unlikely. |
| Large clips fill disk | Medium | Medium | `max_clip_age_days` auto-cleanup. `ClipGenerator.cleanup_old_clips()` called periodically (or on new clip generation). |
| Concurrent clip generation | Low | Low | Each clip has unique filename (`{camera}_{event_id}_{timestamp}.mp4`). No collision. |
| Event has no camera_id | Low | Low | `create_event` sets camera_id by looking up camera name. If camera is deleted, camera_id becomes NULL. Clip generation still works (uses camera_name from event metadata). |

---

## Parallelization Strategy

### Independence Analysis

All 5 features touch **different packages** with **zero code-level dependencies** on each other:

| Feature | Primary package | Touches |
|---------|----------------|---------|
| 1. Apprise | `alerts/` | `alerts/apprise.py`, `config.py`, `factory.py`, web templates |
| 2. YOLO/DNN | `detectors/builtin/` | `dnn.py`, `model_utils.py`, `registry.py`, bundled model files |
| 3. ONVIF events | `onvif/` | `events.py`, `config.py`, web routes, template |
| 4. PTZ presets | `onvif/` + `config.py` | `presets.py`, `config.py`, web routes, template |
| 5. Clip generation | `clips.py` + `db/` | `clips.py`, `config.py`, `db/models.py`, `db/schema.py`, migration, web routes, template |

**Shared files** (potential merge conflicts):
- `config.py` — All 5 features add config types. This is the only conflict point.
- `src/rtsp_warden/web/templates/onvif/index.html` — Features 3 and 4 both modify this template.

### Recommended Dispatch Order

**Phase 1 — Parallel (5 coder sub-agents simultaneously):**

| Order | Feature | Agent | Notes |
|-------|---------|-------|-------|
| 1a | Apprise alerts | boomerang-coder | Independent. Only touches alerts/ + config.py |
| 1b | YOLO/DNN detector | boomerang-coder | Independent. Only touches detectors/ + registry.py |
| 1c | ONVIF events | boomerang-coder | Independent. Only touches onvif/ + config.py |
| 1d | PTZ presets | boomerang-coder | Independent. Touches onvif/ + config.py + template |
| 1e | Clip generation | boomerang-coder | Independent. Touches clips.py + db/ + config.py + migration |

**Phase 2 — Integration (sequential, after all 5 land):**

| Order | Task | Agent | Notes |
|-------|------|-------|-------|
| 2a | Resolve config.py merge conflicts | boomerang-coder | All 5 features add to config.py. Manual merge needed. |
| 2b | Resolve onvif/index.html merge | boomerang-coder | Features 3 and 4 both modify this template. |
| 2c | Run full test suite | boomerang-tester | All 325 existing + new tests. Target: 0 failures. |
| 2d | Lint all new code | boomerang-linter | Ruff check. Target: 0 errors. |
| 2e | Update README | boomerang-writer | Consolidate all 5 feature docs into README. |
| 2f | Git commit per feature | boomerang-git | One commit per feature, then integration commit. |

### Context Package Template

Each coder sub-agent receives:

```
## Task: [Feature Name] for rtsp-warden v1.1.0

### Baseline
- v1.0.0, 325 tests passing, 17 runtime deps
- Project root: /home/jcharles/Projects/python/rtsp-warden_v0.2.0/

### Files to Create
[List from plan]

### Files to Modify
[List from plan with specific changes]

### Constraints
- No emoji in code or output
- All public functions need type hints
- All new modules need docstrings
- Stdlib-first; deps only when value is real
- Additive layers — don't destabilize ingest/recording
- Notifier Protocol and Detector Protocol are stable — extend, don't break
- Pytest: asyncio_mode = "auto", tests in tests/
- Use uv for package management
- Config: pydantic v2 models in src/rtsp_warden/config.py
- Web UI: FastAPI + Jinja + htmx + Alpine.js + Pico CSS. No React. No build step.
- Alembic for migrations

### Reference Files to Read
[List of existing files the coder should study]

### Expected Output
- All new files created
- All modifications applied
- Tests passing (run pytest on new test file)
- Return: summary of what was done, any issues encountered
```

---

## Summary: Files Changed Across All 5 Features

### New Files (10)
1. `src/rtsp_warden/alerts/apprise.py`
2. `src/rtsp_warden/detectors/builtin/dnn.py`
3. `src/rtsp_warden/detectors/builtin/model_utils.py`
4. `src/rtsp_warden/detectors/builtin/models/yolov4-tiny.cfg` (bundled)
5. `src/rtsp_warden/onvif/events.py`
6. `src/rtsp_warden/onvif/presets.py`
7. `src/rtsp_warden/clips.py`
8. `src/rtsp_warden/web/routes/clips.py`
9. `migrations/versions/0002_clips_table.py`
10. `tests/test_apprise_notifier.py`
11. `tests/test_dnn_detector.py`
12. `tests/test_model_utils.py`
13. `tests/test_onvif_events.py`
14. `tests/test_ptz_presets.py`
15. `tests/test_clips.py`

### Modified Files (11)
1. `src/rtsp_warden/config.py` — 5 new/extended config types
2. `src/rtsp_warden/alerts/factory.py` — apprise branch
3. `src/rtsp_warden/detectors/registry.py` — dnn type + branch
4. `src/rtsp_warden/db/models.py` — Clip model
5. `src/rtsp_warden/db/schema.py` — clip CRUD functions
6. `src/rtsp_warden/web/routes/onvif.py` — preset routes + dynamic PTZ
7. `src/rtsp_warden/web/routes/events.py` — clip generation route
8. `src/rtsp_warden/web/templates/onvif/index.html` — events status + per-camera PTZ + presets
9. `src/rtsp_warden/web/templates/events/detail.html` — clip button
10. `src/rtsp_warden/web/templates/alerts/new.html` — apprise fields
11. `src/rtsp_warden/web/templates/alerts/edit.html` — apprise fields
12. `src/rtsp_warden/web/templates/alerts/list.html` — apprise display
13. `pyproject.toml` — apprise dep
14. `README.md` — 5 feature docs

### New Runtime Deps
- `apprise>=1.9.0` (Feature 1 only)

### New Test Files
- 5 new test files, estimated ~50-70 new test cases

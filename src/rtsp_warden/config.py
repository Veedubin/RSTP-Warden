from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from datetime import time
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .deprecations import warn_once as _warn_once
from .detectors.model_registry import (
    ModelError,
    camera_model_labels,
    default_models_dir,
    unknown_labels_message,
)
from .detectors.registry import DetectorSpec

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(obj: Any, env: Mapping[str, str] | None = None) -> Any:
    """Replace ``${NAME}`` references in every string of a loaded YAML tree.

    Only the exact ``${NAME}`` form is replaced; a bare ``$`` or ``$NAME`` is
    left alone so passwords containing ``$`` survive. A reference to a
    variable that is not set is a fatal config error.
    """
    source: Mapping[str, str] = os.environ if env is None else env

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in source:
            raise SystemExit(
                f"Config references ${{{name}}} but the environment variable {name} is not set"
            )
        return source[name]

    if isinstance(obj, str):
        return _ENV_REF.sub(_sub, obj)
    if isinstance(obj, list):
        return [expand_env(v, source) for v in obj]
    if isinstance(obj, dict):
        return {k: expand_env(v, source) for k, v in obj.items()}
    return obj


RtspTransport = Literal["tcp", "udp"]
Container = Literal[
    "ts", "mkv", "mp4"
]  # ts is the NVR-grade default; mkv/mp4 retained for backward compat and special cases
ProxyMode = Literal["mjpeg", "rtsp"]
ProxyStream = Literal["main", "sub"]


class StreamRecordConfig(BaseModel):
    enabled: bool = True
    container: Container = "ts"
    chunk_seconds: int = 300
    rtsp_transport: RtspTransport = "tcp"

    # If you ever need to re-encode (not default), this can be extended later.
    mode: Literal["copy"] = "copy"

    @field_validator("chunk_seconds")
    @classmethod
    def _chunk_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_seconds must be > 0")
        return v


class RetentionConfig(BaseModel):
    """Delete old files and/or cap total size per camera (``<record.output_dir>/<camera>``).

    Notes:
      * keep_last_n protects the newest N recorded segments (files in ``main/`` and ``sub/``);
        event thumbnails and clips never count toward it.
      * max_days and max_gb apply to every unprotected file: segments, thumbnails and clips.
    """

    max_days: int | None = None
    max_gb: float | None = None
    keep_last_n: int = 0
    cleanup_interval_seconds: int = 300

    @field_validator("max_days")
    @classmethod
    def _days_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_days must be > 0 (or omitted)")
        return v

    @field_validator("max_gb")
    @classmethod
    def _gb_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("max_gb must be > 0 (or omitted)")
        return v

    @field_validator("cleanup_interval_seconds")
    @classmethod
    def _interval_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("cleanup_interval_seconds must be > 0")
        return v


class RecordConfig(BaseModel):
    enabled: bool = True

    # Base directory; actual layout becomes:
    #   {output_dir}/{camera}/{stream}/{camera}_{stream}_%Y%m%d_%H%M%S.{container}
    output_dir: Path = Field(default_factory=lambda: Path("./recordings"))

    main: StreamRecordConfig = Field(
        default_factory=lambda: StreamRecordConfig(container="ts", chunk_seconds=300)
    )
    sub: StreamRecordConfig = Field(
        default_factory=lambda: StreamRecordConfig(container="ts", chunk_seconds=300)
    )

    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    # Sprint 4 additions — all backward compatible (default to existing behavior)

    # Audio: when True, include audio track in recordings (requires camera
    # to have audio in its RTSP stream; falls back to silent if absent).
    audio: bool = False

    # Recording mode: "continuous" (default) records 24/7. "event" only
    # records around detector events (see event_record below).
    mode: Literal["continuous", "event"] = "continuous"

    # Event-mode config: pre/post seconds of buffer to keep around events.
    event_record: EventRecordConfig = Field(default_factory=lambda: EventRecordConfig())


class ProxyConfig(BaseModel):
    enabled: bool = True
    mode: ProxyMode = "mjpeg"
    stream: ProxyStream = "sub"

    bind_host: str = "0.0.0.0"
    port: int = 9001

    # RTSP (MediaMTX)
    path: str = "live"
    source_on_demand: bool = True

    # MJPEG-over-HTTP
    fps: int = 7
    scale_width: int = 0  # 0 disables scaling

    @field_validator("port")
    @classmethod
    def _port_valid(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("port must be between 1 and 65535")
        return v

    @field_validator("fps")
    @classmethod
    def _fps_valid(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("fps must be > 0")
        return v

    @field_validator("scale_width")
    @classmethod
    def _scale_valid(cls, v: int) -> int:
        if v < 0:
            raise ValueError("scale_width must be >= 0")
        return v


class OnvifEventConfig(BaseModel):
    """Per-camera ONVIF event subscription configuration.

    Attributes:
        type: Event type filter -- "motion", "tamper", or "all".
        min_interval_seconds: Minimum seconds between firing callbacks
            for the same event type on this camera (debounce).
    """

    type: Literal["motion", "tamper", "all"] = "all"
    min_interval_seconds: float = 30.0

    @field_validator("min_interval_seconds")
    @classmethod
    def _min_interval_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("min_interval_seconds must be > 0")
        return v


class PTZPresetConfig(BaseModel):
    """A named PTZ preset position for a camera.

    Stored in CameraConfig.presets as a list. Each preset has a
    human-readable name and normalized pan/tilt/zoom values.

    Attributes:
        name: Preset label (must be non-empty, max 64 chars).
        pan: Pan position, -1.0 to 1.0.
        tilt: Tilt position, -1.0 to 1.0.
        zoom: Zoom level, 0.0 to 1.0.
    """

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
        if len(v2) > 64:
            raise ValueError("preset name must be 64 characters or fewer")
        return v2

    @field_validator("pan", "tilt")
    @classmethod
    def _pan_tilt_range(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError("pan/tilt must be between -1.0 and 1.0")
        return v

    @field_validator("zoom")
    @classmethod
    def _zoom_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("zoom must be between 0.0 and 1.0")
        return v


class GridZoneConfig(BaseModel):
    """Grid-based detection zone for a camera.

    Divides the frame into an NxM grid of cells; cells NOT in blocked_cells
    are active. What the cells mean depends on ``kind``:

    - ``ignore`` (default): detections whose bounding-box center falls within
      a blocked cell are discarded; detections in active cells are kept.
    - ``area``: nothing is discarded. The active cells form a named area; an
      event's ``zone`` is the first area (config order) containing the center
      of its best box, and ``rules[].zones`` refer to areas by name.

    Cells are fractions of the frame, so a zone drawn on a full-size snapshot
    applies unchanged to the smaller detection frame.

    Attributes:
        name: Zone label, unique per camera, e.g. "exclude road" (1-64
            characters after stripping, no "/").
        kind: "ignore" or "area".
        grid_cols: Number of columns in the grid (2-64).
        grid_rows: Number of rows in the grid (2-64).
        blocked_cells: Set of (col, row) tuples. Cells NOT in this set are active.
        frame_width: Snapshot resolution width at time of save.
        frame_height: Snapshot resolution height at time of save.
        enabled: Whether this zone is active.
    """

    name: str
    kind: Literal["ignore", "area"] = "ignore"
    grid_cols: int = 16
    grid_rows: int = 16
    blocked_cells: set[tuple[int, int]] = set()
    frame_width: int
    frame_height: int
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _zone_name(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("zone name must not be empty")
        if len(v2) > 64:
            raise ValueError("zone name must be at most 64 characters")
        if "/" in v2:
            raise ValueError("zone name must not contain '/'")
        return v2

    @field_validator("grid_cols", "grid_rows")
    @classmethod
    def _grid_bounds(cls, v: int) -> int:
        if v < 2:
            raise ValueError("grid_cols and grid_rows must be >= 2")
        if v > 64:
            raise ValueError("grid_cols and grid_rows must be <= 64")
        return v

    @model_validator(mode="after")
    def _validate_blocked_cells(self) -> GridZoneConfig:
        """Validate that all blocked_cells are within grid dimensions."""
        for col, row in self.blocked_cells:
            if not (0 <= col < self.grid_cols) or not (0 <= row < self.grid_rows):
                raise ValueError(
                    f"blocked cell ({col},{row}) out of bounds for "
                    f"{self.grid_cols}x{self.grid_rows} grid"
                )
        return self


def validate_camera_zones(zones: list[GridZoneConfig], rules: list[RuleConfig]) -> None:
    """Check one camera's zones against each other and against its rules.

    Zone names are unique per camera, and every name in ``rules[].zones`` is an
    ``area`` zone of the same camera (enabled or not: a disabled area never
    matches). Raises ValueError naming the zone (and rule) at fault. Used by
    ``CameraConfig`` at load and by the zone routes before they write back.
    """
    seen: set[str] = set()
    for zone in zones:
        if zone.name in seen:
            raise ValueError(f"zone name {zone.name!r} is used more than once")
        seen.add(zone.name)
    kinds = {zone.name: zone.kind for zone in zones}
    areas = ", ".join(repr(zone.name) for zone in zones if zone.kind == "area") or "none"
    for rule in rules:
        for name in rule.zones:
            if name not in kinds:
                raise ValueError(
                    f"rule {rule.name!r} names zone {name!r}, which is not a zone of "
                    f"this camera (area zones: {areas})"
                )
            if kinds[name] != "area":
                raise ValueError(
                    f"rule {rule.name!r} names zone {name!r}, which is an ignore zone; "
                    "rules can only name area zones"
                )


# --- Detection and automation (RW-3) ---

DETECT_FPS_MIN = 0.5
DETECT_FPS_MAX = 30.0
TAP_MIN_WIDTH = 320  # frame tap width for motion-only cameras (spec 5.3)

_BETWEEN_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


def parse_between(spec: str) -> tuple[time, time]:
    """Parse a rule window "HH:MM-HH:MM" (local time) into (start, end).

    The window may wrap midnight ("22:00-06:00"). Raises ValueError for a
    malformed value, an hour above 23, a minute above 59, or start == end.
    """
    m = _BETWEEN_RE.match(spec)
    if m is None:
        raise ValueError(f"between must look like 'HH:MM-HH:MM', got {spec!r}")
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    if h1 > 23 or h2 > 23 or m1 > 59 or m2 > 59:
        raise ValueError(f"between needs hours 00-23 and minutes 00-59, got {spec!r}")
    start, end = time(h1, m1), time(h2, m2)
    if start == end:
        raise ValueError(f"between start and end are the same time: {spec!r}")
    return start, end


class RuleConfig(BaseModel):
    """A per-camera rule: which events fire which actions (spec 8.2).

    An event matches when its label is in `labels` (empty = any), its zone is in
    `zones` (empty = any), its confidence is >= `min_confidence`, and the local
    time is inside `between` when set. `actions` names entries of the top-level
    `actions` list. Unknown keys are an error so a typo such as `lables:` cannot
    silently turn a rule into "match everything".
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str
    labels: list[str] = Field(default_factory=list)
    zones: list[str] = Field(default_factory=list)
    min_confidence: float = 0.0
    between: str | None = None  # "HH:MM-HH:MM" local time, may wrap midnight
    cooldown_seconds: float = 60.0
    clip: bool = False
    actions: list[str]

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("rule name must be non-empty")
        if len(v2) > 64:
            raise ValueError("rule name must be at most 64 characters")
        return v2

    @field_validator("labels", "zones", "actions")
    @classmethod
    def _names_nonempty(cls, v: list[str]) -> list[str]:
        out = [item.strip() for item in v]
        if any(not item for item in out):
            raise ValueError("list entries must be non-empty names")
        return out

    @field_validator("min_confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        return v

    @field_validator("between", mode="before")
    @classmethod
    def _between_format(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("between must be a quoted string like '22:00-06:00'")
        if not v.strip():
            return None
        start, end = parse_between(v)
        return f"{start:%H:%M}-{end:%H:%M}"

    @field_validator("cooldown_seconds")
    @classmethod
    def _cooldown_non_negative(cls, v: float) -> float:
        if not v >= 0:
            raise ValueError("cooldown_seconds must be >= 0")
        return v


class CameraConfig(BaseModel):
    # Errors name the camera and zone in their text; never echo the input (it holds URLs).
    model_config = {"hide_input_in_errors": True}

    name: str
    main_url: str
    sub_url: str | None = None  # optional; every sub-stream consumer falls back to main

    record: RecordConfig = Field(default_factory=RecordConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    detectors: list[DetectorSpec] = Field(default_factory=list)
    events: list[OnvifEventConfig] = Field(default_factory=list)
    presets: list[PTZPresetConfig] = Field(default_factory=list)

    # Sprint 6 additions
    retention: RetentionConfig | None = None  # per-camera override
    zones: list[GridZoneConfig] = Field(default_factory=list)
    sensitivity: float = 50.0  # 0-100 scale, used for ALL detectors
    detect_classes: list[str] | None = None  # intersection with detector's allowed_classes

    # Detection and automation (RW-3)
    detect_fps: float = 5.0  # frame tap rate; a change restarts the camera's ingest
    track_grace_seconds: float = 3.0  # an unmatched track closes after this long
    min_track_frames: int = 2  # matched frames before a track becomes an event
    rules: list[RuleConfig] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("name must be non-empty")
        return v2

    @field_validator("sensitivity")
    @classmethod
    def _sensitivity_range(cls, v: float) -> float:
        if not 0.0 <= v <= 100.0:
            raise ValueError("sensitivity must be between 0.0 and 100.0")
        return v

    @field_validator("detect_fps")
    @classmethod
    def _detect_fps_range(cls, v: float) -> float:
        if not DETECT_FPS_MIN <= v <= DETECT_FPS_MAX:
            raise ValueError(
                f"detect_fps must be between {DETECT_FPS_MIN:g} and {DETECT_FPS_MAX:g}"
            )
        return v

    @field_validator("track_grace_seconds")
    @classmethod
    def _grace_positive(cls, v: float) -> float:
        if not v > 0:
            raise ValueError("track_grace_seconds must be > 0")
        return v

    @field_validator("min_track_frames")
    @classmethod
    def _min_frames_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_track_frames must be >= 1")
        return v

    @model_validator(mode="after")
    def _fallback_proxy_stream(self) -> CameraConfig:
        """Without a sub stream, the proxy (and frame tap) read the main stream."""
        if self.sub_url is None and self.proxy.stream == "sub":
            self.proxy.stream = "main"
        return self

    @model_validator(mode="after")
    def _check_zones(self) -> CameraConfig:
        """Zone names are unique per camera; rules[].zones name this camera's area zones."""
        try:
            validate_camera_zones(self.zones, self.rules)
        except ValueError as exc:
            raise ValueError(f"camera {self.name!r}: {exc}") from None
        return self

    @model_validator(mode="after")
    def _detector_rates(self) -> CameraConfig:
        """A detector cannot run faster than the tap: every spec fps <= detect_fps.

        An fps that came from the deprecated interval_seconds is clamped with a
        warning instead, so a config that loaded before keeps loading.
        """
        for i, spec in enumerate(self.detectors):
            if spec.fps is None or spec.fps <= self.detect_fps:
                continue
            if spec._fps_from_interval:
                _warn_once(
                    f"interval-clamp:{self.name}:{i}",
                    "camera %r detectors[%d] (%s): interval_seconds gives fps %g, above "
                    "detect_fps %g; using fps %g",
                    self.name,
                    i,
                    spec.type,
                    spec.fps,
                    self.detect_fps,
                    self.detect_fps,
                )
                spec.fps = self.detect_fps
                continue
            raise ValueError(
                f"detectors[{i}] ({spec.type}): fps {spec.fps:g} is above the camera's "
                f"detect_fps {self.detect_fps:g}; lower fps or raise detect_fps"
            )
        return self

    @model_validator(mode="after")
    def _rules_valid(self) -> CameraConfig:
        """Rule names are unique per camera; a clip rule needs recording on."""
        seen: set[str] = set()
        for rule in self.rules:
            if rule.name in seen:
                raise ValueError(f"duplicate rule name {rule.name!r}")
            seen.add(rule.name)
            if rule.clip and not self.record.enabled:
                _warn_once(
                    f"clip-without-recording:{self.name}:{rule.name}",
                    "camera %r rule %r has clip: true but record.enabled is false; "
                    "no clip will be made",
                    self.name,
                    rule.name,
                )
        return self

    def effective_fps(self, spec: DetectorSpec) -> float:
        """The rate a detector runs at: its own fps, else detect_fps, never above detect_fps."""
        if spec.fps is None:
            return float(self.detect_fps)
        return float(min(spec.fps, self.detect_fps))

    def motion_events_enabled(self, spec: DetectorSpec) -> bool:
        """Whether a motion spec writes events rows (spec 7.4, 11).

        An explicit `events` wins. Unset means yes, unless the camera has an
        enabled `onnx` detector.
        """
        if spec.events is not None:
            return spec.events
        return not any(s.type == "onnx" and s.enabled for s in self.detectors)


def _no_input_width(spec: DetectorSpec) -> int | None:
    return None


def compute_tap_settings(
    cam: CameraConfig,
    input_width_for: Callable[[DetectorSpec], int | None] = _no_input_width,
) -> tuple[float, int]:
    """Return the frame tap's (fps, scale width) for a camera (spec 5.3).

    fps is the camera's detect_fps. The width is the largest model input width
    among the camera's enabled detectors, as reported by `input_width_for`
    (None for a detector without a model input), and never below TAP_MIN_WIDTH.
    """
    width = TAP_MIN_WIDTH
    for spec in cam.detectors:
        if not spec.enabled:
            continue
        w = input_width_for(spec)
        if w is not None and w > width:
            width = int(w)
    return float(cam.detect_fps), width


# --- Sprint 4: event recording + alerts + ONVIF config types ---


class EventRecordConfig(BaseModel):
    """Event-only recording config.

    When RecordConfig.mode == "event", the recorder starts a new segment
    `pre_seconds` before the first event in a quiet period, and stops
    `post_seconds` after the last event. This is the "motion-buffered"
    NVR style.
    """

    pre_seconds: int = 5
    post_seconds: int = 10
    min_segment_seconds: int = 10  # don't create a segment shorter than this
    max_segment_seconds: int = 600  # safety cap to avoid unbounded growth

    @field_validator("pre_seconds", "post_seconds", "min_segment_seconds", "max_segment_seconds")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be > 0")
        return v

    @field_validator("max_segment_seconds")
    @classmethod
    def _max_gt_min(cls, v: int, info) -> int:
        # Note: pydantic v2 passes ValidationInfo; we use it best-effort
        return v


class NtifySpec(BaseModel):
    """Configuration for an ntfy.sh notifier."""

    name: str
    type: Literal["ntfy"]
    url: str
    enabled: bool = True

    topic: str | None = None
    token: str | None = None
    priority: int | None = None

    min_interval_seconds: int = 30
    severities: list[Literal["info", "warn", "error"]] = Field(
        default_factory=lambda: ["warn", "error"]
    )

    @field_validator("url")
    @classmethod
    def _url_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("url must be non-empty")
        return v2


class WebhookSpec(BaseModel):
    """Configuration for a generic webhook notifier."""

    name: str
    type: Literal["webhook"]
    url: str
    enabled: bool = True

    headers: dict[str, str] = Field(default_factory=dict)
    method: Literal["POST", "PUT"] = "POST"

    min_interval_seconds: int = 30
    severities: list[Literal["info", "warn", "error"]] = Field(
        default_factory=lambda: ["warn", "error"]
    )

    @field_validator("url")
    @classmethod
    def _url_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("url must be non-empty")
        return v2


# Severity level mapping for AppriseSpec.min_severity
_SEVERITY_ORDER: dict[str, list[str]] = {
    "info": ["info", "warn", "error"],
    "warn": ["warn", "error"],
    "error": ["error"],
}


class AppriseSpec(BaseModel):
    """Configuration for an Apprise notifier (email + 90+ services).

    urls: list of Apprise URL strings (e.g. "mailto://user:pass@smtp.gmail.com:587").
    title_template: optional Python format string for notification titles.
        Available keys: {camera_name}, {event_type}, {severity}.
        Defaults to "[{severity}] {camera_name}: {event_type}" if not set.
    min_severity: minimum severity level that triggers this notifier.
        "info" -> all, "warn" -> warn+error, "error" -> error only.
    """

    name: str
    type: Literal["apprise"]
    urls: list[str]
    enabled: bool = True
    title_template: str | None = None
    min_interval_seconds: float = 60.0
    min_severity: str = "info"

    model_config = {"extra": "forbid"}

    @field_validator("urls")
    @classmethod
    def _urls_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("urls must contain at least one Apprise URL")
        cleaned = [u.strip() for u in v if u.strip()]
        if not cleaned:
            raise ValueError("urls must contain at least one non-empty Apprise URL")
        return cleaned

    @field_validator("min_severity")
    @classmethod
    def _min_severity_valid(cls, v: str) -> str:
        if v not in _SEVERITY_ORDER:
            raise ValueError(f"min_severity must be one of {list(_SEVERITY_ORDER)}")
        return v

    @property
    def severities(self) -> list[str]:
        """Derive severities list from min_severity for AlertManager compatibility."""
        return _SEVERITY_ORDER[self.min_severity]


NotifierSpec = Annotated[NtifySpec | WebhookSpec | AppriseSpec, Field(discriminator="type")]


class AlertsConfig(BaseModel):
    """Top-level alerts config.

    notifiers: list of notifier destinations
    enabled: master switch (can also be controlled via CLI --alerts/--no-alerts)
    """

    enabled: bool = True
    notifiers: list[NotifierSpec] = Field(default_factory=list)


# --- Actions (spec 8.1): replaces alerts.notifiers ---


class _ActionSpecBase(BaseModel):
    """Fields and checks shared by every action type.

    ``extra="forbid"`` catches typos such as ``tpoic:``. ``hide_input_in_errors``
    keeps tokens and credential-bearing URLs out of validation messages.
    """

    model_config = {"extra": "forbid", "hide_input_in_errors": True}

    name: str

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("action name must be non-empty")
        if len(v2) > 64:
            raise ValueError("action name must be at most 64 characters")
        if "/" in v2:
            raise ValueError("action name must not contain '/'")
        return v2


class NtfyActionSpec(_ActionSpecBase):
    """Publish to an ntfy server: ``POST {url}/{topic}`` (or ``{url}`` without a topic)."""

    type: Literal["ntfy"]
    url: str
    topic: str | None = None
    token: str | None = None
    priority: int | None = None

    @field_validator("url")
    @classmethod
    def _url_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("url must be non-empty")
        return v2

    @field_validator("priority")
    @classmethod
    def _priority_range(cls, v: int | None) -> int | None:
        if v is not None and not 1 <= v <= 5:
            raise ValueError("priority must be between 1 and 5")
        return v


class WebhookActionSpec(_ActionSpecBase):
    """Send the event payload as JSON to ``url``."""

    type: Literal["webhook"]
    url: str
    method: Literal["POST", "PUT"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def _url_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("url must be non-empty")
        return v2

    @field_validator("headers")
    @classmethod
    def _headers_ascii(cls, v: dict[str, str]) -> dict[str, str]:
        for key, value in v.items():
            if not (key.isascii() and value.isascii()):
                raise ValueError(f"header {key!r} must be ASCII (name and value)")
        return v


class AppriseActionSpec(_ActionSpecBase):
    """Deliver through Apprise URLs (mailto://, tgram://, discord://, mqtt://, ...)."""

    type: Literal["apprise"]
    urls: list[str]

    @field_validator("urls")
    @classmethod
    def _urls_nonempty(cls, v: list[str]) -> list[str]:
        cleaned = [u.strip() for u in v if u.strip()]
        if not cleaned:
            raise ValueError("urls must contain at least one non-empty Apprise URL")
        return cleaned


ActionSpec = Annotated[
    NtfyActionSpec | WebhookActionSpec | AppriseActionSpec, Field(discriminator="type")
]

# Legacy notifier keys that have no meaning for actions (named in the deprecation warning).
_LEGACY_IGNORED_KEYS = ("severities", "min_interval_seconds", "min_severity", "title_template")


def _legacy_to_action(
    legacy: NtifySpec | WebhookSpec | AppriseSpec,
) -> NtfyActionSpec | WebhookActionSpec | AppriseActionSpec:
    """Project a validated legacy notifier onto the matching action model."""
    target: type[NtfyActionSpec] | type[WebhookActionSpec] | type[AppriseActionSpec]
    if isinstance(legacy, NtifySpec):
        target = NtfyActionSpec
    elif isinstance(legacy, WebhookSpec):
        target = WebhookActionSpec
    else:
        target = AppriseActionSpec
    data = legacy.model_dump(include=set(target.model_fields))
    try:
        return target.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        prefix = f"{where}: " if where else ""
        raise ValueError(
            f"alerts.notifiers entry {legacy.name!r} cannot become an action: "
            f"{prefix}{first['msg']}"
        ) from None


def _duplicate_names(names: list[str]) -> list[str]:
    """Names that occur more than once, in first-repeat order."""
    seen: set[str] = set()
    dups: list[str] = []
    for name in names:
        if name in seen and name not in dups:
            dups.append(name)
        seen.add(name)
    return dups


class OnvifConfig(BaseModel):
    """ONVIF camera discovery + PTZ + events config.

    discovery_enabled: enable WS-Discovery (UDP multicast on 239.255.255.250:3702)
    ptz_enabled: enable PTZ control surface (requires ONVIF device service on camera)
    events_enabled: enable ONVIF event subscriptions (PullPoint polling)
    discovery_timeout_seconds: how long to wait for probe responses
    ptz_timeout_seconds: per-PTZ-command HTTP timeout
    events_poll_interval_seconds: seconds between PullMessages calls
    username/password: optional default credentials used when not overridden per-camera
    """

    discovery_enabled: bool = False  # opt-in (network noise on default networks)
    ptz_enabled: bool = False
    events_enabled: bool = False
    discovery_timeout_seconds: int = 5
    ptz_timeout_seconds: int = 10
    events_poll_interval_seconds: int = 10
    username: str | None = None
    password: str | None = None

    @field_validator(
        "discovery_timeout_seconds", "ptz_timeout_seconds", "events_poll_interval_seconds"
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be > 0")
        return v


class RuntimeConfig(BaseModel):
    ffmpeg_path: str = "ffmpeg"
    mediamtx_path: str = "mediamtx"

    ffmpeg_loglevel: str = "warning"
    workspace_dir: Path = Field(default_factory=lambda: Path("./workspace"))

    auto_restart: bool = True
    restart_backoff_min_s: float = 1.0
    restart_backoff_max_s: float = 60.0
    restart_backoff_factor: float = 2.0
    stderr_tail_lines: int = 200

    status_interval_s: float = 15.0

    # ONNX model files and user model directories (<models_dir>/<name>/model.yaml).
    models_dir: Path = Field(default_factory=default_models_dir)

    @field_validator("models_dir")
    @classmethod
    def _expand_models_dir(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator(
        "restart_backoff_min_s",
        "restart_backoff_max_s",
        "restart_backoff_factor",
        "status_interval_s",
    )
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("value must be > 0")
        return v

    @field_validator("stderr_tail_lines")
    @classmethod
    def _stderr_tail(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("stderr_tail_lines must be > 0")
        return v


class ClipsConfig(BaseModel):
    """Clip generation configuration.

    Controls the time window around an event used to generate
    downloadable MP4 clips from HLS segments.
    """

    enabled: bool = True
    pre_seconds: float = 10.0
    post_seconds: float = 10.0
    output_dir: str = "{recordings_root}/../clips"
    max_duration: float = 120.0

    @field_validator("pre_seconds", "post_seconds", "max_duration")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("value must be > 0")
        return v


class AppConfig(BaseModel):
    # Validation errors never echo the input: it holds env-expanded camera URLs.
    model_config = {"hide_input_in_errors": True}

    cameras: list[CameraConfig]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    actions: list[ActionSpec] = Field(default_factory=list)
    onvif: OnvifConfig = Field(default_factory=OnvifConfig)
    clips: ClipsConfig = Field(default_factory=ClipsConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)  # global fallback

    @model_validator(mode="after")
    def _migrate_legacy_notifiers(self) -> AppConfig:
        """Map deprecated ``alerts.notifiers`` onto ``actions`` (one release only).

        Runs only when the list is non-empty: every dump carries an empty
        ``alerts`` block. ``enabled: true`` entries move to ``actions``;
        ``enabled: false`` entries stay where they are; ``alerts.enabled`` is
        ignored. A fresh ``AlertsConfig`` is assigned so a caller's instance is
        never mutated. The warning goes through ``_warn_once``: once per process.
        """
        if not self.alerts.notifiers:
            return self
        moved: list[NtfyActionSpec | WebhookActionSpec | AppriseActionSpec] = []
        kept: list[NtifySpec | WebhookSpec | AppriseSpec] = []
        ignored: list[str] = []
        for legacy in self.alerts.notifiers:
            if not legacy.enabled:
                kept.append(legacy)
                continue
            moved.append(_legacy_to_action(legacy))
            ignored.extend(
                f"{legacy.name}.{key}"
                for key in _LEGACY_IGNORED_KEYS
                if key in legacy.model_fields_set
            )
        names = [a.name for a in self.actions] + [a.name for a in moved] + [n.name for n in kept]
        dups = _duplicate_names(names)
        if dups:
            raise ValueError(
                f"action name {dups[0]!r} is defined more than once across actions and "
                "alerts.notifiers; action names must be unique"
            )
        self.actions = [*self.actions, *moved]
        self.alerts = AlertsConfig(enabled=self.alerts.enabled, notifiers=kept)
        parts: list[str] = []
        if moved:
            parts.append("moved to actions: " + ", ".join(a.name for a in moved))
        if ignored:
            parts.append(
                "ignored (rate limits now come from rules[].cooldown_seconds): "
                + ", ".join(ignored)
            )
        if kept:
            parts.append(
                "left in alerts.notifiers because enabled is false: "
                + ", ".join(n.name for n in kept)
            )
        if not self.alerts.enabled:
            parts.append("alerts.enabled: false is ignored")
        _warn_once(
            "alerts.notifiers",
            "alerts.notifiers is deprecated and will be removed in the next release; %s. "
            "Move these entries under the top-level actions: key.",
            "; ".join(parts),
        )
        return self

    @model_validator(mode="after")
    def _check_action_names(self) -> AppConfig:
        """Action names are unique, and every ``rules[].actions`` name exists (spec 8.2)."""
        names = [a.name for a in self.actions]
        dups = _duplicate_names(names)
        if dups:
            raise ValueError(f"action name {dups[0]!r} is defined more than once in actions")
        known = set(names)
        disabled = {n.name for n in self.alerts.notifiers}
        for cam in self.cameras:
            for rule in cam.rules:
                for action_name in rule.actions:
                    if action_name in known:
                        continue
                    defined = ", ".join(sorted(known)) or "none"
                    hint = ""
                    if action_name in disabled:
                        hint = (
                            f"; {action_name!r} is under alerts.notifiers with enabled: false,"
                            " so it was not moved to actions"
                        )
                    raise ValueError(
                        f"camera {cam.name!r} rule {rule.name!r} names unknown action "
                        f"{action_name!r} (defined actions: {defined}){hint}"
                    )
        return self

    @model_validator(mode="after")
    def _validate_labels(self) -> AppConfig:
        """detect_classes and rules[].labels must be labels of the camera's onnx models.

        The valid set is the union of the labels of every ``onnx`` detector on the
        camera, enabled or not; rule labels may also be ``motion``. Cameras without
        an ``onnx`` detector are skipped. Only descriptors and labels files are
        read, never the model file, so this works before the first download.
        """
        for cam in self.cameras:
            try:
                per_model = camera_model_labels(cam, self.runtime.models_dir)
            except ModelError as exc:
                raise ValueError(f"camera {cam.name!r}: {exc}") from None
            if not per_model:
                continue
            universe = {label for labels in per_model.values() for label in labels}
            if cam.detect_classes is not None:
                unknown = [c for c in cam.detect_classes if c not in universe]
                if unknown:
                    raise ValueError(
                        unknown_labels_message(cam.name, "detect_classes", unknown, per_model)
                    )
            for rule in cam.rules:
                unknown = [lb for lb in rule.labels if lb != "motion" and lb not in universe]
                if unknown:
                    raise ValueError(
                        unknown_labels_message(
                            cam.name, f"rule {rule.name!r} labels", unknown, per_model
                        )
                    )
        return self


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = expand_env(raw)
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        raise SystemExit(f"Config validation failed:\n{e}") from e

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from .detectors.registry import DetectorSpec

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
    """Delete old segments and/or cap total size per camera.

    Notes:
      * keep_last_n is always honored first (newest N files are kept).
      * max_days and max_gb are applied to the remaining files.
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
    #   {output_dir}/{camera}/{stream}/%Y%m%d_%H%M%S.{container}
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


class CameraConfig(BaseModel):
    name: str
    main_url: str
    sub_url: str

    record: RecordConfig = Field(default_factory=RecordConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    detectors: list[DetectorSpec] = Field(default_factory=list)
    events: list[OnvifEventConfig] = Field(default_factory=list)
    presets: list[PTZPresetConfig] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("name must be non-empty")
        return v2


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


class DNNDetectorConfig(BaseModel):
    """Configuration for the YOLOv4-tiny DNN detector.

    This config type is used within DetectorSpec.config when
    type="dnn". It provides typed defaults and validation for
    the DNN detector parameters.

    Fields:
        type: Detector type discriminator, always "dnn".
        model: Pretrained model name. Currently only "yolov4-tiny" is
            supported. Determines which .cfg and .weights files are used.
        confidence: Minimum confidence threshold (0.0-1.0). Detections
            below this threshold are discarded.
        nms_threshold: Non-maximum suppression threshold (0.0-1.0).
            Higher values keep more overlapping detections.
        classes: List of COCO class names to detect. When None, defaults
            to vehicle + animal classes (car, truck, bus, motorcycle,
            bicycle, bird, cat, dog, horse, sheep, cow, elephant, bear,
            zebra, giraffe).
    """

    type: Literal["dnn"] = "dnn"
    model: str = "yolov4-tiny"
    confidence: float = 0.5
    nms_threshold: float = 0.4
    classes: list[str] | None = None

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("confidence must be in (0.0, 1.0]")
        return v

    @field_validator("nms_threshold")
    @classmethod
    def _nms_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("nms_threshold must be in (0.0, 1.0]")
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
    cameras: list[CameraConfig]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    onvif: OnvifConfig = Field(default_factory=OnvifConfig)
    clips: ClipsConfig = Field(default_factory=ClipsConfig)


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        raise SystemExit(f"Config validation failed:\n{e}") from e

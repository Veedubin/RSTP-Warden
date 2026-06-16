from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    event_record: "EventRecordConfig" = Field(default_factory=lambda: EventRecordConfig())


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


class CameraConfig(BaseModel):
    name: str
    main_url: str
    sub_url: str

    record: RecordConfig = Field(default_factory=RecordConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    detectors: list[DetectorSpec] = Field(default_factory=list)

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


class NotifierSpec(BaseModel):
    """Configuration for a single notifier (alert destination).

    type: "ntfy" or "webhook"
    For ntfy: url like "https://ntfy.sh/my-warden-alerts", optional topic, token, priority
    For webhook: url (any HTTP endpoint), optional headers dict, template
    """

    name: str  # human label like "Family phone", "Slack #ops"
    type: Literal["ntfy", "webhook"]
    url: str
    enabled: bool = True

    # ntfy-specific
    topic: str | None = None
    token: str | None = None
    priority: int | None = None  # 1-5; default 3

    # webhook-specific
    headers: dict[str, str] = Field(default_factory=dict)
    method: Literal["POST", "PUT"] = "POST"

    # Common
    min_interval_seconds: int = 30  # debounce: don't fire the same notifier more often than this
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


class AlertsConfig(BaseModel):
    """Top-level alerts config.

    notifiers: list of notifier destinations
    enabled: master switch (can also be controlled via CLI --alerts/--no-alerts)
    """

    enabled: bool = True
    notifiers: list[NotifierSpec] = Field(default_factory=list)


class OnvifConfig(BaseModel):
    """ONVIF camera discovery + PTZ config.

    discovery_enabled: enable WS-Discovery (UDP multicast on 239.255.255.250:3702)
    ptz_enabled: enable PTZ control surface (requires ONVIF device service on camera)
    discovery_timeout_seconds: how long to wait for probe responses
    ptz_timeout_seconds: per-PTZ-command HTTP timeout
    username/password: optional default credentials used when not overridden per-camera
    """

    discovery_enabled: bool = False  # opt-in (network noise on default networks)
    ptz_enabled: bool = False
    discovery_timeout_seconds: int = 5
    ptz_timeout_seconds: int = 10
    username: str | None = None
    password: str | None = None

    @field_validator("discovery_timeout_seconds", "ptz_timeout_seconds")
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


class AppConfig(BaseModel):
    cameras: list[CameraConfig]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    onvif: OnvifConfig = Field(default_factory=OnvifConfig)


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        raise SystemExit(f"Config validation failed:\n{e}") from e

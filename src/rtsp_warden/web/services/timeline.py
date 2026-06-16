"""Build timeline data for a recording (segments + events).

Scans on-disk .ts segment files within a recording's time range and
queries the events table for detector events, returning a unified
TimelineData object suitable for JSON serialization by the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ...db.engine import get_session
from ...db.models import Camera, Event
from .htl import scan_segments


@dataclass(slots=True)
class TimelineSegment:
    """A single .ts segment within the timeline."""

    start_ts: float  # unix seconds
    end_ts: float
    path: str  # filename relative to the recordings subdir
    size_bytes: int


@dataclass(slots=True)
class TimelineEvent:
    """A single detector event on the timeline."""

    id: int
    event_type: str
    severity: str
    ts_unix: float


@dataclass(slots=True)
class TimelineData:
    """Aggregated timeline data for a recording."""

    recording_id: int
    camera_name: str
    stream: str
    start_ts: float
    end_ts: float
    segments: list[TimelineSegment] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)


def build_timeline(
    recording_id: int,
    camera_name: str,
    stream: str,
    start_time: datetime,
    end_time: datetime | None,
    recordings_dir: Path,
) -> TimelineData:
    """Build the timeline data for a recording.

    Scans recordings_dir for .ts files whose mtime falls within the
    recording's time range, and queries the events table for detector
    events from this camera in the same window.

    Parameters
    ----------
    recording_id:
        Database ID of the recording.
    camera_name:
        Name of the camera that produced the recording.
    stream:
        Stream identifier (``"main"`` or ``"sub"``).
    start_time:
        Recording start timestamp.
    end_time:
        Recording end timestamp, or ``None`` for ongoing recordings.
    recordings_dir:
        Path to ``{output_dir}/{camera_name}/{stream}`` containing .ts files.

    Returns
    -------
    TimelineData
        Aggregated segments and events for the timeline UI.
    """
    start_ts = start_time.timestamp()
    end_ts = end_time.timestamp() if end_time is not None else start_ts + 3600.0

    # --- Segments ---
    raw_segments = scan_segments(recordings_dir)

    segments: list[TimelineSegment] = []
    for seg in raw_segments:
        seg_start = seg["start_time"]
        seg_end = seg_start + seg["duration"]

        # Only include segments that overlap the recording's time range.
        if seg_start >= end_ts or seg_end <= start_ts:
            continue

        # Resolve the file path to get its size.
        file_path = recordings_dir / seg["path"]
        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0

        segments.append(
            TimelineSegment(
                start_ts=seg_start,
                end_ts=seg_end,
                path=seg["path"],
                size_bytes=size,
            )
        )

    # --- Events ---
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    events: list[TimelineEvent] = []
    with get_session() as session:
        query = session.query(Event).join(Camera, Event.camera_id == Camera.id, isouter=True)
        query = query.filter(Camera.name == camera_name)
        query = query.filter(Event.created_at >= start_dt)
        query = query.filter(Event.created_at <= end_dt)
        query = query.order_by(Event.created_at.asc())
        rows = query.all()
        for row in rows:
            events.append(
                TimelineEvent(
                    id=row.id,
                    event_type=row.event_type,
                    severity=row.severity,
                    ts_unix=row.created_at.timestamp(),
                )
            )

    return TimelineData(
        recording_id=recording_id,
        camera_name=camera_name,
        stream=stream,
        start_ts=start_ts,
        end_ts=end_ts,
        segments=segments,
        events=events,
    )

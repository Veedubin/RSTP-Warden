"""Dynamic HLS playlist generation for time-windowed playback.

Given a camera, stream, and time window, build a virtual m3u8 that
references the on-disk .ts segments whose start_time falls within
the window.

The HTL (HLS Timeline) endpoint produces a playlist that allows
playback of historical recordings by selecting segments from a
specified time range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import AppConfig

# Default segment duration when gap-based estimation is unavailable.
_DEFAULT_SEGMENT_DURATION = 5.0


def parse_m3u8(m3u8_text: str) -> list[dict[str, Any]]:
    """Parse a standard HLS m3u8 file into a list of segment descriptors.

    Parameters
    ----------
    m3u8_text:
        Raw content of an m3u8 playlist file.

    Returns
    -------
    list of dict
        Each dict has ``"duration"`` (float, seconds) and ``"uri"`` (str).
        Non-segment lines (tags like #EXTM3U, #EXT-X-VERSION, etc.) are
        skipped.
    """
    segments: list[dict[str, Any]] = []
    lines = m3u8_text.strip().splitlines()
    current_duration: float | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#EXTINF:"):
            # Format: #EXTINF:<duration>,[<title>]
            duration_str = line[len("#EXTINF:") :].split(",")[0].strip()
            try:
                current_duration = float(duration_str)
            except ValueError:
                current_duration = _DEFAULT_SEGMENT_DURATION
        elif line.startswith("#"):
            # Skip all other tags
            continue
        else:
            # This is a segment URI
            if current_duration is not None:
                segments.append({"duration": current_duration, "uri": line})
                current_duration = None

    return segments


def build_htl_playlist(
    camera_name: str,
    stream_name: str,
    start_ts: float,
    end_ts: float,
    segments_dir: Path,
    segments_index: list[dict[str, Any]],
) -> str:
    """Build a virtual m3u8 playlist for a time window.

    Parameters
    ----------
    camera_name:
        Camera identifier (used in segment URLs).
    stream_name:
        Stream identifier (used in segment URLs).
    start_ts:
        Start of the time window as a unix timestamp.
    end_ts:
        End of the time window as a unix timestamp.
    segments_dir:
        Path to the recordings directory for this camera/stream.
    segments_index:
        List of segment descriptors, each with ``"path"``, ``"start_time"``,
        and ``"duration"`` keys.

    Returns
    -------
    str
        A valid m3u8 playlist string.
    """
    lines: list[str] = ["#EXTM3U", "#EXT-X-VERSION:3"]

    # Filter segments that overlap the time window.
    # A segment overlaps if its start_time < end_ts and
    # (start_time + duration) > start_ts.
    overlapping: list[dict[str, Any]] = []
    for seg in segments_index:
        seg_start = seg["start_time"]
        seg_duration = seg["duration"]
        seg_end = seg_start + seg_duration
        if seg_start < end_ts and seg_end > start_ts:
            overlapping.append(seg)

    if overlapping:
        max_duration = max(seg["duration"] for seg in overlapping)
        target_duration = int(max_duration) + (1 if max_duration > int(max_duration) else 0)
    else:
        target_duration = int(_DEFAULT_SEGMENT_DURATION)

    lines.append(f"#EXT-X-TARGETDURATION:{target_duration}")
    lines.append("#EXT-X-MEDIA-SEQUENCE:0")

    for seg in overlapping:
        lines.append(f"#EXTINF:{seg['duration']:.6f},")
        # Segment URL: relative path from the recordings root.
        # The /segments/ route serves: /segments/{camera_name}/{stream_name}/{path}
        lines.append(f"/segments/{camera_name}/{stream_name}/{seg['path']}")

    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines)


def scan_segments(segments_dir: Path) -> list[dict[str, Any]]:
    """Scan a directory for .ts segment files.

    Uses file mtime as an approximation of the segment start time.
    Estimates duration from the gap between consecutive segments,
    falling back to a default of 5 seconds for the last segment.

    TODO: Sprint 3 will replace mtime-based scanning with a proper
    segment index file ({dir}/index.json) for accurate timestamps.

    Parameters
    ----------
    segments_dir:
        Path to the directory containing .ts segment files.

    Returns
    -------
    list of dict
        Each dict has ``"path"`` (filename), ``"start_time"`` (float,
        unix timestamp from mtime), and ``"duration"`` (float, seconds).
    """
    if not segments_dir.is_dir():
        return []

    ts_files = sorted(
        (f for f in segments_dir.iterdir() if f.suffix == ".ts" and f.is_file()),
        key=lambda f: f.stat().st_mtime,
    )

    if not ts_files:
        return []

    segments: list[dict[str, Any]] = []
    for i, f in enumerate(ts_files):
        mtime = f.stat().st_mtime
        if i < len(ts_files) - 1:
            # Estimate duration from gap to next segment
            next_mtime = ts_files[i + 1].stat().st_mtime
            duration = max(next_mtime - mtime, 0.1)
        else:
            duration = _DEFAULT_SEGMENT_DURATION

        segments.append(
            {
                "path": f.name,
                "start_time": mtime,
                "duration": round(duration, 3),
            }
        )

    return segments


def get_recordings_dir(cfg: AppConfig, camera_name: str, stream_name: str) -> Path | None:
    """Return the recordings directory for a camera/stream pair.

    Looks up the camera configuration to find the output_dir, then
    appends the camera name and stream name to form the full path.

    Parameters
    ----------
    cfg:
        Application configuration containing camera definitions.
    camera_name:
        Name of the camera.
    stream_name:
        Stream identifier (e.g. "main" or "sub").

    Returns
    -------
    Path or None
        The recordings directory path, or None if the camera is not
        found in the configuration.
    """
    for cam in cfg.cameras:
        if cam.name == camera_name:
            return cam.record.output_dir / camera_name / stream_name
    return None

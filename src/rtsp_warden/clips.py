"""Clip generation from HLS segment detection events.

Given an event (with camera_id, recording_id, created_at), finds the
HLS segments that overlap the time window
[created_at - pre_seconds, created_at + post_seconds], concatenates
them via the ffmpeg concat demuxer, and produces a single MP4 clip.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from re import compile as re_compile

from .config import ClipsConfig

log = logging.getLogger(__name__)

# HLS segments follow the strftime pattern: %Y%m%d_%H%M%S.ts
_SEGMENT_RE = re_compile(r"^(\d{8}_\d{6})\.ts$")

# Default segment duration in seconds when m3u8 parsing is unavailable.
_DEFAULT_SEGMENT_DURATION = 4.0


class ClipError(Exception):
    """Raised when clip generation fails."""


@dataclass(slots=True)
class ClipInfo:
    """Metadata about a generated clip (not persisted; used as return type)."""

    path: Path
    duration_seconds: float
    size_bytes: int


class ClipGenerator:
    """Generate downloadable MP4 clips from HLS segments around events.

    Args:
        cfg: ClipsConfig with pre/post seconds, output directory, etc.
        recordings_dir: Base recordings directory (from RecordConfig.output_dir).
        ffmpeg_path: Path to ffmpeg binary.
    """

    def __init__(
        self,
        cfg: ClipsConfig,
        recordings_dir: Path,
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        self._cfg = cfg
        self._recordings_dir = recordings_dir
        self._ffmpeg_path = ffmpeg_path

    def find_segments(
        self,
        camera_name: str,
        stream: str,
        start_time: datetime,
        end_time: datetime,
        segment_duration: float = _DEFAULT_SEGMENT_DURATION,
    ) -> list[Path]:
        """Find HLS .ts segment files whose timestamps overlap [start_time, end_time].

        Segments are named %Y%m%d_%H%M%S.ts. We parse the filename to get
        the segment start time, then check overlap with the requested window.
        Each segment extends for segment_duration seconds.

        Args:
            camera_name: Camera directory name.
            stream: Stream subdirectory (e.g., "main" or "sub").
            start_time: Start of the time window (inclusive).
            end_time: End of the time window (exclusive).
            segment_duration: Duration of each segment in seconds (default 4s).

        Returns:
            Sorted list of Path objects for overlapping segments.
        """
        seg_dir = self._recordings_dir / camera_name / stream
        if not seg_dir.is_dir():
            log.warning("[clips] segment directory not found: %s", seg_dir)
            return []

        segments: list[tuple[datetime, Path]] = []
        for entry in seg_dir.iterdir():
            m = _SEGMENT_RE.match(entry.name)
            if not m:
                continue
            try:
                seg_start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

            seg_end = seg_start + __import__("datetime").timedelta(seconds=segment_duration)

            # Check overlap: seg_start < end_time AND seg_end > start_time
            if seg_start < end_time and seg_end > start_time:
                segments.append((seg_start, entry))

        # Sort by start time and return paths only
        segments.sort(key=lambda t: t[0])
        return [p for _, p in segments]

    def _parse_m3u8_durations(self, m3u8_path: Path) -> dict[str, float]:
        """Parse an m3u8 playlist file to extract segment durations.

        Returns a dict mapping segment filename to duration in seconds.
        Falls back to empty dict if the file cannot be parsed.
        """
        durations: dict[str, float] = {}
        if not m3u8_path.is_file():
            return durations

        try:
            lines = m3u8_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return durations

        current_duration: float | None = None
        for line in lines:
            line = line.strip()
            if line.startswith("#EXTINF:"):
                try:
                    dur_str = line[len("#EXTINF:") :]
                    # EXTINF can be "duration,title" format
                    dur_str = dur_str.split(",")[0].strip()
                    current_duration = float(dur_str)
                except (ValueError, IndexError):
                    current_duration = None
            elif line and not line.startswith("#") and current_duration is not None:
                # This is a segment filename
                durations[os.path.basename(line)] = current_duration
                current_duration = None

        return durations

    def _hls_time_to_datetime(
        self,
        recording_dir: Path,
        segment_duration: float = _DEFAULT_SEGMENT_DURATION,
    ) -> dict[str, datetime]:
        """Parse segment filenames to estimate start times.

        Returns a dict mapping segment filename to its estimated start datetime.
        Also attempts to read m3u8 for more accurate durations.
        """
        result: dict[str, datetime] = {}

        if not recording_dir.is_dir():
            return result

        for entry in recording_dir.iterdir():
            m = _SEGMENT_RE.match(entry.name)
            if not m:
                continue
            try:
                seg_start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                result[entry.name] = seg_start
            except ValueError:
                continue

        return result

    def _build_ffmpeg_concat_cmd(self, segment_paths: list[Path], output_path: Path) -> list[str]:
        """Build ffmpeg command to concat segments into a single mp4.

        Uses the concat demuxer with stream copy (no re-encoding).
        """
        return [
            self._ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(output_path.parent / "concat_list.txt"),
            "-c",
            "copy",
            "-an",
            str(output_path),
        ]

    def _write_concat_list(self, segment_paths: list[Path], concat_list_path: Path) -> None:
        """Write the ffmpeg concat list file.

        Each line: file '/absolute/path/to/segment.ts'
        """
        lines = []
        for seg_path in segment_paths:
            lines.append(f"file '{seg_path}'")
        concat_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def generate(
        self,
        camera_name: str,
        stream: str,
        event_start: datetime,
        event_id: int,
        pre_seconds: float | None = None,
        post_seconds: float | None = None,
    ) -> Path:
        """Generate an MP4 clip for an event.

        Args:
            camera_name: Camera name for finding recording segments.
            stream: Stream identifier (e.g., "main" or "sub").
            event_start: Event timestamp (used to compute the clip window).
            event_id: Event ID (used in output filename).
            pre_seconds: Seconds before event to include (default from config).
            post_seconds: Seconds after event to include (default from config).

        Returns:
            Path to the generated clip file.

        Raises:
            ClipError: If no segments found or ffmpeg fails.
        """
        pre = pre_seconds if pre_seconds is not None else self._cfg.pre_seconds
        post = post_seconds if post_seconds is not None else self._cfg.post_seconds

        start_time = event_start - __import__("datetime").timedelta(seconds=pre)
        end_time = event_start + __import__("datetime").timedelta(seconds=post)

        # Cap duration
        max_dur = self._cfg.max_duration
        actual_dur = (end_time - start_time).total_seconds()
        if actual_dur > max_dur:
            end_time = start_time + __import__("datetime").timedelta(seconds=max_dur)

        # Find overlapping segments
        segments = self.find_segments(camera_name, stream, start_time, end_time)
        if not segments:
            raise ClipError(
                f"No HLS segments found for camera={camera_name} stream={stream} "
                f"in window [{start_time}, {end_time})"
            )

        # Determine output path
        output_dir = Path(self._cfg.output_dir.format(recordings_root=str(self._recordings_dir)))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = event_start.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{camera_name}_{event_id}_{timestamp_str}.mp4"

        # Write concat list and run ffmpeg
        concat_list_path = output_dir / "concat_list.txt"
        try:
            self._write_concat_list(segments, concat_list_path)
            cmd = self._build_ffmpeg_concat_cmd(segments, output_path)

            log.info("[clips] generating clip: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                timeout=120,
                capture_output=True,
            )

            if result.returncode != 0:
                stderr_text = result.stderr.decode("utf-8", errors="replace")[:2000]
                log.error("[clips] ffmpeg failed (rc=%d): %s", result.returncode, stderr_text)
                raise ClipError(f"ffmpeg exited with code {result.returncode}: {stderr_text[:500]}")

            if not output_path.is_file():
                raise ClipError(f"ffmpeg completed but output file not found: {output_path}")

            return output_path

        except subprocess.TimeoutExpired:
            raise ClipError("ffmpeg timed out after 120 seconds") from None
        finally:
            # Clean up concat list
            if concat_list_path.is_file():
                try:
                    concat_list_path.unlink()
                except OSError:
                    pass

    def cleanup_old_clips(self, max_age_days: int | None = None) -> int:
        """Delete clip files older than max_age_days.

        Args:
            max_age_days: Override the config max_clip_age_days if provided.

        Returns:
            Number of clips deleted.
        """
        import time

        days = max_age_days  # Not using config field yet; for future use
        if days is None:
            # Default: keep for 30 days
            days = 30

        output_dir = Path(self._cfg.output_dir.format(recordings_root=str(self._recordings_dir)))
        if not output_dir.is_dir():
            return 0

        cutoff = time.time() - (days * 86400)
        count = 0
        for f in output_dir.glob("*.mp4"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    count += 1
            except OSError:
                pass
        return count

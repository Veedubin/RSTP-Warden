"""Event clips cut from the recorded ``.ts`` segments (spec 8.4; rulings R8 and R20).

A clip covers ``[started_at - pre_seconds, ended_at + post_seconds]``, capped at
``max_duration`` seconds from its start. It is cut from the camera's segments
``<camera_root>/<stream>/<camera>_<stream>_YYYYMMDD_HHMMSS.ts`` in two ffmpeg runs:

1. the concat demuxer with input-side ``-ss <offset> -t <duration>`` and ``-c copy``
   writes ``<camera_root>/clips/<event_id>.ts``;
2. a remux of that file writes ``<camera_root>/clips/<event_id>.mp4`` with
   ``-movflags +faststart``, and the ``.ts`` is deleted.

When the remux fails, the ``.ts`` from step 1 is kept and returned instead. Every ffmpeg
argv ends with its output path. ``subprocess.run`` is looked up at call time, so tests
patch it.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from re import compile as re_compile
from typing import Any

from .config import CameraConfig

log = logging.getLogger(__name__)

# Recorder segments are named {camera}_{stream}_%Y%m%d_%H%M%S.ts (recorder.py).
# A bare timestamp is accepted too, for files produced by older builds.
_SEGMENT_RE = re_compile(r"^(?:.+_)?(\d{8}_\d{6})\.ts$")

# The segment muxer cuts on the first keyframe after chunk_seconds, so a segment runs a
# little long. A segment is taken to end where the next one starts, but never later than
# chunk_seconds + SEGMENT_OVERRUN_S after its own start (a gap in the recording).
SEGMENT_OVERRUN_S = 10.0

# Upper bound for each ffmpeg run. Both runs are stream copies (seconds of I/O).
CLIP_FFMPEG_TIMEOUT_S = 120


def _to_utc(dt: datetime) -> datetime:
    """Aware UTC datetime. A naive value is taken to be UTC (SQLite returns naive UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clip_stream_and_chunk(cam: CameraConfig) -> tuple[str, int]:
    """Pick the stream to cut clips from and its segment length in seconds.

    The sub stream is smaller, so it wins when the camera has a ``sub_url`` and records it
    as ``.ts``. Otherwise the main stream. A camera that records nothing still gets
    ``("main", chunk)``; ``build_event_clip`` then finds no segments and returns None.
    """
    sub = cam.record.sub
    if cam.sub_url and sub.enabled and sub.container == "ts":
        return "sub", int(sub.chunk_seconds)
    return "main", int(cam.record.main.chunk_seconds)


def clip_rel_path(camera: str, clip_file: Path) -> str:
    """Clip path relative to ``record.output_dir``, as stored in ``events.clip_path``."""
    return f"{camera}/clips/{clip_file.name}"


def select_segments(
    camera_root: Path,
    stream: str,
    start: datetime,
    end: datetime,
    chunk_seconds: int,
) -> list[tuple[Path, datetime]]:
    """Return ``(path, segment start)`` for every segment overlapping ``[start, end)``.

    ``start``/``end`` are aware datetimes (naive values are read as UTC). A segment's start
    comes from its file name, which ffmpeg writes in local time. Its end is the next
    segment's start, capped at ``chunk_seconds + SEGMENT_OVERRUN_S``. The result is sorted
    by start time; returned starts are aware UTC.
    """
    seg_dir = camera_root / stream
    if not seg_dir.is_dir():
        return []
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)

    found: list[tuple[datetime, Path]] = []
    for entry in seg_dir.iterdir():
        m = _SEGMENT_RE.match(entry.name)
        if not m or not entry.is_file():
            continue
        try:
            # Naive local wall time -> aware UTC (astimezone presumes the local zone).
            seg_start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").astimezone(timezone.utc)
        except ValueError:
            continue
        found.append((seg_start, entry))
    found.sort(key=lambda item: item[0])

    longest = timedelta(seconds=chunk_seconds + SEGMENT_OVERRUN_S)
    selected: list[tuple[Path, datetime]] = []
    for i, (seg_start, path) in enumerate(found):
        seg_end = seg_start + longest
        if i + 1 < len(found):
            seg_end = min(seg_end, found[i + 1][0])
        if seg_start < end_utc and seg_end > start_utc:
            selected.append((path, seg_start))
    return selected


def _concat_list(paths: list[Path]) -> str:
    """Concat-demuxer list: absolute paths (ffmpeg resolves relative ones against the
    list's own directory), single quotes escaped as ``'\\''``."""
    lines = []
    for p in paths:
        quoted = str(p.absolute()).replace("'", "'\\''")
        lines.append(f"file '{quoted}'")
    return "\n".join(lines) + "\n"


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("[clips] could not delete %s: %s", p, exc)


def _nonempty(p: Path) -> bool:
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


def _run_ffmpeg(run: Callable[..., Any], cmd: list[str], *, event_id: int, step: str) -> bool:
    """Run one ffmpeg command. True on exit code 0; failures are logged, never raised."""
    try:
        result = run(cmd, capture_output=True, timeout=CLIP_FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log.warning(
            "[clips] event %d: ffmpeg %s timed out after %ds", event_id, step, CLIP_FFMPEG_TIMEOUT_S
        )
        return False
    except OSError as exc:
        log.warning("[clips] event %d: cannot run ffmpeg for %s: %s", event_id, step, exc)
        return False
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()[-500:]
        log.warning(
            "[clips] event %d: ffmpeg %s exited %d: %s", event_id, step, result.returncode, stderr
        )
        return False
    return True


def build_event_clip(
    *,
    camera_root: Path,
    stream: str,
    chunk_seconds: int,
    started_at: datetime,
    ended_at: datetime,
    pre_seconds: float,
    post_seconds: float,
    max_duration: float,
    event_id: int,
    ffmpeg_path: str,
    runner: Callable[..., Any] | None = None,
) -> Path | None:
    """Cut the clip for one closed event and return its path, or None when none was made.

    ``camera_root`` is ``<record.output_dir>/<camera>``. ``started_at``/``ended_at`` are the
    event bounds (aware, or naive UTC). The clip is ``<camera_root>/clips/<event_id>.mp4``,
    or ``<event_id>.ts`` when the MP4 remux fails. ``runner`` defaults to
    ``subprocess.run``, looked up at call time. Never raises for missing segments, ffmpeg
    failures or file-system errors; it logs them and returns None.
    """
    run = runner if runner is not None else subprocess.run

    window_start = _to_utc(started_at) - timedelta(seconds=pre_seconds)
    window_end = _to_utc(ended_at) + timedelta(seconds=post_seconds)
    if (window_end - window_start).total_seconds() > max_duration:
        window_end = window_start + timedelta(seconds=max_duration)

    segments = select_segments(camera_root, stream, window_start, window_end, chunk_seconds)
    if not segments:
        log.warning(
            "[clips] event %d: no recorded %s segments cover %s to %s",
            event_id,
            stream,
            window_start.isoformat(),
            window_end.isoformat(),
        )
        return None

    # The concat timeline starts at 0 at the first file's first packet, so the seek offset
    # is measured from the first segment's name time. A window that starts before the
    # first segment starts the clip at that segment instead (offset 0).
    first_start = segments[0][1]
    clip_start = max(window_start, first_start)
    offset = (clip_start - first_start).total_seconds()
    duration = (window_end - clip_start).total_seconds()

    clips_dir = camera_root / "clips"
    list_path = clips_dir / f"{event_id}.concat.txt"
    ts_path = clips_dir / f"{event_id}.ts"
    mp4_path = clips_dir / f"{event_id}.mp4"

    try:
        # Retention may have pruned an empty clips/ before; create it right before use.
        clips_dir.mkdir(parents=True, exist_ok=True)
        list_path.write_text(_concat_list([p for p, _ in segments]), encoding="utf-8")
    except OSError as exc:
        log.warning("[clips] event %d: cannot write %s: %s", event_id, list_path, exc)
        return None

    concat_cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{offset:.3f}",
        "-t",
        f"{duration:.3f}",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        "-an",
        "-f",
        "mpegts",
        str(ts_path),
    ]
    try:
        concat_ok = _run_ffmpeg(run, concat_cmd, event_id=event_id, step="concat")
    finally:
        _unlink(list_path)
    if not concat_ok or not _nonempty(ts_path):
        _unlink(ts_path)
        return None

    remux_cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(ts_path),
        "-c",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(mp4_path),
    ]
    if _run_ffmpeg(run, remux_cmd, event_id=event_id, step="remux") and _nonempty(mp4_path):
        _unlink(ts_path)
        return mp4_path
    _unlink(mp4_path)
    log.warning("[clips] event %d: MP4 remux failed; keeping %s", event_id, ts_path)
    return ts_path

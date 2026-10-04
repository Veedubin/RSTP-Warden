"""Event queries and view models for the web UI.

Every function here returns plain dicts (never ORM rows), so templates and the
dashboard can use them after the session is closed.

Time rules:
- the database stores naive UTC (``as_utc`` turns a stored value into aware UTC);
- a naive datetime handed to a filter here is LOCAL wall time (the dashboard
  passes local midnight); an aware one is converted; queries compare naive UTC;
- pages show local time, already formatted in the dicts (``started_display``),
  so no template needs a custom Jinja filter.

Thumbnails and clips are stored relative to the camera's ``record.output_dir``
(``<camera>/thumbnails/<id>.jpg``, ``<camera>/clips/<id>.mp4|.ts``).
``resolve_media_path`` turns a stored path into a file that exists inside one
of the configured output directories, or None; nothing is ever built from
request input.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import and_, or_
from sqlalchemy import func as sa_func

from ...config import AppConfig
from ...db.engine import get_session
from ...db.models import Event
from ...db.schema import as_utc, list_action_runs

log = logging.getLogger(__name__)

PAGE_SIZE = 24
TEST_EVENT_TYPE = "test"
DISPLAY_FORMAT = "%Y-%m-%d %H:%M:%S"

MediaState = Literal["none", "ok", "expired", "unknown"]
_CLIP_KINDS = {".mp4": "mp4", ".ts": "ts"}


def to_db_utc(dt: datetime | None) -> datetime | None:
    """Return ``dt`` as naive UTC for comparing with stored values.

    A naive input is local wall time (``datetime.astimezone`` reads it that way).
    """
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _local_midnight(day: date, tz: tzinfo | None) -> datetime:
    naive = datetime.combine(day, time.min)
    aware = naive.replace(tzinfo=tz) if tz is not None else naive.astimezone()
    return aware.astimezone(timezone.utc)


def parse_date_range(
    date_from: str | None, date_to: str | None, *, tz: tzinfo | None = None
) -> tuple[datetime | None, datetime | None]:
    """Turn the filter form's ``YYYY-MM-DD`` dates into an aware-UTC [since, until) window.

    ``date_from`` is the start of that day and ``date_to`` the start of the day
    after it (so the "to" day is included), both in ``tz`` (None = the server's
    local time zone). A missing or malformed date is no bound.
    """
    since = until = None
    if date_from:
        try:
            since = _local_midnight(date.fromisoformat(date_from), tz)
        except ValueError:
            since = None
    if date_to:
        try:
            until = _local_midnight(date.fromisoformat(date_to) + timedelta(days=1), tz)
        except ValueError:
            until = None
    return since, until


def format_local(dt: datetime | None, *, tz: tzinfo | None = None) -> str:
    """Format a stored datetime in ``tz`` (None = local time); "" for None."""
    aware = as_utc(dt)
    if aware is None:
        return ""
    return aware.astimezone(tz).strftime(DISPLAY_FORMAT)


def _duration_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def _percent(confidence: float | None) -> int | None:
    if confidence is None:
        return None
    return max(0, min(100, int(round(float(confidence) * 100))))


def _media_roots(cfg: AppConfig, camera_name: str | None) -> list[Path]:
    """The event's camera output dir first, then every other configured one.

    A camera deleted from config.yaml keeps its files on disk (decisions R10),
    so its events are still served when another camera shares the directory.
    """
    roots: list[Path] = []
    for cam in cfg.cameras:
        if cam.name == camera_name:
            roots.append(Path(cam.record.output_dir))
    for cam in cfg.cameras:
        root = Path(cam.record.output_dir)
        if root not in roots:
            roots.append(root)
    return roots


def resolve_media_path(
    cfg: AppConfig | None, camera_name: str | None, stored_path: str | None
) -> Path | None:
    """Return the existing file a stored thumbnail/clip path points to, or None.

    The result always lies inside a configured ``record.output_dir`` (after
    symlinks are resolved); a path with ``..`` or one that escapes every root is
    refused. Absolute stored paths are accepted only inside a root.
    """
    if cfg is None or not stored_path:
        return None
    stored = Path(stored_path)
    if ".." in stored.parts:
        return None
    for root in _media_roots(cfg, camera_name):
        try:
            root_resolved = root.resolve()
            candidate = (stored if stored.is_absolute() else root / stored).resolve()
        except (OSError, RuntimeError):
            continue
        if candidate.is_relative_to(root_resolved) and candidate.is_file():
            return candidate
    return None


def remove_event_files(cfg: AppConfig | None, row: Event) -> int:
    """Delete a (deleted) event's thumbnail and clip files; returns how many were removed.

    Paths resolve through :func:`resolve_media_path`, so nothing outside a configured
    ``record.output_dir`` is ever touched; a missing file is not an error.
    """
    removed = 0
    for stored in (row.thumbnail_path, row.clip_path):
        path = resolve_media_path(cfg, row.camera_name, stored)
        if path is None:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            log.warning("event %s: could not remove %s", row.id, path, exc_info=True)
    return removed


def _media_state(cfg: AppConfig | None, camera_name: str | None, stored: str | None) -> MediaState:
    if not stored:
        return "none"
    if cfg is None:
        return "unknown"
    return "ok" if resolve_media_path(cfg, camera_name, stored) is not None else "expired"


def _night_flag(metadata_json: str | None) -> bool | None:
    """The ``night`` flag of an event's metadata (RW-5), or None when absent or unreadable."""
    if not metadata_json:
        return None
    try:
        data = json.loads(metadata_json)
    except (TypeError, ValueError):
        return None
    value = data.get("night") if isinstance(data, dict) else None
    return value if isinstance(value, bool) else None


def event_to_dict(row: Event, cfg: AppConfig | None = None) -> dict[str, Any]:
    """Build the view model one event card / detail page renders.

    With ``cfg`` the thumbnail and clip files are checked on disk: a stored path
    whose file is gone gives ``thumbnail_expired`` / ``clip_expired``. Without a
    config (the dashboard's call) the URLs are kept and the card's ``onerror``
    fallback shows "expired" if the file is gone.
    """
    created_at = as_utc(row.created_at)
    ended_at = as_utc(row.ended_at)
    base = f"/events/{row.id}"
    thumb_state = _media_state(cfg, row.camera_name, row.thumbnail_path)
    clip_state = _media_state(cfg, row.camera_name, row.clip_path)
    clip_kind = None
    if clip_state in ("ok", "unknown") and row.clip_path:
        clip_kind = _CLIP_KINDS.get(Path(row.clip_path).suffix.lower())
    duration = ""
    if created_at is not None and ended_at is not None:
        duration = _duration_text((ended_at - created_at).total_seconds())
    return {
        "id": row.id,
        "detail_url": base,
        "camera_name": row.camera_name or "unknown",
        "event_type": row.event_type,
        "label": row.label or row.event_type or "event",
        "confidence": row.confidence,
        "confidence_pct": _percent(row.confidence),
        "zone": row.zone or "",
        "track_id": row.track_id,
        "message": row.message or "",
        "metadata_json": row.metadata_json or "{}",
        "night": _night_flag(row.metadata_json),
        "created_at": created_at,
        "ended_at": ended_at,
        "started_iso": created_at.isoformat() if created_at is not None else "",
        "started_display": format_local(created_at),
        "ended_display": format_local(ended_at),
        "duration_display": duration,
        "is_test": row.event_type == TEST_EVENT_TYPE,
        "thumbnail_path": row.thumbnail_path,
        "thumbnail_url": f"{base}/thumbnail.jpg" if thumb_state in ("ok", "unknown") else None,
        "thumbnail_expired": thumb_state == "expired",
        "clip_path": row.clip_path,
        "clip_kind": clip_kind,
        "clip_url": f"{base}/clip" if clip_kind else None,
        "clip_playlist_url": f"{base}/clip.m3u8" if clip_kind == "ts" else None,
        "clip_expired": clip_state == "expired",
    }


def _label_clause(label: str):
    # Rows written before migration 0003 have no label; their event_type is the label.
    return or_(Event.label == label, and_(Event.label.is_(None), Event.event_type == label))


def list_events(
    *,
    camera_name: str | None = None,
    label: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
    event_type: str | None = None,
    cfg: AppConfig | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return (event dicts newest first, total matching), filtered and paginated.

    ``since`` is inclusive and ``until`` exclusive; naive values are local time.
    Ties on ``created_at`` are broken by id, newest first.
    """
    since_db = to_db_utc(since)
    until_db = to_db_utc(until)
    with get_session() as session:
        query = session.query(Event)
        if camera_name is not None:
            query = query.filter(Event.camera_name == camera_name)
        if label is not None:
            query = query.filter(_label_clause(label))
        if event_type is not None:
            query = query.filter(Event.event_type == event_type)
        if since_db is not None:
            query = query.filter(Event.created_at >= since_db)
        if until_db is not None:
            query = query.filter(Event.created_at < until_db)
        total = query.count()
        rows = (
            query.order_by(Event.created_at.desc(), Event.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [event_to_dict(row, cfg) for row in rows], int(total)


def count_events_by_type(since: datetime | None = None) -> dict[str, int]:
    """Return {event_type: count}; a naive ``since`` is local time (dashboard midnight)."""
    since_db = to_db_utc(since)
    with get_session() as session:
        query = session.query(Event.event_type, sa_func.count(Event.id)).group_by(Event.event_type)
        if since_db is not None:
            query = query.filter(Event.created_at >= since_db)
        return {str(kind): int(count) for kind, count in query.all()}


def get_recent_events(limit: int = 10, event_type: str | None = None) -> list[dict[str, Any]]:
    """Return the newest events (same dicts as ``list_events``), optionally by type."""
    events, _ = list_events(limit=limit, event_type=event_type)
    return events


def get_event_by_id(event_id: int, cfg: AppConfig | None = None) -> dict[str, Any] | None:
    """Return one event dict, or None."""
    with get_session() as session:
        row = session.get(Event, event_id)
        if row is None:
            return None
        return event_to_dict(row, cfg)


def event_media_path(
    cfg: AppConfig | None, event_id: int, kind: Literal["thumbnail", "clip"]
) -> Path | None:
    """Return the event's thumbnail or clip file if it exists inside an output dir."""
    with get_session() as session:
        row = session.get(Event, event_id)
        if row is None:
            return None
        stored = row.thumbnail_path if kind == "thumbnail" else row.clip_path
        camera_name = row.camera_name
    return resolve_media_path(cfg, camera_name, stored)


def event_filter_options(cfg: AppConfig | None) -> dict[str, list[str]]:
    """Camera and label choices for the filter form.

    Cameras: every configured camera plus every camera name found in events
    (a deleted camera's events stay). Labels: every label (or, for rows from
    before migration 0003, event type) found in events.
    """
    with get_session() as session:
        cameras = {name for (name,) in session.query(Event.camera_name).distinct().all() if name}
        labels = {
            name
            for (name,) in session.query(sa_func.coalesce(Event.label, Event.event_type))
            .distinct()
            .all()
            if name
        }
    if cfg is not None:
        cameras.update(cam.name for cam in cfg.cameras)
    return {"cameras": sorted(cameras), "labels": sorted(labels)}


def list_event_action_runs(event_id: int) -> list[dict[str, Any]]:
    """Return the event's action runs, oldest first, as dicts for the detail table."""
    runs = sorted(list_action_runs(event_id), key=lambda run: run.id)
    return [
        {
            "action_name": run.action_name,
            "status": run.status,
            "error": run.error or "",
            "created_at": as_utc(run.created_at),
            "created_display": format_local(run.created_at),
        }
        for run in runs
    ]


def estimate_clip_seconds(
    started_at: datetime | None,
    ended_at: datetime | None,
    *,
    pre_seconds: float,
    post_seconds: float,
    max_duration: float,
) -> float:
    """Length the clip job cut (spec 8.4): pre + event span + post, capped, at least 1 s."""
    span = 0.0
    if started_at is not None and ended_at is not None:
        span = max(0.0, (as_utc(ended_at) - as_utc(started_at)).total_seconds())
    return float(max(1.0, min(max_duration, span + pre_seconds + post_seconds)))


def clip_playlist(event_id: int, duration_s: float) -> str:
    """One-entry VOD HLS playlist for a ``.ts`` clip; the entry is ``/events/<id>/clip``."""
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        f"#EXT-X-TARGETDURATION:{max(1, math.ceil(duration_s))}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        f"#EXTINF:{duration_s:.3f},",
        f"/events/{event_id}/clip",
        "#EXT-X-ENDLIST",
    ]
    return "\n".join(lines) + "\n"

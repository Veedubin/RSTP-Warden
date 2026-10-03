"""detection events: camera_name and event columns, action_runs; drop dead tables

Revision ID: 0003_detection_events
Revises: 0002_clips
Create Date: 2026-10-02 00:00:00.000000

Events gain the columns the detection pipeline writes (camera_name, label, confidence, zone,
track_id, ended_at, thumbnail_path, clip_path) and lose camera_id. camera_name is backfilled
from the cameras join, then metadata_json["camera"], then the EventSink message
"<kind> detected on <camera>/<stream> ...". The never-written cameras, recordings and
ingest_health tables and the clips table (replaced by events.clip_path) are dropped.

SQLite needs op.batch_alter_table for every ALTER. ix_events_camera_id must be dropped before
camera_id, or the batch table rebuild fails with "no such column: camera_id".
"""

from __future__ import annotations

import json
import re

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_detection_events"
down_revision: str | None = "0002_clips"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

# EventSink message format before this revision: f"{kind} detected on {camera}/{stream} (...)".
_MESSAGE_CAMERA_RE = re.compile(r" detected on (?P<camera>[^/]+)/")

_NEW_EVENT_COLUMNS = (
    "camera_name",
    "label",
    "confidence",
    "zone",
    "track_id",
    "ended_at",
    "thumbnail_path",
    "clip_path",
)


def _camera_from_row(metadata_json: str | None, message: str | None) -> str | None:
    """Camera name from metadata_json["camera"], else from the EventSink message, else None."""
    try:
        meta = json.loads(metadata_json or "{}")
    except ValueError:
        meta = None
    if isinstance(meta, dict) and meta.get("camera"):
        return str(meta["camera"])[:64]
    match = _MESSAGE_CAMERA_RE.search(message or "")
    return match.group("camera")[:64] if match else None


def upgrade() -> None:
    """Add the event columns, backfill camera_name, add action_runs, drop the dead tables."""
    with op.batch_alter_table("events") as batch_op:
        batch_op.add_column(sa.Column("camera_name", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("label", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("confidence", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("zone", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("track_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("thumbnail_path", sa.String(length=1024), nullable=True))
        batch_op.add_column(sa.Column("clip_path", sa.String(length=1024), nullable=True))

    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE events SET camera_name = "
            "(SELECT cameras.name FROM cameras WHERE cameras.id = events.camera_id) "
            "WHERE camera_id IS NOT NULL"
        )
    )
    rows = bind.execute(
        sa.text("SELECT id, metadata_json, message FROM events WHERE camera_name IS NULL")
    ).fetchall()
    for row_id, metadata_json, message in rows:
        camera = _camera_from_row(metadata_json, message)
        if camera is not None:
            bind.execute(
                sa.text("UPDATE events SET camera_name = :camera WHERE id = :id"),
                {"camera": camera, "id": row_id},
            )

    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_index("ix_events_camera_id")
        batch_op.drop_column("camera_id")
        batch_op.create_index("ix_events_camera_name", ["camera_name"])

    op.create_table(
        "action_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("action_name", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_action_runs_event_id", "action_runs", ["event_id"])

    op.drop_index("ix_clips_event_id", table_name="clips")
    op.drop_table("clips")
    op.drop_index("ix_ingest_health_camera_id", table_name="ingest_health")
    op.drop_table("ingest_health")
    op.drop_index("ix_recordings_start_time", table_name="recordings")
    op.drop_index("ix_recordings_camera_id", table_name="recordings")
    op.drop_table("recordings")
    op.drop_index("ix_cameras_name", table_name="cameras")
    op.drop_table("cameras")


def downgrade() -> None:
    """Recreate the 0002 tables and events.camera_id. New columns and action_runs rows are lost."""
    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("main_url", sa.String(length=512), nullable=False),
        sa.Column("sub_url", sa.String(length=512), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("config_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_cameras_name", "cameras", ["name"])

    op.create_table(
        "recordings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column("stream", sa.String(length=8), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("container", sa.String(length=8), nullable=False, server_default="ts"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("path"),
    )
    op.create_index("ix_recordings_camera_id", "recordings", ["camera_id"])
    op.create_index("ix_recordings_start_time", "recordings", ["start_time"])

    op.create_table(
        "ingest_health",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column("stream", sa.String(length=8), nullable=False),
        sa.Column("last_frame_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_segment_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingest_running", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("mjpeg_clients", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ingest_health_camera_id", "ingest_health", ["camera_id"])

    op.create_table(
        "clips",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=True),
        sa.Column("recording_id", sa.String(length=64), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.String(length=1024), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_clips_event_id", "clips", ["event_id"])

    op.drop_index("ix_action_runs_event_id", table_name="action_runs")
    op.drop_table("action_runs")

    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_index("ix_events_camera_name")
        for column in reversed(_NEW_EVENT_COLUMNS):
            batch_op.drop_column(column)
        batch_op.add_column(sa.Column("camera_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_events_camera_id_cameras", "cameras", ["camera_id"], ["id"], ondelete="SET NULL"
        )
        batch_op.create_index("ix_events_camera_id", ["camera_id"])

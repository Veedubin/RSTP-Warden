from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="viewer"
    )  # "admin" | "viewer"
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions: Mapped[list[Session]] = relationship(
        "Session", back_populates="user", cascade="all, delete-orphan"
    )
    api_tokens: Mapped[list[ApiToken]] = relationship(
        "ApiToken", back_populates="user", cascade="all, delete-orphan"
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship("User", back_populates="sessions", lazy="joined")


class ApiToken(Base):
    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    token_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship("User", back_populates="api_tokens", lazy="joined")


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    main_url: Mapped[str] = mapped_column(String(512), nullable=False)
    sub_url: Mapped[str] = mapped_column(String(512), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    config_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )  # JSON blob for per-camera overrides
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    recordings: Mapped[list[Recording]] = relationship(
        "Recording", back_populates="camera", cascade="all, delete-orphan"
    )
    events: Mapped[list[Event]] = relationship("Event", back_populates="camera")
    ingest_health: Mapped[list[IngestHealth]] = relationship(
        "IngestHealth", back_populates="camera", cascade="all, delete-orphan"
    )


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stream: Mapped[str] = mapped_column(String(8), nullable=False)  # "main" | "sub"
    path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    container: Mapped[str] = mapped_column(String(8), nullable=False, default="ts")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    camera: Mapped[Camera] = relationship("Camera", back_populates="recordings")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )  # motion, person, vehicle, ingest_lost, ...
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default="info"
    )  # info | warn | error
    message: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    camera: Mapped[Camera | None] = relationship("Camera", back_populates="events")


class IngestHealth(Base):
    __tablename__ = "ingest_health"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stream: Mapped[str] = mapped_column(String(8), nullable=False)
    last_frame_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_segment_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingest_running: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mjpeg_clients: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    camera: Mapped[Camera] = relationship("Camera", back_populates="ingest_health")


class Clip(Base):
    """Generated MP4 clip from HLS segments around an event."""

    __tablename__ = "clips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    recording_id: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(nullable=False, default=0.0)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )  # "pending" | "ready" | "failed"
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    event: Mapped[Event] = relationship("Event")
    camera: Mapped[Camera | None] = relationship("Camera")

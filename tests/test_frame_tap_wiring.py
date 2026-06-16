"""Tests for frame tap structured wiring — StreamIngestor, CameraRecorder, AppRuntime."""

from __future__ import annotations

from rtsp_warden.app import AppRuntime
from rtsp_warden.config import (
    AppConfig,
    CameraConfig,
    RecordConfig,
    RuntimeConfig,
    StreamRecordConfig,
)
from rtsp_warden.frame_tap import FrameConsumer, FrameTapDispatcher
from rtsp_warden.recorder import CameraRecorder, StreamIngestor

# ---------------------------------------------------------------------------
# FrameConsumer stub for testing
# ---------------------------------------------------------------------------


class StubConsumer(FrameConsumer):
    name: str = "stub"

    def on_frame(self, camera: str, stream: str, jpeg_bytes: bytes, ts_unix: float) -> None:
        pass


# ---------------------------------------------------------------------------
# StreamIngestor frame tap fields
# ---------------------------------------------------------------------------


def test_stream_ingestor_has_frame_tap_fields() -> None:
    """StreamIngestor has frame_tap_enabled, frame_tap_fps, frame_tap_scale_width, frame_tap_dispatcher."""
    ingestor = StreamIngestor(
        camera_name="test",
        stream_name="main",
        upstream_url="rtsp://example.com/main",
        runtime=RuntimeConfig(),
    )
    # Default values
    assert ingestor.frame_tap_enabled is False
    assert ingestor.frame_tap_fps == 5
    assert ingestor.frame_tap_scale_width == 320
    assert ingestor.frame_tap_dispatcher is None

    # Populated values
    dispatcher = FrameTapDispatcher()
    ingestor2 = StreamIngestor(
        camera_name="test2",
        stream_name="sub",
        upstream_url="rtsp://example.com/sub",
        runtime=RuntimeConfig(),
        frame_tap_enabled=True,
        frame_tap_fps=10,
        frame_tap_scale_width=160,
        frame_tap_dispatcher=dispatcher,
    )
    assert ingestor2.frame_tap_enabled is True
    assert ingestor2.frame_tap_fps == 10
    assert ingestor2.frame_tap_scale_width == 160
    assert ingestor2.frame_tap_dispatcher is dispatcher


# ---------------------------------------------------------------------------
# CameraRecorder frame_tap_dispatcher
# ---------------------------------------------------------------------------


def test_camera_recorder_accepts_dispatcher() -> None:
    """CameraRecorder frame_tap_dispatcher field round-trips."""
    cam_cfg = CameraConfig(
        name="test_cam",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        record=RecordConfig(
            enabled=True,
            main=StreamRecordConfig(enabled=True, container="ts"),
            sub=StreamRecordConfig(enabled=True, container="ts"),
        ),
    )
    dispatcher = FrameTapDispatcher(consumers=[StubConsumer()])
    recorder = CameraRecorder(
        camera=cam_cfg,
        runtime=RuntimeConfig(),
        frame_tap_dispatcher=dispatcher,
    )
    assert recorder.frame_tap_dispatcher is dispatcher


# ---------------------------------------------------------------------------
# AppRuntime frame_tap_dispatcher
# ---------------------------------------------------------------------------


def _make_minimal_config() -> AppConfig:
    return AppConfig(
        cameras=[
            CameraConfig(
                name="test_cam",
                main_url="rtsp://example.com/main",
                sub_url="rtsp://example.com/sub",
                record=RecordConfig(
                    enabled=True,
                    main=StreamRecordConfig(enabled=True, container="ts"),
                    sub=StreamRecordConfig(enabled=True, container="ts"),
                ),
            )
        ],
        runtime=RuntimeConfig(),
    )


def test_app_runtime_builds_with_dispatcher() -> None:
    """AppRuntime(cfg, frame_tap_dispatcher=d).build() succeeds."""
    cfg = _make_minimal_config()
    dispatcher = FrameTapDispatcher(consumers=[StubConsumer()])
    rt = AppRuntime(cfg=cfg, frame_tap_dispatcher=dispatcher)
    rt.build()
    assert len(rt.cameras) == 1
    # The dispatcher should be propagated to the recorder
    assert rt.cameras[0].recorder.frame_tap_dispatcher is dispatcher


def test_app_runtime_no_dispatcher_builds_cleanly() -> None:
    """frame_tap_dispatcher=None (default) builds without error."""
    cfg = _make_minimal_config()
    rt = AppRuntime(cfg=cfg, frame_tap_dispatcher=None)
    rt.build()
    assert len(rt.cameras) == 1
    assert rt.cameras[0].recorder.frame_tap_dispatcher is None


def test_frame_tap_enabled_propagates_to_ingestor() -> None:
    """When dispatcher is set, recorder.main.frame_tap_enabled is True and frame_tap_dispatcher is dispatcher."""
    cfg = _make_minimal_config()
    dispatcher = FrameTapDispatcher(consumers=[StubConsumer()])
    rt = AppRuntime(cfg=cfg, frame_tap_dispatcher=dispatcher)
    rt.build()

    cam_rt = rt.cameras[0]
    ingestor = cam_rt.recorder.main
    assert ingestor is not None
    assert ingestor.frame_tap_enabled is True
    assert ingestor.frame_tap_dispatcher is dispatcher


def test_motion_demo_consumer_still_works() -> None:
    """Sanity check the FrameConsumer contract is preserved."""
    from rtsp_warden.consumers.motion_demo import MotionHeuristicConsumer

    consumer = MotionHeuristicConsumer(threshold_ratio=0.20)
    dispatcher = FrameTapDispatcher(consumers=[consumer])

    # Should not raise
    dispatcher.dispatch("cam1", "sub", b"x" * 1000, 100.0)
    dispatcher.dispatch("cam1", "sub", b"x" * 1020, 100.1)
    dispatcher.dispatch("cam1", "sub", b"x" * 1400, 100.2)

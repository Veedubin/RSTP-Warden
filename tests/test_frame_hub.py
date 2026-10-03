"""FrameHub history (RW-4): the full-size frame nearest a detection time for thumbnails."""

from __future__ import annotations

from rtsp_warden.proxy.mjpeg import FrameHub

A, B, C, D = b"\xff\xd8a\xff\xd9", b"\xff\xd8b\xff\xd9", b"\xff\xd8c\xff\xd9", b"\xff\xd8d\xff\xd9"


def test_frame_near_is_none_on_an_empty_hub() -> None:
    assert FrameHub().frame_near(10.0) is None


def test_frame_near_returns_the_closest_frame_within_tolerance() -> None:
    hub = FrameHub()
    hub.update(A, ts_unix=10.0)
    hub.update(B, ts_unix=10.4)
    hub.update(C, ts_unix=10.8)

    assert hub.frame_near(10.5) == (B, 10.4)
    assert hub.frame_near(10.75) == (C, 10.8)
    assert hub.frame_near(9.9) == (A, 10.0)


def test_frame_near_is_none_outside_the_tolerance() -> None:
    hub = FrameHub()
    hub.update(A, ts_unix=10.0)

    assert hub.frame_near(11.0, tolerance_s=0.75) is None
    assert hub.frame_near(10.7, tolerance_s=0.75) == (A, 10.0)


def test_history_keeps_one_frame_per_interval_and_forgets_old_ones() -> None:
    hub = FrameHub(history_seconds=2.0, history_interval_s=0.5)
    hub.update(A, ts_unix=10.0)
    hub.update(B, ts_unix=10.1)  # too soon after A: not kept in the history
    hub.update(C, ts_unix=10.5)

    assert hub.frame_near(10.1) == (A, 10.0)  # B was never kept
    assert hub.snapshot()[0] == C  # the latest frame is still the latest

    hub.update(D, ts_unix=12.4)  # A (10.0) is now older than history_seconds, C (10.5) is not
    assert hub.frame_near(10.0, tolerance_s=0.3) is None
    assert hub.frame_near(10.5, tolerance_s=0.3) == (C, 10.5)
    assert hub.history_len == 2


def test_update_without_a_timestamp_uses_the_clock() -> None:
    hub = FrameHub()
    hub.update(A)
    _jpeg, _fid, ts = hub.snapshot()
    assert hub.frame_near(ts) == (A, ts)

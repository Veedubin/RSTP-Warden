"""Tests for detectors/event_builder.py: MotionBurst (one motion event per burst)."""

from __future__ import annotations

import pytest

from rtsp_warden.detectors.event_builder import MotionBurst


def _feed(burst: MotionBurst, frames: list[tuple[float, bool]]) -> list[tuple[float, bool, bool]]:
    """Feed (ts, has_motion) pairs; return (ts, opened, closed) for every transition."""
    out: list[tuple[float, bool, bool]] = []
    for ts, has_motion in frames:
        opened, closed = burst.update(has_motion, ts)
        if opened or closed:
            out.append((ts, opened, closed))
    return out


def test_burst_opens_after_min_frames_consecutive_motion_frames() -> None:
    burst = MotionBurst(min_frames=2, grace_seconds=3.0)
    assert burst.update(False, 0.0) == (False, False)
    assert burst.update(True, 0.2) == (False, False)
    assert burst.update(True, 0.4) == (True, False)
    assert burst.is_open
    assert burst.start_ts == 0.2
    assert burst.last_motion_ts == 0.4


def test_single_frame_noise_never_opens_a_burst() -> None:
    burst = MotionBurst(min_frames=2, grace_seconds=3.0)
    frames = [(0.0, True), (0.2, False), (0.4, True), (0.6, False), (0.8, True), (1.0, False)]
    assert _feed(burst, frames) == []
    assert not burst.is_open


def test_burst_closes_grace_seconds_after_the_last_motion_frame() -> None:
    burst = MotionBurst(min_frames=2, grace_seconds=3.0)
    _feed(burst, [(0.0, True), (0.2, True), (0.4, True), (0.6, True)])
    assert burst.update(False, 1.0) == (False, False)
    assert burst.update(False, 3.5) == (False, False)  # 3.5 - 0.6 = 2.9 < 3.0
    assert burst.update(False, 3.7) == (False, True)  # 3.7 - 0.6 = 3.1 >= 3.0
    assert burst.closed_end_ts == 0.6
    assert not burst.is_open
    assert burst.start_ts is None


def test_gap_shorter_than_grace_keeps_one_burst() -> None:
    burst = MotionBurst(min_frames=2, grace_seconds=3.0)
    frames = [(0.0, True), (0.2, True)]
    frames += [(round(0.4 + 0.2 * i, 6), False) for i in range(9)]  # 0.4 .. 2.0
    frames += [(2.3, True), (2.5, True)]  # 2.3 - 0.2 < 3.0: same burst
    frames += [(round(2.6 + 0.2 * i, 6), False) for i in range(20)]  # 2.6 .. 6.4
    transitions = _feed(burst, frames)
    # Closes on the first frame >= 3.0 s after 2.5: 5.4 is 2.9 s, 5.6 is 3.1 s.
    assert transitions == [(0.2, True, False), (5.6, False, True)]
    assert burst.closed_end_ts is None  # reset by the updates after the close


def test_gap_longer_than_grace_gives_two_bursts() -> None:
    burst = MotionBurst(min_frames=2, grace_seconds=3.0)
    frames = [(0.0, True), (0.2, True), (1.0, False), (5.0, True), (5.2, True)]
    transitions = _feed(burst, frames)
    # 5.0: the first burst closes (5.0 - 0.2 >= 3.0) and a new run starts; 5.2 opens it.
    assert transitions == [(0.2, True, False), (5.0, False, True), (5.2, True, False)]
    assert burst.start_ts == 5.0


def test_close_and_open_on_the_same_update_with_min_frames_one() -> None:
    burst = MotionBurst(min_frames=1, grace_seconds=1.0)
    assert burst.update(True, 0.0) == (True, False)
    assert burst.update(True, 5.0) == (True, True)
    assert burst.closed_end_ts == 0.0
    assert burst.start_ts == 5.0


def test_close_returns_last_motion_ts_once() -> None:
    burst = MotionBurst(min_frames=1, grace_seconds=3.0)
    assert burst.close() is None
    burst.update(True, 10.0)
    burst.update(True, 10.5)
    assert burst.close() == 10.5
    assert not burst.is_open
    assert burst.close() is None


@pytest.mark.parametrize(
    ("min_frames", "grace"),
    [(0, 3.0), (2, 0.0), (2, -1.0)],
)
def test_invalid_arguments_raise(min_frames: int, grace: float) -> None:
    with pytest.raises(ValueError):
        MotionBurst(min_frames=min_frames, grace_seconds=grace)

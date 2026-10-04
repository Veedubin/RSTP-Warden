"""Tests for detectors/daylight.py: grayscale (IR) detection with hysteresis."""

from __future__ import annotations

import numpy as np
import pytest

from rtsp_warden.detectors.daylight import (
    NIGHT_SPREAD,
    SWITCH_FRAMES,
    DayNight,
    allows,
    channel_spread,
)

H, W = 90, 160


def grey(level: int = 120) -> np.ndarray:
    return np.full((H, W, 3), level, dtype=np.uint8)


def colour() -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[..., 0] = 40  # B
    frame[..., 1] = 120  # G
    frame[..., 2] = 200  # R
    return frame


def dark() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def test_spread_is_zero_for_grey_and_large_for_colour() -> None:
    assert channel_spread(grey()) == 0.0
    assert channel_spread(colour()) == pytest.approx(160.0)
    assert channel_spread(dark()) == 0.0


def test_spread_tolerates_bad_input() -> None:
    assert channel_spread(np.zeros((H, W), dtype=np.uint8)) == 0.0
    assert channel_spread(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
    assert channel_spread(None) == 0.0  # type: ignore[arg-type]


def test_first_frame_sets_the_state_at_once() -> None:
    dn = DayNight()
    assert dn.night is None
    assert dn.update(grey(), 10.0) is True
    assert dn.night is True and dn.since_ts == 10.0 and dn.switches == 0
    assert dn.last_spread == 0.0


def test_two_frames_do_not_flip_three_do() -> None:
    dn = DayNight()
    dn.update(colour(), 1.0)
    assert dn.update(grey(), 2.0) is False
    assert dn.update(grey(), 3.0) is False
    assert dn.update(grey(), 4.0) is True
    assert dn.since_ts == 4.0 and dn.switches == 1


def test_a_single_odd_frame_resets_the_pending_count() -> None:
    dn = DayNight()
    dn.update(colour(), 1.0)
    dn.update(grey(), 2.0)
    dn.update(grey(), 3.0)
    dn.update(colour(), 4.0)  # back on the current side: pending resets
    dn.update(grey(), 5.0)
    assert dn.update(grey(), 6.0) is False
    assert dn.update(grey(), 7.0) is True


def test_dark_then_colour_flips_after_three_frames() -> None:
    """(review focus) Black warm-up frames count as night; daylight takes over quickly."""
    dn = DayNight()
    for ts in (1.0, 2.0, 3.0):
        assert dn.update(dark(), ts) is True
    assert dn.update(colour(), 4.0) is True
    assert dn.update(colour(), 5.0) is True
    assert dn.update(colour(), 6.0) is False
    assert dn.switches == 1


def test_threshold_and_switch_frames_are_configurable() -> None:
    dn = DayNight(threshold=200.0, switch_frames=1)
    dn.update(grey(), 1.0)
    assert dn.update(colour(), 2.0) is True  # 160 < 200: still "night"
    dn2 = DayNight(threshold=NIGHT_SPREAD, switch_frames=1)
    dn2.update(grey(), 1.0)
    assert dn2.update(colour(), 2.0) is False
    assert SWITCH_FRAMES == 3


def test_allows() -> None:
    assert allows("always", None) and allows("always", True) and allows("always", False)
    assert allows("night", True) and not allows("night", False) and not allows("night", None)
    assert allows("day", False) and allows("day", None) and not allows("day", True)

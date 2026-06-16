"""Tests for detectors/sensitivity.py: sensitivity scaling functions."""

from __future__ import annotations

from rtsp_warden.detectors.sensitivity import (
    apply_sensitivity_to_confidence,
    apply_sensitivity_to_motion,
    apply_sensitivity_to_nms,
    motion_detector_sensitivity,
)

# ---------------------------------------------------------------------------
# apply_sensitivity_to_motion
# ---------------------------------------------------------------------------


def test_motion_sensitivity_100_most_sensitive() -> None:
    """sensitivity=100 (most sensitive) -> varThreshold=8 (lowest threshold)."""
    assert apply_sensitivity_to_motion(100) == 8


def test_motion_sensitivity_0_least_sensitive() -> None:
    """sensitivity=0 (least sensitive) -> varThreshold=100 (highest threshold)."""
    assert apply_sensitivity_to_motion(0) == 100


def test_motion_sensitivity_50_neutral() -> None:
    """sensitivity=50 (neutral) -> varThreshold=50."""
    assert apply_sensitivity_to_motion(50) == 50


def test_motion_sensitivity_150_clamps_to_8() -> None:
    """sensitivity > 100 is clamped, resulting in varThreshold=8."""
    assert apply_sensitivity_to_motion(150) == 8


def test_motion_sensitivity_negative_clamps_to_100() -> None:
    """sensitivity < 0 is clamped, resulting in varThreshold=100."""
    assert apply_sensitivity_to_motion(-10) == 100


def test_motion_sensitivity_with_custom_base() -> None:
    """Custom base_var_threshold changes the range."""
    # base=25: sensitivity=0 -> (1-0)*25*2=50, sensitivity=100 -> (1-1)*25*2=0 -> clamped 8
    assert apply_sensitivity_to_motion(0, base_var_threshold=25) == 50
    assert apply_sensitivity_to_motion(100, base_var_threshold=25) == 8


# ---------------------------------------------------------------------------
# apply_sensitivity_to_confidence
# ---------------------------------------------------------------------------


def test_confidence_100_near_01() -> None:
    """sensitivity=100 (most sensitive) -> confidence near 0.1."""
    result = apply_sensitivity_to_confidence(100)
    assert abs(result - 0.1) < 0.01


def test_confidence_0_near_09() -> None:
    """sensitivity=0 (least sensitive) -> confidence near 0.9."""
    result = apply_sensitivity_to_confidence(0)
    assert abs(result - 0.9) < 0.01


def test_confidence_50_default() -> None:
    """sensitivity=50 with default=0.5 -> confidence near 0.5."""
    result = apply_sensitivity_to_confidence(50)
    assert abs(result - 0.5) < 0.01


def test_confidence_25() -> None:
    """sensitivity=25 -> confidence higher than neutral."""
    result = apply_sensitivity_to_confidence(25)
    assert result > 0.5


def test_confidence_75() -> None:
    """sensitivity=75 -> confidence lower than neutral."""
    result = apply_sensitivity_to_confidence(75)
    assert result < 0.5


def test_confidence_clamps_to_range() -> None:
    """Confidence is clamped to [0.1, 0.9]."""
    assert apply_sensitivity_to_confidence(-100) == 0.9
    assert apply_sensitivity_to_confidence(200) == 0.1


def test_confidence_custom_default() -> None:
    """Custom default shifts the midpoint."""
    result = apply_sensitivity_to_confidence(50, default=0.7)
    assert abs(result - 0.7) < 0.01


# ---------------------------------------------------------------------------
# apply_sensitivity_to_nms
# ---------------------------------------------------------------------------


def test_nms_100_near_03() -> None:
    """sensitivity=100 -> NMS near 0.3 (less suppression)."""
    result = apply_sensitivity_to_nms(100)
    assert abs(result - 0.3) < 0.01


def test_nms_0_near_06() -> None:
    """sensitivity=0 -> NMS near 0.6 (more suppression)."""
    result = apply_sensitivity_to_nms(0)
    assert abs(result - 0.6) < 0.01


def test_nms_50() -> None:
    """sensitivity=50 -> NMS at midpoint."""
    result = apply_sensitivity_to_nms(50)
    assert 0.4 <= result <= 0.5


def test_nms_clamps_to_range() -> None:
    """NMS is clamped to [0.3, 0.7]."""
    assert apply_sensitivity_to_nms(-100) == 0.6  # clamped high
    assert apply_sensitivity_to_nms(200) == 0.3  # clamped low


# ---------------------------------------------------------------------------
# motion_detector_sensitivity
# ---------------------------------------------------------------------------


def test_motion_detector_sensitivity_0() -> None:
    """camera_sensitivity=0 -> MD sensitivity=0.0 (least sensitive)."""
    assert motion_detector_sensitivity(0) == 0.0


def test_motion_detector_sensitivity_100() -> None:
    """camera_sensitivity=100 -> MD sensitivity=1.0 (most sensitive)."""
    assert motion_detector_sensitivity(100) == 1.0


def test_motion_detector_sensitivity_50() -> None:
    """camera_sensitivity=50 -> MD sensitivity=0.5."""
    assert motion_detector_sensitivity(50) == 0.5


def test_motion_detector_sensitivity_clamps() -> None:
    """camera_sensitivity outside [0, 100] is clamped."""
    assert motion_detector_sensitivity(-10) == 0.0
    assert motion_detector_sensitivity(150) == 1.0

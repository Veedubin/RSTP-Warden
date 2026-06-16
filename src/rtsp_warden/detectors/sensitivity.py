"""Map camera sensitivity (0-100) to per-detector parameters.

Sensitivity is a single knob on the camera level (0-100) that controls
how aggressive detection is.  Higher values mean more detections (lower
thresholds), lower values mean fewer detections (higher thresholds).

Each detector type maps sensitivity differently:
  - MotionDetector -> varThreshold (MOG2 background subtractor)
  - PersonDetector / VehicleDetector -> min_confidence
  - DNNDetector   -> confidence_threshold and nms_threshold
"""

from __future__ import annotations


def apply_sensitivity_to_motion(sensitivity: float, base_var_threshold: int = 50) -> int:
    """Map 0-100 sensitivity to MOG2 varThreshold.

    sensitivity=100 (most sensitive) -> varThreshold=8 (lowest, fires on small changes)
    sensitivity=50  (neutral)        -> varThreshold=50
    sensitivity=0   (least sensitive) -> varThreshold=100 (only big changes fire)
    """
    sens = max(0.0, min(100.0, sensitivity))
    return max(8, min(100, int((1 - sens / 100) * base_var_threshold * 2)))


def apply_sensitivity_to_confidence(sensitivity: float, default: float = 0.5) -> float:
    """Map 0-100 sensitivity to detection confidence threshold.

    sensitivity=100 -> confidence=0.1 (very lax, lots of detections)
    sensitivity=50  -> confidence=0.55
    sensitivity=0   -> confidence=0.9 (very strict, few detections)
    """
    sens = max(0.0, min(100.0, sensitivity))
    return max(0.1, min(0.9, default + (50 - sens) / 100))


def apply_sensitivity_to_nms(sensitivity: float) -> float:
    """Map 0-100 sensitivity to NMS threshold.

    sensitivity=100 -> nms=0.3 (less suppression, more overlapping boxes kept)
    sensitivity=50  -> nms=0.45
    sensitivity=0   -> nms=0.6 (more suppression, fewer overlapping boxes)
    """
    sens = max(0.0, min(100.0, sensitivity))
    return max(0.3, min(0.7, 0.3 + (1 - sens / 100) * 0.3))


def motion_detector_sensitivity(camera_sensitivity: float) -> float:
    """Convert camera sensitivity (0-100) to MotionDetector's internal sensitivity (0-1).

    This is the 0-1 value passed to MotionDetector.sensitivity, which is
    then converted internally to varThreshold via _sensitivity_to_var_threshold.

    Higher camera sensitivity = more detections = lower varThreshold.
    The MotionDetector's sensitivity parameter is inversely related to varThreshold:
        varThreshold = int((1 - sensitivity) * 50), clamped to [8, 100]

    Mapping:
        camera_sensitivity=0  -> MD sensitivity=0.0  -> varThreshold=50 (moderate)
        camera_sensitivity=50 -> MD sensitivity=0.5  -> varThreshold=25 (default-like)
        camera_sensitivity=100 -> MD sensitivity=1.0 -> varThreshold=8  (very sensitive)

    For very low camera sensitivity (0-50), the varThreshold produced by
    this mapping tops out at 50. To get higher varThreshold values
    (up to 100), use apply_sensitivity_to_motion() and pass var_threshold
    directly to MotionDetector.
    """
    return max(0.0, min(1.0, camera_sensitivity / 100.0))


__all__ = [
    "apply_sensitivity_to_motion",
    "apply_sensitivity_to_confidence",
    "apply_sensitivity_to_nms",
    "motion_detector_sensitivity",
]

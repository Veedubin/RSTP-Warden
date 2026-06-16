"""Camera-level detection class filtering.

Camera's detect_classes (if set) is intersected with each detector's
allowed_classes. None means "no preference, use detector's default".
"""

from __future__ import annotations


def effective_classes(
    camera_classes: list[str] | None,
    detector_classes: list[str] | None,
) -> list[str] | None:
    """Return the intersection of camera_classes and detector_classes.

    Cases:
    - both None: return None (no filter, detector sees everything)
    - one None, one set: return the set (the non-None one is the only filter)
    - both set: return intersection; if empty, return [] (which means "match nothing")
    """
    if camera_classes is None and detector_classes is None:
        return None
    if camera_classes is None:
        return list(detector_classes)
    if detector_classes is None:
        return list(camera_classes)
    cam_set = set(camera_classes)
    det_set = set(detector_classes)
    return sorted(cam_set & det_set)


__all__ = ["effective_classes"]

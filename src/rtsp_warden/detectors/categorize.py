"""Map detector labels to object type categories for timeline coloring.

COCO 80-class labels and common custom detector labels are grouped into
five categories that the timeline UI renders with distinct colors.

Categories:
    person:  human detections
    pet:     domestic animals (cat, dog)
    critter: wildlife and outdoor pests
    vehicle: motorized and non-motorized transport
    other:   everything else (including "motion" from MotionDetector)
"""

from __future__ import annotations

_PERSON_LABELS: frozenset[str] = frozenset({"person"})

_PET_LABELS: frozenset[str] = frozenset({"cat", "dog"})

_CRITTER_LABELS: frozenset[str] = frozenset(
    {
        "bird",
        "butterfly",
        "rabbit",
        "squirrel",
        "deer",
        "raccoon",
        "fox",
        "coyote",
        "skunk",
        "mouse",
        "bear",
        "horse",
        "sheep",
        "cow",
    }
)

_VEHICLE_LABELS: frozenset[str] = frozenset(
    {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
)


def categorize_object(label: str) -> str:
    """Map a detector label to an object type category.

    Performs a case-insensitive lookup against known COCO labels. Falls
    back to heuristic substring matching for custom detector labels that
    do not exactly match a known class name.

    Args:
        label: Raw detector label (e.g. "car", "PERSON", "my-custom-dog-detector").

    Returns:
        One of: "person", "pet", "critter", "vehicle", "other".
    """
    label_lower = label.lower().strip()

    if label_lower in _PERSON_LABELS:
        return "person"
    if label_lower in _PET_LABELS:
        return "pet"
    if label_lower in _CRITTER_LABELS:
        return "critter"
    if label_lower in _VEHICLE_LABELS:
        return "vehicle"

    # Heuristic fallback for custom detector labels that contain a known keyword.
    if "person" in label_lower or "human" in label_lower or "face" in label_lower:
        return "person"
    if "dog" in label_lower or "cat" in label_lower or "pet" in label_lower:
        return "pet"
    if (
        "deer" in label_lower
        or "raccoon" in label_lower
        or "fox" in label_lower
        or "critter" in label_lower
    ):
        return "critter"
    if (
        "car" in label_lower
        or "truck" in label_lower
        or "vehicle" in label_lower
        or "bus" in label_lower
    ):
        return "vehicle"

    return "other"


__all__ = ["categorize_object"]

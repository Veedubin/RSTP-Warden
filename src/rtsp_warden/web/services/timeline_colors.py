"""Color palette for timeline markers by object type.

Provides hex color constants for each object category and a lookup
function that returns the color for a given category string. Used by
both the Python timeline builder and the JavaScript renderer.

The palette is chosen for readability against the dark timeline canvas
background (#1a1a2e).
"""

from __future__ import annotations

OBJECT_TYPE_COLORS: dict[str, str] = {
    "person": "#e74c3c",
    "pet": "#3498db",
    "critter": "#f1c40f",
    "vehicle": "#e67e22",
    "other": "#7f8c8d",
}


def color_for_object_type(object_type: str) -> str:
    """Return the hex color for an object type category.

    Falls back to the "other" color (gray) for unknown categories.

    Args:
        object_type: One of the category strings returned by
            ``categorize_object`` ("person", "pet", "critter",
            "vehicle", "other").

    Returns:
        Hex color string (e.g. "#e74c3c").
    """
    return OBJECT_TYPE_COLORS.get(object_type, OBJECT_TYPE_COLORS["other"])


__all__ = ["OBJECT_TYPE_COLORS", "color_for_object_type"]

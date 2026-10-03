"""Removed in 1.4 (ruling R5): nothing ever wrote the recordings table.

Kept as a stub until feat/detection is rebased onto the RW-2 UI pass, so RW-2's
``camera_detail`` and dashboard routes (which still import ``list_recordings``)
keep importing cleanly. Delete this module once those imports are gone.
"""

from __future__ import annotations

from typing import Any


def list_recordings(*args: object, **kwargs: object) -> tuple[list[dict[str, Any]], int]:
    """No recordings rows exist any more: always ``([], 0)``."""
    return [], 0


def get_recording_by_id(recording_id: int) -> dict[str, Any] | None:
    """No recordings rows exist any more: always ``None``."""
    return None

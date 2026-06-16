"""Runtime status service for the web UI.

Provides a status dict adapted from the CLI build_status function,
safe for template rendering when no live runtime is attached.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ... import __version__

if TYPE_CHECKING:
    from ...app import AppRuntime
    from ...config import AppConfig


def get_runtime_status(rt: AppRuntime | None, cfg: AppConfig) -> dict[str, Any]:
    """Return a status dict suitable for web display.

    If *rt* is None (no recorder started), returns a minimal status
    with no live data. When provided, inspects each camera runtime
    for process state, frame timestamps, and proxy availability.
    """
    if rt is None:
        return {
            "ok": True,
            "version": __version__,
            "now": time.time(),
            "cameras": [
                {
                    "name": c.name,
                    "ok": True,
                    "status": "stopped",
                    "streams": {},
                }
                for c in cfg.cameras
            ],
            "errors": [],
        }

    # Delegate to the CLI build_status when a live runtime is attached.
    # Import here to avoid circular dependency at module level.
    from ...cli import build_status

    return build_status(rt, cfg, __version__)

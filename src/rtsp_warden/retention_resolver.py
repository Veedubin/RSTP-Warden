"""Per-camera retention policy resolution.

Provides the `resolve_retention` function that determines the effective
retention configuration for a camera by checking the per-camera override
first, then falling back to the global default.  Returns a deep copy so
that runtime mutations (e.g. setting `_next_run`) never leak across cameras.
"""

from __future__ import annotations

from .config import CameraConfig, RetentionConfig


def resolve_retention(camera: CameraConfig, global_cfg: RetentionConfig) -> RetentionConfig:
    """Return the effective retention config for a camera.

    Resolution order:
        1. camera.retention  (per-camera override, if set)
        2. global_cfg        (app-wide fallback)

    A deep copy is always returned so that RetentionManager can mutate
    the config (e.g. tracking internal state) without causing cross-camera
    state leaks.

    Args:
        camera: The camera configuration (may have ``retention`` override).
        global_cfg: The app-wide retention default.

    Returns:
        A new RetentionConfig instance (deep copy) ready for use by
        RetentionManager.
    """
    if camera.retention is not None:
        return camera.retention.model_copy(deep=True)
    return global_cfg.model_copy(deep=True)

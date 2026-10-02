"""Resolve the effective retention policy for a camera."""

from __future__ import annotations

import logging

from .config import CameraConfig, RetentionConfig

log = logging.getLogger(__name__)


def resolve_retention(camera: CameraConfig, global_cfg: RetentionConfig) -> RetentionConfig:
    """Return the effective retention config for a camera.

    Resolution order:
        1. camera.retention           (per-camera override)
        2. camera.record.retention    (deprecated location; honored with a warning)
        3. global_cfg                 (app-wide fallback)

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
    if "retention" in camera.record.model_fields_set:
        log.warning(
            "camera %r sets record.retention, which is deprecated; move it to cameras[].retention",
            camera.name,
        )
        return camera.record.retention.model_copy(deep=True)
    return global_cfg.model_copy(deep=True)

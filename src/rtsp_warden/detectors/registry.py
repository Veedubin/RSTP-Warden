"""DetectorSpec and factory for creating Detector instances from config."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from .base import Detector, NullDetector
from .class_filter import effective_classes
from .grid_mask import GridMask
from .roi import ROI, Mask
from .sensitivity import (
    apply_sensitivity_to_confidence,
    apply_sensitivity_to_motion,
    apply_sensitivity_to_nms,
)

logger = logging.getLogger(__name__)

DetectorType = Literal["motion", "person", "vehicle", "dnn", "custom"]


class DetectorSpec(BaseModel):
    """Configuration for a single detector attached to a camera.

    The `type` field determines which detector implementation is loaded.
    The `config` dict is passed to custom detectors. Type-specific
    fields (min_area, sensitivity, etc.) are optional and only used
    by some detector types.
    """

    type: DetectorType
    enabled: bool = True
    interval_seconds: float = 1.0
    config: dict[str, Any] = Field(default_factory=dict)
    # Type-specific fields (optional, only used by some types)
    min_area: int | None = None
    sensitivity: float | None = None
    min_confidence: float | None = None
    min_size: int | None = None
    scale_factor: float | None = None
    min_neighbors: int | None = None
    import_path: str | None = None  # for type=custom
    # ROI and privacy masks (Batch 4)
    roi: list[tuple[int, int]] | None = None
    masks: list[list[tuple[int, int]]] | None = None


def build_detector(spec: DetectorSpec, camera_name: str) -> Detector:
    """Build a Detector from a DetectorSpec.

    Lazy imports builtin detectors to keep base package light.
    Raises ValueError for unknown type or missing import_path on custom.
    Falls back to NullDetector if the builtin module is not yet implemented.
    """
    if not spec.enabled:
        logger.info("detector %s for %s is disabled, using NullDetector", spec.type, camera_name)
        return NullDetector()

    if spec.type == "motion":
        try:
            from .builtin.motion import MotionDetector

            return MotionDetector(
                min_area=spec.min_area or 500,
                sensitivity=spec.sensitivity or 0.5,
            )
        except ImportError:
            logger.warning(
                "MotionDetector not yet implemented for %s, using NullDetector", camera_name
            )
            return NullDetector()

    if spec.type == "person":
        try:
            from .builtin.person import PersonDetector

            return PersonDetector(
                min_confidence=spec.min_confidence or 0.5,
                scale_factor=spec.scale_factor or 1.1,
                min_neighbors=spec.min_neighbors or 3,
            )
        except ImportError:
            logger.warning(
                "PersonDetector not yet implemented for %s, using NullDetector", camera_name
            )
            return NullDetector()

    if spec.type == "vehicle":
        try:
            from .builtin.vehicle import VehicleDetector

            return VehicleDetector(
                min_confidence=spec.min_confidence or 0.5,
            )
        except ImportError:
            logger.warning(
                "VehicleDetector not yet implemented for %s, using NullDetector", camera_name
            )
            return NullDetector()

    if spec.type == "dnn":
        try:
            from .builtin.dnn import DNNDetector

            config = spec.config or {}
            allowed_classes = config.get("classes", None)
            if allowed_classes is not None:
                allowed_classes = list(allowed_classes)

            return DNNDetector(
                model_path=config.get("model_path", None),
                config_path=config.get("config_path", None),
                names_path=config.get("names_path", None),
                confidence=float(config.get("confidence_threshold", 0.5)),
                nms_threshold=float(config.get("nms_threshold", 0.4)),
                allowed_classes=allowed_classes,
                input_width=int(config.get("input_width", 416)),
                input_height=int(config.get("input_height", 416)),
            )
        except ImportError:
            logger.warning(
                "DNNDetector not yet implemented for %s, using NullDetector", camera_name
            )
            return NullDetector()

    if spec.type == "custom":
        if not spec.import_path:
            raise ValueError(f"custom detector requires import_path: {spec}")
        return _build_custom_detector(spec, camera_name)

    raise ValueError(f"unknown detector type: {spec.type}")


def _build_custom_detector(spec: DetectorSpec, camera_name: str) -> Detector:
    """Dynamically import and instantiate a custom detector."""
    assert spec.import_path is not None  # guaranteed by caller
    try:
        module_path, _, class_name = spec.import_path.rpartition(":")
        if not module_path:
            module_path, _, class_name = spec.import_path.rpartition(".")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instance = cls(**spec.config)
        # Verify it satisfies the Detector protocol
        _validate_detector(instance)
        return instance  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning(
            "failed to load custom detector %s for %s: %s, using NullDetector",
            spec.import_path,
            camera_name,
            e,
        )
        return NullDetector()


def _validate_detector(obj: Any) -> None:
    """Verify an object satisfies the Detector protocol."""
    for attr in ("name", "kind", "setup", "process", "teardown"):
        if not hasattr(obj, attr):
            raise ValueError(f"custom detector missing attribute: {attr}")


def build_roi(spec: DetectorSpec) -> ROI | None:
    """Build an ROI from a DetectorSpec, or return None if no ROI configured."""
    if spec.roi is None:
        return None
    return ROI(polygon=spec.roi)


def build_masks(spec: DetectorSpec) -> list[Mask]:
    """Build a list of Mask objects from a DetectorSpec, or empty list if none."""
    if spec.masks is None:
        return []
    return [Mask(polygon=m) for m in spec.masks]


# ---------------------------------------------------------------------------
# Sprint 6: build_detectors_for_camera
# ---------------------------------------------------------------------------

# Use TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import CameraConfig


@dataclass
class CameraDetectorBundle:
    """Aggregated result of building detectors for a camera.

    Contains the detector instances, masks, ROI, and grid masks
    needed to create a DetectorRunner.
    """

    detectors: list[Detector] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    roi: ROI | None = None
    grid_masks: list[GridMask] = field(default_factory=list)


def build_detector_with_sensitivity(
    spec: DetectorSpec,
    camera_name: str,
    camera_sensitivity: float = 50.0,
    camera_detect_classes: list[str] | None = None,
) -> Detector:
    """Build a Detector from a DetectorSpec, applying camera-level sensitivity and class filter.

    This extends build_detector() with:
    - Camera-level sensitivity mapping to per-detector parameters.
      Spec-level overrides (spec.sensitivity, spec.min_confidence) take precedence.
    - Camera-level detect_classes intersection with detector's allowed_classes (DNN only).

    Args:
        spec: DetectorSpec for the detector to build.
        camera_name: Name of the camera (for logging).
        camera_sensitivity: Camera-level sensitivity (0-100 scale).
        camera_detect_classes: Camera-level class filter, or None.

    Returns:
        A Detector instance with sensitivity and class filter applied.
    """
    if not spec.enabled:
        logger.info("detector %s for %s is disabled, using NullDetector", spec.type, camera_name)
        return NullDetector()

    if spec.type == "motion":
        try:
            from .builtin.motion import MotionDetector

            # Spec-level sensitivity takes precedence over camera-level.
            # If spec.sensitivity is set, use it directly.
            # Otherwise, map camera_sensitivity to varThreshold.
            if spec.sensitivity is not None:
                # Spec explicitly sets sensitivity -- use as-is.
                return MotionDetector(
                    min_area=spec.min_area or 500,
                    sensitivity=spec.sensitivity,
                )
            # Map camera sensitivity (0-100) to MotionDetector varThreshold.
            var_threshold = apply_sensitivity_to_motion(camera_sensitivity)
            return MotionDetector(
                min_area=spec.min_area or 500,
                sensitivity=0.5,  # Placeholder; var_threshold overrides it.
                var_threshold=var_threshold,
            )
        except ImportError:
            logger.warning(
                "MotionDetector not yet implemented for %s, using NullDetector", camera_name
            )
            return NullDetector()

    if spec.type == "person":
        try:
            from .builtin.person import PersonDetector

            # Spec-level min_confidence takes precedence.
            conf = (
                spec.min_confidence
                if spec.min_confidence is not None
                else apply_sensitivity_to_confidence(camera_sensitivity)
            )
            return PersonDetector(
                min_confidence=conf,
                scale_factor=spec.scale_factor or 1.1,
                min_neighbors=spec.min_neighbors or 3,
            )
        except ImportError:
            logger.warning(
                "PersonDetector not yet implemented for %s, using NullDetector", camera_name
            )
            return NullDetector()

    if spec.type == "vehicle":
        try:
            from .builtin.vehicle import VehicleDetector

            # Spec-level min_confidence takes precedence.
            conf = (
                spec.min_confidence
                if spec.min_confidence is not None
                else apply_sensitivity_to_confidence(camera_sensitivity)
            )
            return VehicleDetector(
                min_confidence=conf,
            )
        except ImportError:
            logger.warning(
                "VehicleDetector not yet implemented for %s, using NullDetector", camera_name
            )
            return NullDetector()

    if spec.type == "dnn":
        try:
            from .builtin.dnn import DNNDetector

            config = spec.config or {}

            # Spec-level confidence/nms take precedence.
            spec_confidence = config.get("confidence_threshold", None)
            if spec_confidence is not None:
                confidence = float(spec_confidence)
            else:
                confidence = apply_sensitivity_to_confidence(camera_sensitivity)

            spec_nms = config.get("nms_threshold", None)
            if spec_nms is not None:
                nms = float(spec_nms)
            else:
                nms = apply_sensitivity_to_nms(camera_sensitivity)

            # Compute effective allowed_classes from intersection.
            spec_classes = config.get("classes", None)
            if spec_classes is not None:
                spec_classes = list(spec_classes)
            effective = effective_classes(camera_detect_classes, spec_classes)

            # Log warning if intersection is empty.
            if effective is not None and len(effective) == 0:
                logger.warning(
                    "detect_classes intersection is empty for camera %s; "
                    "no detections will be reported for detector type=%s",
                    camera_name,
                    spec.type,
                )

            return DNNDetector(
                model_path=config.get("model_path", None),
                config_path=config.get("config_path", None),
                names_path=config.get("names_path", None),
                confidence=confidence,
                nms_threshold=nms,
                allowed_classes=effective,
                input_width=int(config.get("input_width", 416)),
                input_height=int(config.get("input_height", 416)),
            )
        except ImportError:
            logger.warning(
                "DNNDetector not yet implemented for %s, using NullDetector", camera_name
            )
            return NullDetector()

    if spec.type == "custom":
        if not spec.import_path:
            raise ValueError(f"custom detector requires import_path: {spec}")
        return _build_custom_detector(spec, camera_name)

    raise ValueError(f"unknown detector type: {spec.type}")


def build_grid_masks_from_config(
    zones: list[Any],  # list[GridZoneConfig]
) -> list[GridMask]:
    """Build GridMask objects from camera zone config.

    Args:
        zones: List of GridZoneConfig objects from CameraConfig.zones.

    Returns:
        List of GridMask objects (only enabled zones).
    """
    result: list[GridMask] = []
    for zc in zones:
        if not zc.enabled:
            continue
        gm = GridMask(
            name=zc.name,
            grid_cols=zc.grid_cols,
            grid_rows=zc.grid_rows,
            blocked_cells=set(zc.blocked_cells),
            frame_width=zc.frame_width,
            frame_height=zc.frame_height,
        )
        result.append(gm)
    return result


def build_detectors_for_camera(
    camera_cfg: CameraConfig,
    base_specs: list[DetectorSpec],
) -> CameraDetectorBundle:
    """Build all detectors, masks, ROI, and grid masks for a single camera.

    Applies camera-level sensitivity and detect_classes to each detector,
    skips disabled specs, and builds GridMasks from camera zone config.

    Args:
        camera_cfg: The camera configuration with sensitivity, detect_classes, zones.
        base_specs: The list of DetectorSpec from the camera config.

    Returns:
        A CameraDetectorBundle with detectors, masks, roi, and grid_masks.
    """
    # Filter out disabled specs.
    enabled_specs = [s for s in base_specs if s.enabled]
    if not enabled_specs:
        return CameraDetectorBundle()

    # Build individual detectors with sensitivity and class filter applied.
    detectors: list[Detector] = []
    for spec in enabled_specs:
        try:
            det = build_detector_with_sensitivity(
                spec=spec,
                camera_name=camera_cfg.name,
                camera_sensitivity=camera_cfg.sensitivity,
                camera_detect_classes=camera_cfg.detect_classes,
            )
            detectors.append(det)
        except Exception:
            logger.warning(
                "failed to build detector type=%s for camera=%s",
                spec.type,
                camera_cfg.name,
                exc_info=True,
            )

    if not detectors:
        return CameraDetectorBundle()

    # Build ROI and masks from specs.
    runner_roi: ROI | None = None
    runner_masks: list[Mask] = []
    for spec in enabled_specs:
        roi = build_roi(spec)
        masks = build_masks(spec)
        if roi is not None:
            runner_roi = roi
        if masks:
            runner_masks.extend(masks)

    # Build grid masks from camera zones.
    grid_masks = build_grid_masks_from_config(camera_cfg.zones)

    return CameraDetectorBundle(
        detectors=detectors,
        masks=runner_masks,
        roi=runner_roi,
        grid_masks=grid_masks,
    )


__all__ = [
    "DetectorSpec",
    "DetectorType",
    "build_detector",
    "build_detector_with_sensitivity",
    "build_detectors_for_camera",
    "build_grid_masks_from_config",
    "build_masks",
    "build_roi",
    "CameraDetectorBundle",
]

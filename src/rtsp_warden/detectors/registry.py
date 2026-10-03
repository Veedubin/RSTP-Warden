"""DetectorSpec and factory for creating Detector instances from config."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    field_validator,
    model_validator,
)

from ..deprecations import warn_once
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

DetectorType = Literal["motion", "person", "vehicle", "dnn", "custom", "onnx"]
DetectorDevice = Literal["auto", "cuda", "cpu"]

# Kept for one release; each logs a one-time deprecation warning naming `onnx`.
LEGACY_DETECTOR_TYPES: frozenset[str] = frozenset({"person", "vehicle", "dnn"})


class DetectorSpec(BaseModel):
    """Configuration for a single detector attached to a camera.

    The `type` field determines which detector implementation is loaded.
    The `config` dict is passed to custom detectors. Type-specific
    fields (min_area, sensitivity, etc.) are optional and only used
    by some detector types.
    """

    type: DetectorType
    enabled: bool = True
    # Deprecated: `_convert_interval` turns it into `fps`, so after validation it
    # is always None. Excluded from dumps so round trips never re-add it.
    interval_seconds: float | None = Field(default=None, exclude=True)
    fps: float | None = None  # None = the camera's detect_fps
    model: str | None = None  # onnx: model registry name; None = "yolox-s"
    device: DetectorDevice = "auto"  # onnx: execution provider choice
    events: bool | None = None  # motion: write events; None = CameraConfig decides
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

    # True when `fps` was derived from a deprecated `interval_seconds`. Lets
    # CameraConfig clamp such a value to detect_fps instead of rejecting it.
    _fps_from_interval: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def _convert_interval(
        cls, data: Any, handler: ModelWrapValidatorHandler[DetectorSpec]
    ) -> DetectorSpec:
        """Accept the deprecated `interval_seconds` key and convert it to `fps`.

        `fps = 1 / interval_seconds` when `fps` is not set; an explicit `fps`
        wins. Either way the key is dropped and a one-time warning is logged.
        """
        converted = False
        if isinstance(data, dict) and data.get("interval_seconds") is not None:
            data = dict(data)
            raw = data.pop("interval_seconds")
            try:
                interval = float(raw)
            except (TypeError, ValueError):
                raise ValueError("interval_seconds must be a number") from None
            if not interval > 0:
                raise ValueError("interval_seconds must be > 0")
            det_type = data.get("type")
            key = f"interval_seconds:{det_type}:{interval:g}"
            if data.get("fps") is None:
                data["fps"] = 1.0 / interval
                converted = True
                warn_once(
                    key,
                    "detectors[].interval_seconds is deprecated and will be removed in the "
                    "next release; use fps (type=%s: interval_seconds %g became fps %g)",
                    det_type,
                    interval,
                    data["fps"],
                )
            else:
                warn_once(
                    key,
                    "detectors[].interval_seconds is deprecated and is ignored because fps "
                    "is also set (type=%s); remove interval_seconds",
                    det_type,
                )
        spec = handler(data)
        if converted:
            spec._fps_from_interval = True
        return spec

    @field_validator("type")
    @classmethod
    def _warn_legacy(cls, v: str) -> str:
        if v in LEGACY_DETECTOR_TYPES:
            warn_once(
                f"detector-type:{v}",
                "detector type %r is deprecated and will be removed in the next release; "
                "use type: onnx (model: yolox-s)",
                v,
            )
        return v

    @field_validator("fps")
    @classmethod
    def _fps_positive(cls, v: float | None) -> float | None:
        if v is not None and not v > 0:
            raise ValueError("fps must be > 0")
        return v


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

    if spec.type == "onnx":
        return _build_onnx_detector(
            spec, camera_sensitivity=50.0, camera_detect_classes=None, models_dir=None
        )

    if spec.type == "custom":
        if not spec.import_path:
            raise ValueError(f"custom detector requires import_path: {spec}")
        return _build_custom_detector(spec, camera_name)

    raise ValueError(f"unknown detector type: {spec.type}")


DEFAULT_ONNX_MODEL = "yolox-s"


def _default_models_dir() -> Path:
    """``runtime.models_dir`` when the caller passed none (tests, legacy callers)."""
    from ..config import RuntimeConfig

    return RuntimeConfig().models_dir


def _build_onnx_detector(
    spec: DetectorSpec,
    *,
    camera_sensitivity: float,
    camera_detect_classes: list[str] | None,
    models_dir: Path | None,
) -> Detector:
    """Build an OnnxDetector from a registry descriptor (no model loading here).

    ``model`` defaults to ``yolox-s``. ``min_confidence`` comes from the spec, else
    from the camera sensitivity (0.5 at the default sensitivity of 50). The camera's
    ``detect_classes`` filters the model labels (None = every label). Raises
    ``ModelNotFound`` for an unknown model name.
    """
    from .builtin.onnx import OnnxDetector
    from .model_registry import load_descriptor

    resolved_dir = models_dir if models_dir is not None else _default_models_dir()
    descriptor = load_descriptor(spec.model or DEFAULT_ONNX_MODEL, resolved_dir)
    min_confidence = (
        spec.min_confidence
        if spec.min_confidence is not None
        else apply_sensitivity_to_confidence(camera_sensitivity)
    )
    classes = list(camera_detect_classes) if camera_detect_classes is not None else None
    return OnnxDetector(
        descriptor=descriptor,
        models_dir=resolved_dir,
        device=spec.device,
        min_confidence=min_confidence,
        classes=classes,
    )


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


@dataclass(slots=True)
class DetectorSlot:
    """A built detector plus what the runner needs to schedule and route its output.

    ``index`` is the spec's position in ``CameraConfig.detectors`` (ruling R15), so UI
    rows and runtime status line up even when a disabled or failing spec is skipped.
    """

    index: int
    spec: DetectorSpec
    detector: Detector
    fps: float  # effective rate: spec.fps, else the camera's detect_fps
    tracked: bool  # onnx output goes to the Tracker, not to result_sinks
    motion_events: bool  # motion spec whose events resolve true (ruling R11)
    input_width: int | None  # model input width (onnx), None otherwise


def _make_slot(
    camera_cfg: CameraConfig, index: int, spec: DetectorSpec, det: Detector
) -> DetectorSlot:
    """Describe one successfully built detector (called in lock-step with the append)."""
    width = getattr(det, "input_width", None)
    return DetectorSlot(
        index=index,
        spec=spec,
        detector=det,
        fps=float(camera_cfg.effective_fps(spec)),
        tracked=spec.type == "onnx",
        motion_events=spec.type == "motion" and camera_cfg.motion_events_enabled(spec),
        input_width=int(width) if isinstance(width, int) else None,
    )


@dataclass
class CameraDetectorBundle:
    """Aggregated result of building detectors for a camera.

    Contains the detector instances, masks, ROI, and grid masks
    needed to create a DetectorRunner. ``grid_masks`` holds the camera's
    enabled ``ignore`` zones (they filter detections); ``area_masks`` holds
    its enabled ``area`` zones as ``(name, mask)`` in config order (they
    name regions for events and rules and never filter).
    """

    detectors: list[Detector] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    roi: ROI | None = None
    grid_masks: list[GridMask] = field(default_factory=list)
    area_masks: list[tuple[str, GridMask]] = field(default_factory=list)
    # Parallel to ``detectors`` (same length and order); empty for an empty bundle.
    slots: list[DetectorSlot] = field(default_factory=list)


def build_detector_with_sensitivity(
    spec: DetectorSpec,
    camera_name: str,
    camera_sensitivity: float = 50.0,
    camera_detect_classes: list[str] | None = None,
    *,
    models_dir: Path | None = None,
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

    if spec.type == "onnx":
        return _build_onnx_detector(
            spec,
            camera_sensitivity=camera_sensitivity,
            camera_detect_classes=camera_detect_classes,
            models_dir=models_dir,
        )

    if spec.type == "custom":
        if not spec.import_path:
            raise ValueError(f"custom detector requires import_path: {spec}")
        return _build_custom_detector(spec, camera_name)

    raise ValueError(f"unknown detector type: {spec.type}")


def build_grid_masks_from_config(
    zones: list[Any],  # list[GridZoneConfig]
) -> list[GridMask]:
    """Build the filtering GridMasks from camera zone config.

    Args:
        zones: List of GridZoneConfig objects from CameraConfig.zones.

    Returns:
        List of GridMask objects (enabled ``ignore`` zones only, config order).
    """
    return [GridMask.from_zone(zc) for zc in zones if zc.enabled and zc.kind == "ignore"]


def build_area_masks_from_config(
    zones: list[Any],  # list[GridZoneConfig]
) -> list[tuple[str, GridMask]]:
    """Build the named-area masks from camera zone config.

    Args:
        zones: List of GridZoneConfig objects from CameraConfig.zones.

    Returns:
        ``(zone name, GridMask)`` for each enabled ``area`` zone, in config order
        (the order `zone_for_point` resolves overlaps by).
    """
    return [(zc.name, GridMask.from_zone(zc)) for zc in zones if zc.enabled and zc.kind == "area"]


def build_detectors_for_camera(
    camera_cfg: CameraConfig,
    base_specs: list[DetectorSpec],
    *,
    models_dir: Path | None = None,
) -> CameraDetectorBundle:
    """Build all detectors, masks, ROI, and grid masks for a single camera.

    Applies camera-level sensitivity and detect_classes to each detector,
    skips disabled specs, and builds GridMasks from camera zone config.

    Args:
        camera_cfg: The camera configuration with sensitivity, detect_classes, zones.
        base_specs: The list of DetectorSpec from the camera config.
        models_dir: Where ``onnx`` models live (``runtime.models_dir``); None uses
            the default from ``RuntimeConfig``.

    Returns:
        A CameraDetectorBundle with detectors, masks, roi, and grid_masks.
    """
    # Filter out disabled specs.
    enabled_specs = [s for s in base_specs if s.enabled]
    if not enabled_specs:
        return CameraDetectorBundle()

    # Build individual detectors with sensitivity and class filter applied.
    detectors: list[Detector] = []
    slots: list[DetectorSlot] = []
    for spec_index, spec in enumerate(base_specs):
        if not spec.enabled:
            continue
        try:
            det = build_detector_with_sensitivity(
                spec=spec,
                camera_name=camera_cfg.name,
                camera_sensitivity=camera_cfg.sensitivity,
                camera_detect_classes=camera_cfg.detect_classes,
                models_dir=models_dir,
            )
            slot = _make_slot(camera_cfg, spec_index, spec, det)
            detectors.append(det)
            slots.append(slot)
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

    # Build grid masks from camera zones: ignore zones filter, area zones only name regions.
    grid_masks = build_grid_masks_from_config(camera_cfg.zones)
    area_masks = build_area_masks_from_config(camera_cfg.zones)

    return CameraDetectorBundle(
        detectors=detectors,
        slots=slots,
        masks=runner_masks,
        roi=runner_roi,
        grid_masks=grid_masks,
        area_masks=area_masks,
    )


__all__ = [
    "DEFAULT_ONNX_MODEL",
    "DetectorSpec",
    "DetectorType",
    "build_detector",
    "build_detector_with_sensitivity",
    "build_area_masks_from_config",
    "build_detectors_for_camera",
    "build_grid_masks_from_config",
    "build_masks",
    "build_roi",
    "CameraDetectorBundle",
    "DetectorSlot",
]

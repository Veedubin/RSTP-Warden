"""DetectorSpec and factory for creating Detector instances from config."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from .base import Detector, NullDetector
from .roi import ROI, Mask

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


__all__ = ["DetectorSpec", "DetectorType", "build_detector", "build_masks", "build_roi"]

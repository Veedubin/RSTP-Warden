"""Detector framework for rtsp-warden.

Architecture overview:
- Detector (Protocol): The interface all detectors implement.
- Detection (dataclass): A single detection result (kind, confidence, bbox, etc.).
- DetectorRunner: A FrameConsumer that decodes JPEG frames, applies masks,
  and dispatches to Detector instances on worker threads.
- EventSink: Receives detection results and writes Event rows to the database.
- DetectorSpec (Pydantic model): Config schema for declaring detectors in YAML.
- build_detector (factory): Creates Detector instances from DetectorSpec.
- build_detectors_for_camera: Creates all detectors for a camera with
  sensitivity, class filtering, and grid masks applied.
- GridMask: Grid-based zone mask for suppressing detections in blocked cells.
- sensitivity: Module for mapping camera sensitivity (0-100) to per-detector params.
- class_filter: Module for intersecting camera detect_classes with detector classes.

Flow:
  FrameTapDispatcher -> DetectorRunner.on_frame()
      -> worker thread: decode JPEG -> apply_masks -> detector.process()
      -> grid_mask filter -> ROI filter -> result_sinks: EventSink(camera, stream, detections)
"""

from __future__ import annotations

from .base import Detection, Detector, NullDetector, apply_masks
from .class_filter import effective_classes
from .grid_mask import GridMask
from .registry import (
    CameraDetectorBundle,
    DetectorSpec,
    build_detector,
    build_detector_with_sensitivity,
    build_detectors_for_camera,
    build_grid_masks_from_config,
    build_masks,
    build_roi,
)
from .roi import ROI, Mask, filter_by_roi
from .runner import DetectorRunner
from .sensitivity import (
    apply_sensitivity_to_confidence,
    apply_sensitivity_to_motion,
    apply_sensitivity_to_nms,
    motion_detector_sensitivity,
)
from .sinks import EventSink

__all__ = [
    "CameraDetectorBundle",
    "Detection",
    "Detector",
    "DetectorRunner",
    "DetectorSpec",
    "EventSink",
    "GridMask",
    "Mask",
    "NullDetector",
    "ROI",
    "apply_masks",
    "apply_sensitivity_to_confidence",
    "apply_sensitivity_to_motion",
    "apply_sensitivity_to_nms",
    "build_detector",
    "build_detector_with_sensitivity",
    "build_detectors_for_camera",
    "build_grid_masks_from_config",
    "build_masks",
    "build_roi",
    "effective_classes",
    "filter_by_roi",
    "motion_detector_sensitivity",
]

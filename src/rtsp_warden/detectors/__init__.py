"""Detector framework for rtsp-warden.

Architecture overview:
- Detector (Protocol): The interface all detectors implement.
- Detection (dataclass): A single detection result (kind, confidence, bbox, etc.).
- DetectorRunner: A FrameConsumer that decodes JPEG frames, applies masks,
  and dispatches to Detector instances on worker threads.
- EventSink: Receives detection results and writes Event rows to the database.
- DetectorSpec (Pydantic model): Config schema for declaring detectors in YAML.
- build_detector (factory): Creates Detector instances from DetectorSpec.

Flow:
  FrameTapDispatcher -> DetectorRunner.on_frame()
      -> worker thread: decode JPEG -> apply_masks -> detector.process()
      -> result_sinks: EventSink(camera, stream, detections)

The detector framework is wired into AppRuntime when cameras have detectors
configured and --detectors is enabled (default: True).
"""

from __future__ import annotations

from .base import Detection, Detector, NullDetector, apply_masks
from .registry import DetectorSpec, build_detector
from .roi import ROI, Mask, filter_by_roi
from .runner import DetectorRunner
from .sinks import EventSink

__all__ = [
    "Detection",
    "Detector",
    "DetectorRunner",
    "DetectorSpec",
    "EventSink",
    "Mask",
    "NullDetector",
    "ROI",
    "apply_masks",
    "build_detector",
    "filter_by_roi",
]

"""Builtin detector implementations for rtsp-warden.

This package contains the concrete detector classes that ship with
rtsp-warden. Each module implements the Detector protocol defined in
``rtsp_warden.detectors.base`` and is lazily imported by the registry
so that heavy dependencies (e.g. OpenCV) are only loaded when actually
needed.

Available detectors:

- ``motion`` -- MOG2 background-subtraction motion detector
  (``builtin.motion.MotionDetector``)
- ``person`` -- HOG + linear SVM person detector
  (``builtin.person.PersonDetector``)
- ``vehicle`` -- Haar cascade car detector
  (``builtin.vehicle.VehicleDetector``)
- ``custom`` -- Template/demo custom detector
  (``builtin.custom.DemoCustomDetector``)
"""

from __future__ import annotations

__all__: list[str] = []

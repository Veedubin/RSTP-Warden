"""YOLOv4-tiny DNN detector for vehicles and animals.

Uses OpenCV's ``cv2.dnn`` module to run YOLOv4-tiny inference on CPU.
Detects COCO classes: car, truck, bus, motorcycle, bicycle (vehicles)
and bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
(animals) by default. The ``allowed_classes`` filter can restrict or
extend this list.

Algorithm
---------
1. On ``setup()``, load the Darknet model (.cfg + .weights) and class
   names. If weights are not found locally, they are auto-downloaded
   to ``~/.cache/rtsp-warden/models/`` (see ``model_utils.ensure_model``).
2. For each frame:
   a. Create a blob via ``cv2.dnn.blobFromImage`` with 1/255.0 scale,
      416x416 input size, and RGB channel swap.
   b. Forward-pass through the network to get raw YOLO outputs.
   c. Parse detections: extract class scores, confidence threshold,
      and compute bounding boxes in original frame coordinates.
   d. Apply Non-Maximum Suppression (NMS) to deduplicate overlapping
      detections.
   e. Filter results by ``allowed_classes`` (if set).
   f. Return a list of ``Detection`` objects with class-specific
      ``kind`` (e.g. ``"car"``, ``"dog"``) and confidence scores.
3. On ``teardown()``, release the network reference.

Edge cases
----------
- Frames smaller than 8x8, empty, or None-like return ``[]``.
- If model setup fails (missing file, corrupt weights, OpenCV error),
  logs an error and ``process()`` returns ``[]`` without crashing.
- If ``allowed_classes`` is ``None``, all COCO classes above the
  confidence threshold are reported.
- If ``process()`` is called before ``setup()``, returns ``[]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..base import Detection
from .model_utils import (
    DEFAULT_DNN_CLASSES,
    YOLOV4_TINY_CFG,
    YOLOV4_TINY_NAMES,
    ensure_model,
    load_class_names,
)

logger = logging.getLogger(__name__)

# Minimum frame dimension to attempt detection
_MIN_FRAME_DIM = 8


@dataclass(slots=True)
class DNNDetector:
    """YOLOv4-tiny DNN detector for vehicles and animals.

    Uses OpenCV's DNN module for CPU-based inference.  Weights are
    auto-downloaded on first use; the .cfg and .names files are bundled.

    Args:
        model_path: Path to .weights file.  When ``None``, auto-downloaded
            to the cache directory.
        config_path: Path to .cfg file.  When ``None``, uses the bundled
            yolov4-tiny.cfg.
        names_path: Path to .names file.  When ``None``, uses the bundled
            coco.names.
        confidence: Minimum confidence threshold for detections (0.0-1.0).
        nms_threshold: Non-maximum suppression threshold (0.0-1.0).
            Higher values keep more overlapping detections.
        allowed_classes: List of COCO class names to report.  When
            ``None``, reports all detected classes.  Defaults to
            vehicle + animal classes.
        input_width: DNN input width in pixels.
        input_height: DNN input height in pixels.
        name: Detector name for protocol compliance.
        kind: Detection kind string (default ``"dnn"``).
    """

    model_path: str | None = None
    config_path: str | None = None
    names_path: str | None = None
    confidence: float = 0.5
    nms_threshold: float = 0.4
    allowed_classes: list[str] | None = None
    input_width: int = 416
    input_height: int = 416
    name: str = "dnn"
    kind: str = "dnn"
    _net: cv2.dnn.Net | None = field(default=None, init=False, repr=False)
    _output_layers: list[str] = field(default_factory=list, init=False, repr=False)
    _classes: list[str] = field(default_factory=list, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    # -- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        """Load the YOLO model and class names.

        Must be called once before :meth:`process`.  Downloads weights
        if not found in the cache directory.
        """
        try:
            # Resolve paths
            cfg_path = self.config_path or str(YOLOV4_TINY_CFG)
            weights_path = str(ensure_model(self.model_path))
            names_path = self.names_path or str(YOLOV4_TINY_NAMES)

            # Load class names
            self._classes = load_class_names(names_path)

            # Load network
            net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get output layer names
            layer_names = net.getLayerNames()
            unconnected = net.getUnconnectedOutLayers()
            self._output_layers = [layer_names[i - 1] for i in unconnected.flatten()]

            self._net = net
            self._loaded = True

            # Default allowed_classes to vehicle + animal classes
            if self.allowed_classes is None:
                self.allowed_classes = list(DEFAULT_DNN_CLASSES)

            logger.debug(
                "DNNDetector setup: cfg=%s weights=%s classes=%d "
                "confidence=%.2f nms=%.2f input=%dx%d",
                cfg_path,
                weights_path,
                len(self._classes),
                self.confidence,
                self.nms_threshold,
                self.input_width,
                self.input_height,
            )
        except Exception:
            logger.error("DNNDetector setup failed", exc_info=True)
            self._net = None
            self._loaded = False

    def teardown(self) -> None:
        """Release the network reference."""
        self._net = None
        self._loaded = False

    # -- frame processing ------------------------------------------------

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        """Process one BGR frame and return YOLO detections.

        Args:
            frame_bgr: Decoded BGR frame (HxWx3, uint8).
            ts_unix: Unix timestamp of the frame.

        Returns:
            List of :class:`Detection` instances for each detection
            above the confidence threshold and NMS filter.
        """
        if not self._loaded or self._net is None:
            logger.error("DNNDetector.process called before setup() or setup failed")
            return []

        # Guard: invalid / degenerate frames
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return []
            h, w = frame_bgr.shape[:2]
            if h < _MIN_FRAME_DIM or w < _MIN_FRAME_DIM:
                return []
        except Exception:
            return []

        # Create blob from input image
        try:
            blob = cv2.dnn.blobFromImage(
                frame_bgr,
                1.0 / 255.0,
                (self.input_width, self.input_height),
                swapRB=True,
                crop=False,
            )
        except cv2.error:
            logger.debug("DNNDetector: blobFromImage failed", exc_info=True)
            return []

        # Forward pass
        try:
            self._net.setInput(blob)
            outputs = self._net.forward(self._output_layers)
        except cv2.error:
            logger.debug("DNNDetector: forward pass failed", exc_info=True)
            return []

        # Parse YOLO outputs
        # Each output has shape (batch_size, num_detections, 5+num_classes).
        # OpenCV DNN with YOLO typically returns batch_size=1, so we squeeze
        # the batch dimension to get (num_detections, 5+num_classes).
        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            # Squeeze batch dimension: (1, N, 85) -> (N, 85)
            if output.ndim == 3:
                output = output.squeeze(0)
            for detection in output:
                if detection.shape[0] < 6:
                    continue
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf > self.confidence:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)
                    x = int(center_x - bw / 2)
                    y = int(center_y - bh / 2)
                    boxes.append([x, y, bw, bh])
                    confidences.append(conf)
                    class_ids.append(class_id)

        # Non-maximum suppression
        try:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold)
        except cv2.error:
            logger.debug("DNNDetector: NMSBoxes failed", exc_info=True)
            return []

        # Build results
        results: list[Detection] = []
        if len(indices) > 0:
            for i in indices.flatten():
                cls_id = class_ids[i]
                label = self._classes[cls_id] if cls_id < len(self._classes) else str(cls_id)
                # Filter by allowed classes
                if self.allowed_classes and label not in self.allowed_classes:
                    continue
                bx, by, bw, bh = boxes[i]
                results.append(
                    Detection(
                        kind=label,
                        confidence=confidences[i],
                        bbox=(bx, by, bx + bw, by + bh),
                        metadata={"class_id": cls_id, "class_name": label},
                        ts_unix=ts_unix,
                    )
                )

        return results


__all__ = ["DNNDetector"]

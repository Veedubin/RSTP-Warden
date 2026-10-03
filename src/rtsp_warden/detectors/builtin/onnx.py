"""ONNX Runtime object detector for YOLOX-family models.

The model comes from the model registry (``detectors/model_registry.py``): its
descriptor names the ``.onnx`` file, the labels file, the input size and the
``yolox`` post-processing. ``onnxruntime`` is imported when the model is loaded,
never at module import, so loading the config or running ``rtsp-warden status``
stays cheap.

Tensor contract (YOLOX ONNX exports from release 0.1.1rc0 on):

- input ``images``: float32 ``[1, 3, H, W]``, BGR, raw 0-255 values (no ``/255``,
  no mean/std), letterboxed onto the top-left of a canvas filled with 114.
- output ``output``: float32 ``[1, N, 5 + num_classes]`` of raw grid rows
  ``[x_off, y_off, log_w, log_h, obj, cls_0 ... cls_n]`` for strides 8, 16, 32
  (row-major per level, y outer). ``obj`` and ``cls`` are already sigmoided; a
  class score is ``obj * cls``.

Boxes go back to the frame by dividing by the letterbox ratio (no offset: the
padding is on the right and bottom) and are emitted as ``(x, y, w, h)`` in
frame pixels, the ``Detection.bbox`` contract.
"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from ..base import Detection
from ..model_registry import ModelDescriptor, ensure_model_file, load_labels, model_file_path

log = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
YOLOX_STRIDES: tuple[int, ...] = (8, 16, 32)
PAD_VALUE = 114
GPU_FIX_COMMAND = (
    "uv sync --extra gpu --no-install-package onnxruntime --reinstall-package onnxruntime-gpu"
)
_MIN_FRAME_DIM = 8
_MAX_ERROR_CHARS = 300
_MAX_LOG_WH = 10.0  # clamp before exp(): keeps garbage rows finite


def select_providers(device: str, available: Sequence[str]) -> list[str]:
    """Execution providers to request for ``device``, best first.

    ``cpu`` always gets CPU only. ``auto`` and ``cuda`` ask for CUDA first when the
    installed onnxruntime build lists it. CUDA is never requested when the build
    does not list it, which avoids onnxruntime's "not in available provider
    names" warning on the CPU package.
    """
    if device != "cpu" and CUDA_PROVIDER in available:
        return [CUDA_PROVIDER, CPU_PROVIDER]
    return [CPU_PROVIDER]


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Scale ``frame`` into ``size`` keeping its aspect ratio, YOLOX style.

    ``size`` is the descriptor's ``input_size``, ``(width, height)``. The frame is
    resized by ``ratio = min(height / h, width / w)`` and pasted at the top-left
    corner of a canvas filled with 114. Returns the NCHW float32 batch
    ``[1, 3, height, width]`` (BGR, 0-255) and ``ratio``.
    """
    in_w, in_h = int(size[0]), int(size[1])
    h, w = frame.shape[:2]
    ratio = min(in_h / h, in_w / w)
    new_w = max(1, min(in_w, int(w * ratio)))
    new_h = max(1, min(in_h, int(h * ratio)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((in_h, in_w, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    blob = canvas.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return np.ascontiguousarray(blob), ratio


@lru_cache(maxsize=8)
def _yolox_grid(in_w: int, in_h: int, strides: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Grid cell offsets ``(N, 2)`` and per-row strides ``(N, 1)`` for one input size."""
    cells: list[np.ndarray] = []
    level_strides: list[np.ndarray] = []
    for stride in strides:
        cols, rows = in_w // stride, in_h // stride
        xv, yv = np.meshgrid(np.arange(cols), np.arange(rows))
        cells.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        level_strides.append(np.full((cols * rows, 1), stride))
    grid = np.concatenate(cells).astype(np.float32)
    stride_col = np.concatenate(level_strides).astype(np.float32)
    return grid, stride_col


def decode_yolox(
    output: np.ndarray,
    input_size: tuple[int, int],
    strides: Sequence[int] = YOLOX_STRIDES,
) -> np.ndarray:
    """Decode raw YOLOX rows into ``(N, 6)`` rows ``x1, y1, x2, y2, score, class_id``.

    Coordinates are in letterboxed input pixels. ``score`` is the best ``obj * cls``
    of the row and ``class_id`` its class index (stored as float). Raises
    ``ValueError`` when the tensor does not fit ``input_size``.
    """
    preds = np.asarray(output, dtype=np.float32)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.ndim != 2 or preds.shape[1] < 6:
        raise ValueError(f"YOLOX output must be [1, N, 5 + classes], got {np.shape(output)}")
    in_w, in_h = int(input_size[0]), int(input_size[1])
    grid, stride_col = _yolox_grid(in_w, in_h, tuple(int(s) for s in strides))
    if preds.shape[0] != grid.shape[0]:
        raise ValueError(
            f"YOLOX output has {preds.shape[0]} rows; a {in_w}x{in_h} input needs {grid.shape[0]}"
        )
    centers = (preds[:, 0:2] + grid) * stride_col
    sizes = np.exp(np.minimum(preds[:, 2:4], _MAX_LOG_WH)) * stride_col
    class_scores = preds[:, 4:5] * preds[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    half = sizes / 2.0
    decoded = np.column_stack((centers - half, centers + half, scores, class_ids))
    return decoded.astype(np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression on xyxy boxes; returns kept indices, best first."""
    order = np.argsort(-scores, kind="stable")
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    areas = widths * heights
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        xx1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0.0, None) * np.clip(yy2 - yy1, 0.0, None)
        union = areas[best] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


_preload_lock = threading.Lock()
_preload_done = False


def _preload_cuda_libraries(ort: Any) -> None:
    """Load the pip CUDA/cuDNN libraries once per process (GPU package only)."""
    global _preload_done
    if getattr(ort, "package_name", "") != "onnxruntime-gpu":
        return
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    with _preload_lock:
        if _preload_done:
            return
        _preload_done = True
        try:
            preload()
        except Exception:
            log.warning("onnxruntime.preload_dlls() failed; CUDA may be unavailable", exc_info=True)


_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _model_file_lock(path: Path) -> threading.Lock:
    """One lock per model file, so two cameras never download the same file at once."""
    with _file_locks_guard:
        return _file_locks.setdefault(str(path), threading.Lock())


def _both_onnxruntime_packages() -> bool:
    """True when the CPU and the GPU onnxruntime distributions share one environment."""
    try:
        importlib.metadata.version("onnxruntime")
        importlib.metadata.version("onnxruntime-gpu")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


@dataclass
class OnnxDetector:
    """``Detector`` backed by an ONNX Runtime session running a YOLOX model.

    Nothing here raises on a model problem. ``setup()`` loads the session only
    when the model file is already on disk; a missing file is downloaded by the
    first ``process()`` call, on the detector's worker thread, so ``serve`` and a
    hot reload never wait on the network. A failed download, a hash mismatch or
    a broken session sets ``error`` and ``process()`` returns no detections; the
    load is retried every ``retry_interval_s`` seconds of frame time, so a home
    lab that was offline at start recovers without a restart.
    """

    descriptor: ModelDescriptor
    models_dir: Path
    device: Literal["auto", "cuda", "cpu"] = "auto"
    min_confidence: float = 0.5
    classes: list[str] | None = None  # None = every label of the model
    nms_iou: float = 0.45
    name: str = "onnx"
    kind: str = "onnx"
    provider: str | None = None  # set by setup(): the provider the session really uses
    fallback_warning: str | None = None  # set when device == "cuda" but CUDA is not used
    error: str | None = None  # set when the model could not be loaded
    labels: list[str] = field(default_factory=list)  # loaded by setup()
    retry_interval_s: float = 300.0
    _session: Any = field(default=None, init=False, repr=False)
    _input_name: str = field(default="images", init=False, repr=False)
    _attempted: bool = field(default=False, init=False, repr=False)
    _retry_at: float | None = field(default=None, init=False, repr=False)
    # Guards the state fields only; never held across a download or a session load, so
    # teardown() (a hot reload on a web thread) never waits on the network.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _loading: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def input_width(self) -> int:
        """Model input width; the frame tap is scaled to the widest enabled model."""
        return int(self.descriptor.input_size[0])

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def setup(self) -> None:
        with self._lock:
            self._closed = False
            self._session = None
            self._attempted = False
            self._retry_at = None
            self.provider = None
            self.fallback_warning = None
            self.error = None
            on_disk = model_file_path(self.descriptor, self.models_dir).is_file()
            if on_disk:
                self._attempted = True
                self._loading = True
        if on_disk:
            self._load()
        else:
            log.info(
                "onnx detector %s: model %s is not in %s yet; fetching it on the first frame",
                self.name,
                self.descriptor.name,
                self.models_dir,
            )

    def teardown(self) -> None:
        """Drop the session. Returns at once even while a load is running (it is discarded)."""
        with self._lock:
            self._closed = True
            self._session = None
            self._retry_at = None

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        session = self._session
        if session is None:
            session = self._reload_if_due(ts_unix)
            if session is None:
                return []
        if self.classes is not None and not self.classes:
            return []
        if (
            frame_bgr is None
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or min(frame_bgr.shape[:2]) < _MIN_FRAME_DIM
        ):
            return []
        blob, ratio = letterbox(frame_bgr, self.descriptor.input_size)
        raw = session.run(None, {self._input_name: blob})[0]
        rows = decode_yolox(self._filter_classes(np.asarray(raw)), self.descriptor.input_size)
        frame_h, frame_w = frame_bgr.shape[:2]
        return self._to_detections(rows, ratio, frame_w, frame_h, ts_unix)

    # -- loading --------------------------------------------------------------

    def _reload_if_due(self, ts_unix: float) -> Any:
        """Load on the first frame, or retry a failed load once ``_retry_at`` has passed.

        The decision is made under ``_lock``; the load itself runs without it.
        """
        with self._lock:
            if self._session is not None:
                return self._session
            if self._closed or self._loading:
                return None
            if self._attempted:
                if self._retry_at is None:  # setup() failed: count the interval from now
                    self._retry_at = ts_unix + self.retry_interval_s
                    return None
                if ts_unix < self._retry_at:
                    return None
            self._attempted = True
            self._loading = True
            self._retry_at = None
            self.provider = None
            self.fallback_warning = None
            self.error = None
        self._load()
        with self._lock:
            if self._session is None and not self._closed:
                self._retry_at = ts_unix + self.retry_interval_s
            return self._session

    def _load(self) -> None:
        """Load labels, model file and session, then install them. Never raises.

        Called with ``_loading`` set and WITHOUT ``_lock``: the download (up to a minute
        per read when offline) and the session creation must never block ``teardown()``.
        A session that finishes after ``teardown()`` is dropped. Failures set ``error``.
        """
        try:
            self._load_unlocked()
        finally:
            with self._lock:
                self._loading = False

    def _load_unlocked(self) -> None:
        model = self.descriptor.name
        try:
            labels = load_labels(self.descriptor)
            with _model_file_lock(model_file_path(self.descriptor, self.models_dir)):
                model_path = ensure_model_file(self.descriptor, self.models_dir)
        except Exception as exc:  # offline download, hash mismatch, missing labels
            self._fail(f"model {model!r} unavailable: {exc}")
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            self._fail(f"onnxruntime is not installed: {exc}")
            return
        if self.device != "cpu":
            _preload_cuda_libraries(ort)
        requested = select_providers(self.device, list(ort.get_available_providers()))
        try:
            session = ort.InferenceSession(str(model_path), providers=requested)
        except Exception as exc:
            self._fail(f"model {model!r} failed to load: {exc}")
            return
        input_name = session.get_inputs()[0].name
        provider = str(session.get_providers()[0])
        fallback: str | None = None
        if self.device == "cuda" and provider != CUDA_PROVIDER:
            fallback = f"CUDA requested but unavailable; running on {provider}"
        with self._lock:
            if self._closed:
                return  # torn down while loading (hot reload): drop the session
            self.labels = labels
            self._input_name = input_name
            self.provider = provider
            self.fallback_warning = fallback
            self.error = None
            self._session = session
        if self.device != "cpu" and provider != CUDA_PROVIDER and _both_onnxruntime_packages():
            log.warning(
                "onnxruntime and onnxruntime-gpu are both installed, so CUDA is not used; run: %s",
                GPU_FIX_COMMAND,
            )
        if fallback is not None:
            log.warning("onnx detector %s (model %s): %s", self.name, model, fallback)
        if self.classes is not None and not self.classes:
            log.warning("onnx detector %s: detect_classes is empty; it reports nothing", self.name)
        log.info(
            "onnx detector %s (model %s, device %s) provider: %s",
            self.name,
            model,
            self.device,
            provider,
        )

    def _fail(self, message: str) -> None:
        with self._lock:
            self.error = message[:_MAX_ERROR_CHARS]
        log.error("onnx detector %s: %s", self.name, message[:_MAX_ERROR_CHARS])

    # -- post-processing ------------------------------------------------------

    def _label(self, class_id: int) -> str:
        return self.labels[class_id] if 0 <= class_id < len(self.labels) else str(class_id)

    def _filter_classes(self, raw: np.ndarray) -> np.ndarray:
        """Zero the scores of unwanted classes before the best class of a row is picked."""
        if self.classes is None:
            return raw
        wanted = set(self.classes)
        num_classes = raw.shape[-1] - 5
        drop = np.array([self._label(i) not in wanted for i in range(num_classes)], dtype=bool)
        if not drop.any():
            return raw
        filtered = np.array(raw, dtype=np.float32, copy=True)
        filtered[..., 5:][..., drop] = 0.0
        return filtered

    def _to_detections(
        self, rows: np.ndarray, ratio: float, frame_w: int, frame_h: int, ts_unix: float
    ) -> list[Detection]:
        rows = rows[(rows[:, 4] >= self.min_confidence) & (rows[:, 4] > 0.0)]
        if rows.shape[0] == 0:
            return []
        boxes = rows[:, 0:4].astype(np.float64) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, frame_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, frame_h)
        scores = rows[:, 4].astype(np.float64)
        class_ids = rows[:, 5].astype(np.int64)
        picked: list[int] = []
        for class_id in np.unique(class_ids):
            members = np.flatnonzero(class_ids == class_id)
            kept = _nms(boxes[members], scores[members], self.nms_iou)
            picked.extend(int(members[k]) for k in kept)
        picked.sort(key=lambda i: (-scores[i], i))
        results: list[Detection] = []
        for i in picked:
            x1, y1, x2, y2 = (int(round(float(v))) for v in boxes[i])
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            class_id = int(class_ids[i])
            label = self._label(class_id)
            results.append(
                Detection(
                    kind=label,
                    confidence=float(scores[i]),
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    metadata={
                        "class_id": class_id,
                        "class_name": label,
                        "model": self.descriptor.name,
                    },
                    ts_unix=float(ts_unix),
                )
            )
        return results


__all__ = [
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "OnnxDetector",
    "decode_yolox",
    "letterbox",
    "select_providers",
]

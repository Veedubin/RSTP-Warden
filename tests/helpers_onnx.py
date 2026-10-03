"""Tiny ONNX models for offline detector tests (no real weights, no network).

``make_constant_yolox_model`` writes a graph with the YOLOX input ``images``
``[1, 3, H, W]`` whose only node is a ``Constant`` feeding the output ``output``.
onnxruntime still checks the fed input's dtype and shape, so a wrong letterbox
(NHWC, float64, wrong size) fails the test, while the output rows are fixed by
the test.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper, numpy_helper

YOLOX_STRIDES = (8, 16, 32)


def yolox_anchor_count(input_size: tuple[int, int]) -> int:
    """Rows of a YOLOX output for ``input_size`` ``(width, height)``: 84 for 64x64."""
    width, height = input_size
    return sum((width // s) * (height // s) for s in YOLOX_STRIDES)


def yolox_output(
    input_size: tuple[int, int], num_classes: int, rows: Mapping[int, Sequence[float]]
) -> np.ndarray:
    """A float32 ``[1, N, 5 + num_classes]`` tensor, zero except the given rows."""
    out = np.zeros((1, yolox_anchor_count(input_size), 5 + num_classes), dtype=np.float32)
    for index, values in rows.items():
        out[0, index, :] = np.asarray(values, dtype=np.float32)
    return out


def make_constant_yolox_model(path: Path, input_size: tuple[int, int], rows: np.ndarray) -> Path:
    """Write an ONNX model: input ``images`` float32 [1, 3, H, W], output ``output`` = ``rows``."""
    width, height = input_size
    value = numpy_helper.from_array(rows.astype(np.float32), name="output_value")
    node = helper.make_node("Constant", inputs=[], outputs=["output"], value=value)
    graph = helper.make_graph(
        [node],
        "constant_yolox",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, height, width])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(rows.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10  # newer onnx writes IR versions older onnxruntime builds refuse
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def write_model_dir(
    models_dir: Path,
    rows: np.ndarray,
    *,
    name: str = "tiny",
    labels: Sequence[str] = ("person", "car"),
    input_size: tuple[int, int] = (64, 64),
    sha256: str | None = None,
) -> Path:
    """Create ``<models_dir>/<name>/`` with model.yaml, labels.txt and ``<name>.onnx``.

    ``sha256`` defaults to the real digest of the written model file.
    """
    model_dir = models_dir / name
    model_path = make_constant_yolox_model(model_dir / f"{name}.onnx", input_size, rows)
    (model_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    digest = sha256 or hashlib.sha256(model_path.read_bytes()).hexdigest()
    descriptor = {
        "name": name,
        "file": model_path.name,
        "labels": "labels.txt",
        "input_size": list(input_size),
        "postprocess": "yolox",
        "sha256": digest,
    }
    (model_dir / "model.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
    return model_dir

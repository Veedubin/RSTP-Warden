"""YOLOX pre/post-processing and provider selection as pure functions.

Worked numbers (input 64x64, two classes, 84 anchor rows = 8*8 + 4*4 + 2*2):
- row 19 = stride 8, cell (gx=3, gy=2): ``[0.5, 0.5, ln 2, ln 2, 0.9, 0.8, 0.1]``
  -> centre (28, 20), size 16x16, x1y1x2y2 (20, 12, 36, 28), class 0, score 0.72.
- row 69 = stride 16 (that level starts at row 64), cell (gx=1, gy=1):
  ``[0, 0, 0, 0, 0.95, 0.05, 0.9]`` -> centre (16, 16), size 16x16,
  x1y1x2y2 (8, 8, 24, 24), class 1, score 0.855.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rtsp_warden.detectors.builtin.onnx import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    decode_yolox,
    letterbox,
    select_providers,
)
from tests.helpers_onnx import yolox_output

LN2 = math.log(2.0)
INPUT = (64, 64)


@pytest.mark.parametrize(
    ("device", "available", "expected"),
    [
        ("cpu", [CUDA_PROVIDER, CPU_PROVIDER], [CPU_PROVIDER]),
        ("auto", [CUDA_PROVIDER, CPU_PROVIDER], [CUDA_PROVIDER, CPU_PROVIDER]),
        ("auto", ["AzureExecutionProvider", CPU_PROVIDER], [CPU_PROVIDER]),
        ("cuda", ["AzureExecutionProvider", CPU_PROVIDER], [CPU_PROVIDER]),
        (
            "cuda",
            ["TensorrtExecutionProvider", CUDA_PROVIDER, CPU_PROVIDER],
            [CUDA_PROVIDER, CPU_PROVIDER],
        ),
    ],
)
def test_select_providers(device: str, available: list[str], expected: list[str]) -> None:
    assert select_providers(device, available) == expected


def test_letterbox_pads_top_left_with_114_and_keeps_bgr_0_255() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)

    blob, ratio = letterbox(frame, INPUT)

    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]
    assert ratio == pytest.approx(0.2)
    # The resized image (64x36) sits at the top-left; channels stay B, G, R; no /255.
    assert blob[0, :, 0, 0].tolist() == [10.0, 20.0, 30.0]
    assert blob[0, :, 35, 63].tolist() == [10.0, 20.0, 30.0]
    # Padding fills the bottom rows with 114.
    assert blob[0, :, 36, 0].tolist() == [114.0, 114.0, 114.0]
    assert blob[0, :, 63, 63].tolist() == [114.0, 114.0, 114.0]
    assert float(blob.max()) == 114.0


def test_letterbox_size_is_width_then_height() -> None:
    frame = np.full((64, 128, 3), 10, dtype=np.uint8)  # h=64, w=128

    blob, ratio = letterbox(frame, (96, 64))  # width 96, height 64

    assert blob.shape == (1, 3, 64, 96)
    assert ratio == pytest.approx(0.75)  # min(64/64, 96/128)
    assert blob[0, 0, 47, 95] == 10.0  # resized to 96x48
    assert blob[0, 0, 48, 0] == 114.0


def test_decode_yolox_grid_strides_and_obj_times_cls() -> None:
    rows = {
        19: [0.5, 0.5, LN2, LN2, 0.9, 0.8, 0.1],
        69: [0.0, 0.0, 0.0, 0.0, 0.95, 0.05, 0.9],
    }

    decoded = decode_yolox(yolox_output(INPUT, 2, rows), INPUT)

    assert decoded.shape == (84, 6)
    assert decoded.dtype == np.float32
    assert decoded[19].tolist() == pytest.approx([20, 12, 36, 28, 0.72, 0], abs=1e-4)
    assert decoded[69].tolist() == pytest.approx([8, 8, 24, 24, 0.855, 1], abs=1e-4)
    assert decoded[0].tolist() == pytest.approx([-4, -4, 4, 4, 0, 0], abs=1e-4)  # empty row


def test_decode_yolox_rejects_output_that_does_not_fit_the_input_size() -> None:
    wrong = np.zeros((1, 80, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="84"):
        decode_yolox(wrong, INPUT)

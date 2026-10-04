"""Load an exported model through rtsp-warden's own registry and detector, run one image.

Run from the repository root with the MAIN venv (not the training one):

    uv run python tools/wildlife/verify.py tools/wildlife/out/wildlife-yolox-s [image.jpg]

This is the proof that the export matches the runtime's tensor contract: the descriptor
loads, the ONNX session opens, and ``decode_yolox`` turns the raw rows into labelled boxes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from rtsp_warden.detectors.builtin.onnx import OnnxDetector
from rtsp_warden.detectors.model_registry import load_descriptor


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    model_dir = Path(sys.argv[1]).resolve()
    desc = load_descriptor(model_dir.name, model_dir.parent)
    det = OnnxDetector(
        descriptor=desc, models_dir=model_dir.parent, device="cpu", min_confidence=0.3
    )
    det.setup()
    if det.error:
        raise SystemExit(det.error)
    if len(sys.argv) > 2:
        frame = cv2.imread(sys.argv[2])
        if frame is None:
            raise SystemExit(f"cannot read {sys.argv[2]}")
    else:
        frame = np.full((720, 1280, 3), 114, dtype=np.uint8)
    found = det.process(frame, 1.0)
    print(f"provider {det.provider}, {len(det.labels)} labels, {len(found)} detections")
    for d in found:
        print(f"  {d.kind:10s} {d.confidence:.2f} {d.bbox}")


if __name__ == "__main__":
    main()

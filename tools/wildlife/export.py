"""Export the trained checkpoint to ONNX (raw YOLOX head) and write the registry descriptor.

    uv run python export.py [--ckpt YOLOX_outputs/wildlife_yolox_s/best_ckpt.pth]
                            [--out out/wildlife-yolox-s]

The output directory is a complete user model for rtsp-warden: ``wildlife_yolox_s.onnx``,
``wildlife.txt`` and ``model.yaml`` (``postprocess: yolox``, SHA-256 of the file). Copy it
into ``WARDEN_MODELS_DIR`` (the compose stack's ``data/models/``). torch and yolox are
imported inside ``main()`` only, so the main test suite can import ``write_descriptor``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MODEL_NAME = "wildlife-yolox-s"
ONNX_NAME = "wildlife_yolox_s.onnx"
INPUT_SIZE = (640, 640)
OPSET = 13


def write_descriptor(out_dir: Path, onnx_name: str, sha256: str) -> Path:
    """Write ``model.yaml`` and copy ``wildlife.txt`` next to the ONNX file in ``out_dir``."""
    shutil.copyfile(HERE / "wildlife.txt", out_dir / "wildlife.txt")
    desc = {
        "name": MODEL_NAME,
        "file": onnx_name,
        "labels": "wildlife.txt",
        "input_size": list(INPUT_SIZE),
        "postprocess": "yolox",
        "sha256": sha256,
    }
    path = out_dir / "model.yaml"
    path.write_text(
        "# wildlife-yolox-s: YOLOX-S fine-tuned on ENA24 + Open Images + raccoon sets (RW-5).\n"
        "# Same tensor contract as yolox-s: images [1,3,640,640] BGR 0-255, raw head output.\n"
        "# Training data: CDLA-Permissive 1.0 (ENA24), CC BY 4.0 / 2.0 (Open Images), MIT.\n"
        + yaml.safe_dump(desc, sort_keys=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    for extra in (HERE, HERE / ".yolox"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import onnxruntime as ort
    import torch
    from exp import Exp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=HERE / "YOLOX_outputs" / "wildlife_yolox_s" / "best_ckpt.pth",
    )
    ap.add_argument("--out", type=Path, default=HERE / "out" / MODEL_NAME)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    exp = Exp()
    assert tuple(exp.test_size) == INPUT_SIZE
    model = exp.get_model().eval()
    # Our own checkpoint (YOLOX stores a numpy scalar in it), so weights_only=False is safe.
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.head.decode_in_inference = False  # raw grid rows: rtsp-warden's decode_yolox decodes
    dummy = torch.randn(1, 3, *INPUT_SIZE)
    onnx_path = args.out / ONNX_NAME
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["output"],
        opset_version=OPSET,
        dynamo=False,
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": dummy.numpy()})[0]
    expected = (1, 8400, 5 + exp.num_classes)
    if tuple(out.shape) != expected:
        raise SystemExit(f"ONNX output shape {out.shape}, expected {expected}")
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    write_descriptor(args.out, ONNX_NAME, sha)
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"wrote {onnx_path} ({size_mb:.1f} MB) sha256 {sha}")
    print(f"install: cp -r {args.out} <models_dir>/   (compose stack: data/models/)")


if __name__ == "__main__":
    main()

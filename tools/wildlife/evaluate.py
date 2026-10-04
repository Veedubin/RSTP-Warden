"""AP50 per class on the validation set: all, colour-only and grayscale-only images.

    uv run python evaluate.py [--ckpt YOLOX_outputs/wildlife_yolox_s/best_ckpt.pth]

Writes out/eval.json and prints one row per label. The grayscale split is what matters for
the night-time use: a fox, a raccoon and a cat on IR frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE / ".yolox"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from exp import Exp  # noqa: E402
from wildlife_data import LABELS  # noqa: E402
from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.utils import postprocess  # noqa: E402


def detect_all(ckpt: Path, conf: float, nms: float) -> tuple[COCO, list[dict]]:
    """Run the model over every validation image; COCO-format results in pixel boxes."""
    exp = Exp()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = exp.get_model().to(device).eval()
    # Our own checkpoint (YOLOX stores a numpy scalar in it), so weights_only=False is safe.
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    coco = COCO(str(Path(exp.data_dir) / "annotations" / exp.val_ann))
    cat_ids = sorted(coco.getCatIds())
    transform = ValTransform(legacy=False)
    results: list[dict] = []
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        img = cv2.imread(str(Path(exp.data_dir) / "val2017" / info["file_name"]))
        if img is None:
            continue
        ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
        tensor, _ = transform(img, None, exp.test_size)
        with torch.no_grad():
            raw = model(torch.from_numpy(tensor).unsqueeze(0).to(device))
            out = postprocess(raw, exp.num_classes, conf, nms, class_agnostic=False)[0]
        if out is None:
            continue
        for x1, y1, x2, y2, obj, cls, cls_id in out.cpu().numpy():
            x1, y1, x2, y2 = (float(v) / ratio for v in (x1, y1, x2, y2))
            results.append(
                {
                    "image_id": img_id,
                    "category_id": cat_ids[int(cls_id)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(obj * cls),
                }
            )
    return coco, results


def ap50_per_class(coco: COCO, results: list[dict], img_ids: list[int]) -> dict[str, float]:
    if not results or not img_ids:
        return {}
    dets = coco.loadRes(results)
    ev = COCOeval(coco, dets, "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    precision = ev.eval["precision"]  # [T (IoU), R (recall), K (class), A (area), M (maxDets)]
    out: dict[str, float] = {}
    for k, cat_id in enumerate(ev.params.catIds):
        p = precision[0, :, k, 0, 2]  # IoU 0.5, every recall step, all areas, maxDets 100
        p = p[p > -1]
        out[LABELS[cat_id - 1]] = float(p.mean()) if p.size else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=HERE / "YOLOX_outputs" / "wildlife_yolox_s" / "best_ckpt.pth",
    )
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms", type=float, default=0.65)
    args = ap.parse_args()

    coco, results = detect_all(args.ckpt, args.conf, args.nms)
    all_ids = coco.getImgIds()
    subsets = {
        "all": all_ids,
        "colour": [i for i in all_ids if not coco.imgs[i].get("is_gray")],
        "gray": [i for i in all_ids if coco.imgs[i].get("is_gray")],
    }
    report = {name: ap50_per_class(coco, results, ids) for name, ids in subsets.items()}
    report["images"] = {name: len(ids) for name, ids in subsets.items()}
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nAP50 on {report['images']} validation images")
    print(f"{'label':10s} {'all':>6s} {'colour':>7s} {'gray':>6s}")
    for label in LABELS:
        row = [report[s].get(label, float("nan")) for s in ("all", "colour", "gray")]
        print(f"{label:10s} {row[0]:6.3f} {row[1]:7.3f} {row[2]:6.3f}")


if __name__ == "__main__":
    main()

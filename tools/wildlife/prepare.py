"""Merge the raw sources into data/coco/{annotations,train2017,val2017} for YOLOX.

    uv run python prepare.py [--val-fraction 0.1] [--seed 20261004]

Category ids are the wildlife label index + 1 (YOLOX sorts category ids, so sorted order
is label order). Validation images carry ``is_gray`` (the runtime's grayscale measure) so
``evaluate.py`` can report colour and night AP50 separately. Images are symlinked, never
copied. Only the standard library, OpenCV and ``wildlife_data`` are imported, so the main
test suite can import the pure functions ``build_records`` and ``to_coco``.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wildlife_data import (  # noqa: E402
    COCO_CATEGORIES,
    LABELS,
    is_gray,
    map_ena24_category,
    normalized_box_to_coco,
    split_image_ids,
    voc_box_to_coco,
)

DATA = HERE / "data"
RAW = DATA / "raw"
COCO = DATA / "coco"

Record = dict  # {"file": Path, "width": int, "height": int, "source": str, "boxes": [...]}


def _xml_text(node: ET.Element | None, tag: str) -> str:
    child = node.find(tag) if node is not None else None
    if child is None or child.text is None:
        raise ValueError(f"missing <{tag}> in VOC annotation")
    return child.text


def build_records(
    ena_json: dict,
    ena_images: Path,
    oi_boxes: dict[str, list[dict]],
    oi_sizes: dict[str, tuple[int, int]],
    oi_images: Path,
    raccoon_xmls: list[Path],
    roboflow: tuple[dict, Path] | None = None,
) -> list[Record]:
    """One record per image across every source, boxes already as ``(label, [x, y, w, h])``."""
    records: list[Record] = []

    # ENA24: COCO Camera Traps json, categories mapped onto the wildlife labels.
    cats = {c["id"]: c["name"] for c in ena_json.get("categories", [])}
    by_image: dict[str, list[dict]] = {}
    for ann in ena_json.get("annotations", []):
        if "bbox" in ann:
            by_image.setdefault(str(ann["image_id"]), []).append(ann)
    for img in ena_json.get("images", []):
        anns = by_image.get(str(img["id"]))
        if not anns:
            continue
        boxes = [
            (map_ena24_category(cats[a["category_id"]]), [float(v) for v in a["bbox"]])
            for a in anns
        ]
        records.append(
            {
                "file": ena_images / img["file_name"],
                "width": int(img["width"]),
                "height": int(img["height"]),
                "source": "ena24",
                "boxes": boxes,
            }
        )

    # Open Images: normalized boxes, scaled with the downloaded image's real size.
    for image_id, boxes in oi_boxes.items():
        size = oi_sizes.get(image_id)
        if size is None:
            continue
        w, h = size
        records.append(
            {
                "file": oi_images / f"{image_id}.jpg",
                "width": int(w),
                "height": int(h),
                "source": "openimages",
                "boxes": [
                    (
                        b["label"],
                        normalized_box_to_coco(b["xmin"], b["xmax"], b["ymin"], b["ymax"], w, h),
                    )
                    for b in boxes
                ],
            }
        )

    # Dat Tran's raccoon set: Pascal VOC xml next to an images/ directory.
    for xml_path in raccoon_xmls:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        w, h = int(_xml_text(size, "width")), int(_xml_text(size, "height"))
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            corners = (float(_xml_text(bb, k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append(("raccoon", voc_box_to_coco(*corners)))
        records.append(
            {
                "file": xml_path.parent.parent / "images" / _xml_text(root, "filename"),
                "width": w,
                "height": h,
                "source": "raccoon",
                "boxes": boxes,
            }
        )

    # Roboflow COCO export (optional): labels matched by lower-cased name, others dropped.
    if roboflow is not None:
        rf_json, rf_dir = roboflow
        rf_cats = {c["id"]: str(c["name"]).strip().lower() for c in rf_json["categories"]}
        rf_by_image: dict[int, list[dict]] = {}
        for ann in rf_json["annotations"]:
            rf_by_image.setdefault(ann["image_id"], []).append(ann)
        for img in rf_json["images"]:
            boxes = [
                (rf_cats[a["category_id"]], [float(v) for v in a["bbox"]])
                for a in rf_by_image.get(img["id"], [])
                if rf_cats.get(a["category_id"]) in LABELS
            ]
            if boxes:
                records.append(
                    {
                        "file": rf_dir / img["file_name"],
                        "width": int(img["width"]),
                        "height": int(img["height"]),
                        "source": "roboflow",
                        "boxes": boxes,
                    }
                )
    return records


def record_key(rec: Record) -> str:
    return f"{rec['source']}/{Path(rec['file']).name}"


def to_coco(records: list[Record], gray_flags: dict[str, bool]) -> dict:
    """COCO json for ``records``; ``gray_flags`` (by record_key) adds ``is_gray`` to images."""
    cat_id = {c["name"]: c["id"] for c in COCO_CATEGORIES}
    images: list[dict] = []
    annotations: list[dict] = []
    for i, rec in enumerate(records, start=1):
        key = record_key(rec)
        entry = {
            "id": i,
            "file_name": key.replace("/", "__"),
            "width": rec["width"],
            "height": rec["height"],
            "source": rec["source"],
        }
        if key in gray_flags:
            entry["is_gray"] = bool(gray_flags[key])
        images.append(entry)
        for label, (x, y, w, h) in rec["boxes"]:
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": i,
                    "category_id": cat_id[label],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
    return {"images": images, "annotations": annotations, "categories": COCO_CATEGORIES}


def _load_roboflow(rf_dir: Path) -> tuple[dict, Path] | None:
    """Merge a Roboflow COCO export's train/valid/test splits into one json, or None."""
    if not rf_dir.exists():
        return None
    merged: dict = {"images": [], "annotations": [], "categories": None}
    for split in ("train", "valid", "test"):
        ann = rf_dir / split / "_annotations.coco.json"
        if not ann.exists():
            continue
        part = json.loads(ann.read_text(encoding="utf-8"))
        offset = len(merged["images"])
        for img in part["images"]:
            img["id"] += offset
            img["file_name"] = f"{split}/{img['file_name']}"
        for a in part["annotations"]:
            a["image_id"] += offset
        merged["images"] += part["images"]
        merged["annotations"] += part["annotations"]
        merged["categories"] = part["categories"]
    return (merged, rf_dir) if merged["categories"] else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    ena_json = json.loads((RAW / "ena24" / "ena24.json").read_text(encoding="utf-8"))
    oi_boxes_path = RAW / "openimages" / "boxes.json"
    oi_boxes = (
        json.loads(oi_boxes_path.read_text(encoding="utf-8")) if oi_boxes_path.exists() else {}
    )
    oi_images = RAW / "openimages" / "images"
    oi_sizes: dict[str, tuple[int, int]] = {}
    for image_id in oi_boxes:
        img = cv2.imread(str(oi_images / f"{image_id}.jpg"))
        if img is not None:
            oi_sizes[image_id] = (img.shape[1], img.shape[0])
    raccoon_xmls = sorted((RAW / "raccoon_dataset-master" / "annotations").glob("*.xml"))
    roboflow = _load_roboflow(RAW / "roboflow-cat-raccoons")

    records = [
        r
        for r in build_records(
            ena_json,
            RAW / "ena24" / "images",
            oi_boxes,
            oi_sizes,
            oi_images,
            raccoon_xmls,
            roboflow,
        )
        if Path(r["file"]).exists()
    ]
    keys = [record_key(r) for r in records]
    train_keys, val_keys = split_image_ids(keys, args.val_fraction, args.seed)
    by_key = {record_key(r): r for r in records}
    gray: dict[str, bool] = {}
    for key in val_keys:
        img = cv2.imread(str(by_key[key]["file"]))
        gray[key] = bool(img is not None and is_gray(img))

    (COCO / "annotations").mkdir(parents=True, exist_ok=True)
    for split, split_keys in (("train", train_keys), ("val", val_keys)):
        recs = [by_key[k] for k in split_keys]
        coco = to_coco(recs, gray)
        img_dir = COCO / ("train2017" if split == "train" else "val2017")
        img_dir.mkdir(parents=True, exist_ok=True)
        for entry, rec in zip(coco["images"], recs, strict=True):
            link = img_dir / entry["file_name"]
            if not link.exists():
                link.symlink_to(Path(rec["file"]).resolve())
        (COCO / "annotations" / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
        counts = Counter(LABELS[a["category_id"] - 1] for a in coco["annotations"])
        sources = Counter(i["source"] for i in coco["images"])
        print(
            f"{split}: {len(coco['images'])} images {dict(sources)}, {len(coco['annotations'])} boxes"
        )
        for label in LABELS:
            print(f"  {counts.get(label, 0):6d}  {label}")
    print(f"val grayscale images: {sum(gray.values())} of {len(gray)}")


if __name__ == "__main__":
    main()

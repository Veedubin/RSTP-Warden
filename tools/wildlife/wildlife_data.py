"""Pure data-prep helpers for the wildlife model (stdlib + NumPy + OpenCV only).

Imported by ``fetch.py``, ``prepare.py``, ``exp.py`` and by ``tests/test_wildlife_tool.py``
(by file path, from the main venv), so this module never imports torch or yolox.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
#: The model's labels in class-id order (index = class id), from wildlife.txt.
LABELS: tuple[str, ...] = tuple((HERE / "wildlife.txt").read_text(encoding="utf-8").split())
#: COCO categories for the merged dataset: id = label index + 1, so YOLOX's sorted
#: category ids map onto the label order.
COCO_CATEGORIES: list[dict] = [
    {
        "id": i + 1,
        "name": name,
        "supercategory": name if name in ("person", "vehicle") else "animal",
    }
    for i, name in enumerate(LABELS)
]


class UnknownCategory(ValueError):
    """A source category with no entry in the mapping table."""


def label_id(name: str) -> int:
    """0-based class id of a label."""
    return LABELS.index(name)


#: ENA24 category (lower-cased, underscores and dashes as spaces) -> label.
ENA24_MAP: dict[str, str] = {
    "american black bear": "bear",
    "american crow": "bird",
    "bird": "bird",
    "bobcat": "bobcat",
    "chicken": "bird",
    "coyote": "coyote",
    "dog": "dog",
    "domestic cat": "cat",
    "eastern chipmunk": "chipmunk",
    "eastern cottontail": "rabbit",
    "eastern fox squirrel": "squirrel",
    "eastern gray squirrel": "squirrel",
    "grey fox": "fox",
    "horse": "horse",
    "human": "person",
    "northern raccoon": "raccoon",
    "red fox": "fox",
    "striped skunk": "skunk",
    "vehicle": "vehicle",
    "virginia opossum": "opossum",
    "white tailed deer": "deer",
    "wild turkey": "bird",
    "woodchuck": "woodchuck",
}


def map_ena24_category(name: str) -> str:
    """Label for an ENA24 category name; raises UnknownCategory naming an unmapped one."""
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    try:
        return ENA24_MAP[key]
    except KeyError:
        raise UnknownCategory(f"ENA24 category {name!r} has no label mapping") from None


#: Open Images V7 boxable class MID -> label (from oidv7-class-descriptions-boxable.csv).
OPEN_IMAGES_MIDS: dict[str, str] = {
    "/m/01yrx": "cat",
    "/m/0306r": "fox",
    "/m/0dq75": "raccoon",
    "/m/0km7z": "skunk",
    "/m/071qp": "squirrel",
    "/m/06mf6": "rabbit",
    "/m/0bt9lr": "dog",
    # Hard negatives: the public ENA24 zip leaves out its human images, so people and cars
    # come from Open Images (the camera's yolox-s slot reports them; this model must not
    # call a person a bear).
    "/m/01g317": "person",
    "/m/0k4j": "vehicle",
}


def filter_open_images_rows(
    rows: Iterable[dict[str, str]],
    cap_per_class: int,
    only_labels: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Boxes of wanted classes grouped by image id, at most ``cap_per_class`` images per label.

    Rows with ``IsGroupOf`` or ``IsDepiction`` set are skipped (crowds and drawings). The
    first ``cap_per_class`` images seen per label are kept; every wanted box of a kept image
    is kept, even one that arrives after its label hit the cap. With ``only_labels`` an image
    is only *admitted* for one of those labels (its other wanted boxes still come along).
    Boxes stay normalized (``xmin, xmax, ymin, ymax`` in 0..1) until the image size is known.
    """
    kept_images: dict[str, set[str]] = {label: set() for label in OPEN_IMAGES_MIDS.values()}
    out: dict[str, list[dict]] = {}
    for row in rows:
        label = OPEN_IMAGES_MIDS.get(row.get("LabelName", ""))
        if label is None or row.get("IsGroupOf") == "1" or row.get("IsDepiction") == "1":
            continue
        image_id = row["ImageID"]
        box = {
            "label": label,
            "xmin": float(row["XMin"]),
            "xmax": float(row["XMax"]),
            "ymin": float(row["YMin"]),
            "ymax": float(row["YMax"]),
        }
        if image_id in out:
            out[image_id].append(box)
            continue
        if only_labels is not None and label not in only_labels:
            continue
        if len(kept_images[label]) >= cap_per_class:
            continue
        kept_images[label].add(image_id)
        out[image_id] = [box]
    return out


def voc_box_to_coco(xmin: float, ymin: float, xmax: float, ymax: float) -> list[float]:
    """Pascal VOC corners -> COCO ``[x, y, w, h]``."""
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


def normalized_box_to_coco(
    xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int
) -> list[float]:
    """Open Images normalized corners -> COCO ``[x, y, w, h]`` in pixels."""
    return [xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height]


def split_image_ids(
    ids: Sequence[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Seeded shuffle; returns ``(train, val)``, each sorted, with no id in both."""
    order = list(ids)
    random.Random(seed).shuffle(order)
    n_val = int(round(len(order) * val_fraction))
    return sorted(order[n_val:]), sorted(order[:n_val])


_SUBSAMPLE = 4


def channel_spread(frame_bgr: np.ndarray) -> float:
    """Same formula as rtsp_warden.detectors.daylight.channel_spread (pinned by a test)."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3 or frame_bgr.size == 0:
        return 0.0
    small = frame_bgr[::_SUBSAMPLE, ::_SUBSAMPLE, :3]
    if small.size == 0:
        return 0.0
    spread = small.max(axis=2).astype(np.int16) - small.min(axis=2).astype(np.int16)
    return float(spread.mean())


def is_gray(frame_bgr: np.ndarray, threshold: float = 4.0) -> bool:
    """True for a grayscale (IR) image under the runtime's night threshold."""
    return channel_spread(frame_bgr) < threshold


def grayscale_copy(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """IR look-alike of a colour image: gray x3 channels, brightness 0.4-1.0, noise sigma 0-6."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray *= float(rng.uniform(0.4, 1.0))
    sigma = float(rng.uniform(0.0, 6.0))
    gray += rng.normal(0.0, sigma, size=gray.shape).astype(np.float32)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


__all__ = [
    "COCO_CATEGORIES",
    "ENA24_MAP",
    "LABELS",
    "OPEN_IMAGES_MIDS",
    "UnknownCategory",
    "channel_spread",
    "filter_open_images_rows",
    "grayscale_copy",
    "is_gray",
    "label_id",
    "map_ena24_category",
    "normalized_box_to_coco",
    "split_image_ids",
    "voc_box_to_coco",
]

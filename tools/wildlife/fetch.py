"""Download the wildlife training sources into data/raw/ (resumable; skips what exists).

    uv run python fetch.py [--cap 1500] [--workers 16] [--only ena24|openimages|raccoon]

Sources and licenses are listed in README.md. Only the standard library and
``wildlife_data`` are imported at module level, so the main test suite can import this
file; httpx and tqdm load inside the functions that download.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wildlife_data import filter_open_images_rows  # noqa: E402

DATA = HERE / "data"
RAW = DATA / "raw"
ENA24_IMAGES_URL = "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip"
ENA24_JSON_URL = "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json"
OI_BOXES_CSV_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
OI_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
# The much smaller validation and test splits add images of the rare labels only.
OI_EXTRA_SPLITS: dict[str, tuple[str, str]] = {
    "validation": (
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg",
    ),
    "test": (
        "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        "https://open-images-dataset.s3.amazonaws.com/test/{image_id}.jpg",
    ),
}
OI_RARE_LABELS = frozenset({"fox", "raccoon", "skunk"})
RACCOON_ZIP_URL = "https://github.com/datitran/raccoon_dataset/archive/refs/heads/master.zip"


def download(url: str, dest: Path, *, desc: str | None = None) -> Path:
    """Stream ``url`` to ``dest`` through ``dest.part``, resuming with a Range header."""
    import httpx
    from tqdm import tqdm

    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120) as r:
        if r.status_code == 416:  # the .part file is already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        resumed = r.status_code == 206
        total = int(r.headers.get("content-length", 0)) + (have if resumed else 0)
        with (
            part.open("ab" if resumed else "wb") as fh,
            tqdm(
                total=total or None,
                initial=have if resumed else 0,
                unit="B",
                unit_scale=True,
                desc=desc or dest.name,
            ) as bar,
        ):
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)
    return dest


def fetch_ena24() -> None:
    """ENA24 metadata and images; the zip is flat (jpgs at its root), extracted to images/."""
    out = RAW / "ena24"
    download(ENA24_JSON_URL, out / "ena24.json")
    zip_path = download(ENA24_IMAGES_URL, out / "ena24.zip", desc="ena24.zip (3.6 GB)")
    images = out / "images"
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
        missing = [n for n in names if not (images / Path(n).name).exists()]
        if missing:
            images.mkdir(parents=True, exist_ok=True)
            print(f"ena24: extracting {len(missing)} of {len(names)} images")
            for name in missing:
                with zf.open(name) as src, (images / Path(name).name).open("wb") as dst:
                    dst.write(src.read())
    print(f"ena24: {sum(1 for _ in images.glob('*.jpg'))} images in {images}")


def _filter_csv(csv_path: Path, cap: int, only_labels: frozenset[str] | None) -> dict:
    print(f"open images: filtering {csv_path.name}")
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return filter_open_images_rows(
            csv.DictReader(fh), cap_per_class=cap, only_labels=only_labels
        )


def _download_images(boxes: dict, image_url: str, images: Path, workers: int, what: str) -> None:
    import httpx

    images.mkdir(parents=True, exist_ok=True)
    todo = [i for i in boxes if not (images / f"{i}.jpg").exists()]
    print(f"{what}: {len(boxes)} images kept, {len(todo)} to download")
    failed = 0

    def one(client: httpx.Client, image_id: str) -> bool:
        r = client.get(image_url.format(image_id=image_id))
        if r.status_code != 200:
            return False
        tmp = images / f"{image_id}.jpg.part"
        tmp.write_bytes(r.content)
        tmp.rename(images / f"{image_id}.jpg")
        return True

    with (
        httpx.Client(timeout=60, follow_redirects=True) as client,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [pool.submit(one, client, image_id) for image_id in todo]
        for n, fut in enumerate(as_completed(futures), start=1):
            if not fut.result():
                failed += 1
            if n % 500 == 0 or n == len(futures):
                print(f"{what}: {n}/{len(futures)} done, {failed} failed")


def fetch_openimages(cap: int, workers: int) -> None:
    """Filter the box CSVs once (boxes.json per split), then fetch the kept images in parallel.

    The train split gives every wanted label up to ``cap`` images; the small validation and
    test splits only add images of the rare labels (fox, raccoon, skunk). Image ids are
    unique across splits, so one images/ directory holds them all and prepare.py reads
    every ``boxes*.json``.
    """
    out = RAW / "openimages"
    images = out / "images"
    boxes_path = out / "boxes.json"
    if not boxes_path.exists():
        csv_path = download(
            OI_BOXES_CSV_URL,
            out / "oidv6-train-annotations-bbox.csv",
            desc="open images boxes csv (2.3 GB)",
        )
        boxes_path.write_text(json.dumps(_filter_csv(csv_path, cap, None)), encoding="utf-8")
    _download_images(
        json.loads(boxes_path.read_text(encoding="utf-8")), OI_IMAGE_URL, images, workers,
        "open images (train)",
    )  # fmt: skip
    for split, (csv_url, image_url) in OI_EXTRA_SPLITS.items():
        split_boxes = out / f"boxes-{split}.json"
        if not split_boxes.exists():
            csv_path = download(csv_url, out / f"{split}-annotations-bbox.csv")
            split_boxes.write_text(
                json.dumps(_filter_csv(csv_path, cap, OI_RARE_LABELS)), encoding="utf-8"
            )
        _download_images(
            json.loads(split_boxes.read_text(encoding="utf-8")), image_url, images, workers,
            f"open images ({split})",
        )  # fmt: skip


def fetch_raccoon() -> None:
    """Dat Tran's 196-image raccoon set (MIT): images/ and Pascal VOC annotations/."""
    out = RAW / "raccoon_dataset-master"
    if out.exists():
        return
    import httpx

    r = httpx.get(RACCOON_ZIP_URL, follow_redirects=True, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(RAW)
    print(f"raccoon: {sum(1 for _ in (out / 'images').glob('*.jpg'))} images")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=1500, help="Open Images: images per class")
    ap.add_argument("--workers", type=int, default=16, help="parallel image downloads")
    ap.add_argument("--only", choices=["ena24", "openimages", "raccoon"])
    args = ap.parse_args()
    if args.only in (None, "raccoon"):
        fetch_raccoon()
    if args.only in (None, "ena24"):
        fetch_ena24()
    if args.only in (None, "openimages"):
        fetch_openimages(args.cap, args.workers)


if __name__ == "__main__":
    main()

"""tools/wildlife/wildlife_data.py: pure data-prep functions, imported by file path (no torch).

The training tool is a separate uv project and never part of the package; these tests pin
the parts that must agree with the runtime (labels order, the grayscale measure, the
descriptor the registry loads) and the pure conversions, all offline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from rtsp_warden.detectors.daylight import channel_spread as runtime_spread

TOOL = Path(__file__).resolve().parents[1] / "tools" / "wildlife"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOL / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wd = _load("wildlife_data")


def test_labels_match_the_shipped_file_and_the_spec_order() -> None:
    lines = (TOOL / "wildlife.txt").read_text(encoding="utf-8").split()
    assert lines == list(wd.LABELS)
    assert wd.LABELS[0] == "cat" and wd.LABELS[-1] == "vehicle" and len(wd.LABELS) == 18
    assert wd.label_id("raccoon") == 3
    assert [c["id"] for c in wd.COCO_CATEGORIES] == list(range(1, 19))
    assert wd.COCO_CATEGORIES[2]["name"] == "fox"


@pytest.mark.parametrize(
    ("ena", "label"),
    [
        ("Red Fox", "fox"),
        ("Grey Fox", "fox"),
        ("Northern Raccoon", "raccoon"),
        ("Domestic Cat", "cat"),
        ("White_Tailed_Deer", "deer"),
        ("Wild Turkey", "bird"),
        ("American Crow", "bird"),
        ("Chicken", "bird"),
        ("Eastern Fox Squirrel", "squirrel"),
        ("Eastern Gray Squirrel", "squirrel"),
        ("Human", "person"),
        ("Vehicle", "vehicle"),
    ],
)
def test_ena24_mapping(ena: str, label: str) -> None:
    assert wd.map_ena24_category(ena) == label


def test_ena24_mapping_covers_every_category_of_the_dataset() -> None:
    names = [
        "American Crow", "Human", "American Black Bear", "Dog", "Chicken", "Virginia Opossum",
        "Horse", "Domestic Cat", "Grey Fox", "Wild Turkey", "Red Fox", "White_Tailed_Deer",
        "Coyote", "Eastern Fox Squirrel", "Eastern Cottontail", "Bobcat", "Eastern Gray Squirrel",
        "Eastern Chipmunk", "Striped Skunk", "Vehicle", "Northern Raccoon", "Bird", "Woodchuck",
    ]  # fmt: skip
    assert len(names) == 23
    for name in names:
        assert wd.map_ena24_category(name) in wd.LABELS


def test_ena24_unknown_category_is_an_error_naming_it() -> None:
    with pytest.raises(wd.UnknownCategory, match="Sasquatch"):
        wd.map_ena24_category("Sasquatch")


def test_open_images_filter_caps_images_and_skips_groups_and_drawings() -> None:
    def row(img: str, mid: str, **extra: str) -> dict[str, str]:
        base = {
            "ImageID": img,
            "LabelName": mid,
            "XMin": "0.1",
            "XMax": "0.5",
            "YMin": "0.2",
            "YMax": "0.6",
            "IsGroupOf": "0",
            "IsDepiction": "0",
        }
        return {**base, **extra}

    cat, fox, bus = "/m/01yrx", "/m/0306r", "/m/01bjv"
    rows = [
        row("a", cat),
        row("a", cat),
        row("b", cat),
        row("c", cat),
        row("d", fox),
        row("e", fox, IsGroupOf="1"),
        row("f", fox, IsDepiction="1"),
        row("g", bus),
    ]
    out = wd.filter_open_images_rows(rows, cap_per_class=2)
    assert sorted(out) == ["a", "b", "d"]
    assert len(out["a"]) == 2 and out["a"][0]["label"] == "cat"
    assert out["d"][0] == {"label": "fox", "xmin": 0.1, "xmax": 0.5, "ymin": 0.2, "ymax": 0.6}


def test_open_images_filter_keeps_every_wanted_box_of_a_kept_image() -> None:
    cat, fox = "/m/01yrx", "/m/0306r"

    def row(img: str, mid: str) -> dict[str, str]:
        return {"ImageID": img, "LabelName": mid, "XMin": "0", "XMax": "1", "YMin": "0",
                "YMax": "1", "IsGroupOf": "0", "IsDepiction": "0"}  # fmt: skip

    # image "x" is kept for cat; its fox box arriving later is kept even though fox is capped
    rows = [row("x", cat), row("y", fox), row("x", fox)]
    out = wd.filter_open_images_rows(rows, cap_per_class=1)
    assert sorted(out) == ["x", "y"]
    assert [b["label"] for b in out["x"]] == ["cat", "fox"]


def test_box_conversions() -> None:
    assert wd.voc_box_to_coco(10, 20, 50, 80) == [10.0, 20.0, 40.0, 60.0]
    assert wd.normalized_box_to_coco(0.1, 0.5, 0.2, 0.6, 200, 100) == pytest.approx(
        [20.0, 20.0, 80.0, 40.0]
    )


def test_split_is_seeded_and_disjoint() -> None:
    ids = [f"img{i}" for i in range(100)]
    train, val = wd.split_image_ids(ids, 0.1, seed=7)
    assert len(val) == 10 and len(train) == 90 and not set(train) & set(val)
    assert wd.split_image_ids(ids, 0.1, seed=7) == (train, val)
    assert wd.split_image_ids(ids, 0.1, seed=8) != (train, val)


def test_channel_spread_matches_the_runtime_formula() -> None:
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(45, 80, 3), dtype=np.uint8)
    for _ in range(5):
        frame = rng.integers(0, 256, size=(45, 80, 3), dtype=np.uint8)
        assert wd.channel_spread(frame) == pytest.approx(runtime_spread(frame))
    assert wd.is_gray(np.full((45, 80, 3), 90, dtype=np.uint8)) is True
    assert wd.is_gray(frame) is False


def test_grayscale_copy_is_grey_and_darker_or_equal() -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(45, 80, 3), dtype=np.uint8)
    out = wd.grayscale_copy(img, rng)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert wd.channel_spread(out) < 12.0  # grey plus a little noise
    assert out.mean() <= img.mean() + 6.0


def test_fetch_urls_are_the_verified_mirrors() -> None:
    fetch = _load("fetch")
    assert fetch.ENA24_IMAGES_URL == (
        "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip"
    )
    assert fetch.ENA24_JSON_URL == (
        "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json"
    )
    assert fetch.OI_BOXES_CSV_URL == (
        "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
    )
    assert fetch.OI_IMAGE_URL.format(image_id="abc") == (
        "https://open-images-dataset.s3.amazonaws.com/train/abc.jpg"
    )
    assert fetch.RACCOON_ZIP_URL == (
        "https://github.com/datitran/raccoon_dataset/archive/refs/heads/master.zip"
    )


def test_build_records_and_to_coco_use_the_wildlife_categories(tmp_path: Path) -> None:
    prep = _load("prepare")
    ena = {
        "images": [{"id": "1", "file_name": "1.jpg", "width": 1920, "height": 1080}],
        "annotations": [{"image_id": "1", "category_id": 5, "bbox": [10, 20, 30, 40]}],
        "categories": [{"id": 5, "name": "Northern Raccoon"}],
    }
    oi_boxes = {"img": [{"label": "fox", "xmin": 0.0, "xmax": 0.5, "ymin": 0.0, "ymax": 0.5}]}
    oi_sizes = {"img": (200, 100)}
    records = prep.build_records(
        ena, tmp_path / "ena", oi_boxes, oi_sizes, tmp_path / "oi", raccoon_xmls=[]
    )
    assert [r["source"] for r in records] == ["ena24", "openimages"]
    assert records[0]["file"] == tmp_path / "ena" / "1.jpg"
    assert records[0]["boxes"] == [("raccoon", [10.0, 20.0, 30.0, 40.0])]
    assert records[1]["boxes"] == [("fox", [0.0, 0.0, 100.0, 50.0])]
    coco = prep.to_coco(records, {"ena24/1.jpg": True, "openimages/img.jpg": False})
    assert [c["name"] for c in coco["categories"]][:4] == ["cat", "dog", "fox", "raccoon"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    assert [cats[a["category_id"]] for a in coco["annotations"]] == ["raccoon", "fox"]
    assert coco["images"][0]["is_gray"] is True and coco["images"][1]["is_gray"] is False
    assert coco["images"][0]["file_name"] == "ena24__1.jpg"
    assert all(a["iscrowd"] == 0 and a["area"] > 0 for a in coco["annotations"])


def test_build_records_reads_voc_xml_and_skips_unknown_roboflow_labels(tmp_path: Path) -> None:
    prep = _load("prepare")
    xml_dir = tmp_path / "raccoon_dataset-master" / "annotations"
    xml_dir.mkdir(parents=True)
    (xml_dir / "r1.xml").write_text(
        "<annotation><filename>r1.jpg</filename><size><width>640</width><height>480</height>"
        "</size><object><name>raccoon</name><bndbox><xmin>10</xmin><ymin>20</ymin>"
        "<xmax>110</xmax><ymax>220</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    rf = {
        "images": [{"id": 1, "file_name": "train/a.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]},
            {"image_id": 1, "category_id": 2, "bbox": [5, 6, 7, 8]},
        ],
        "categories": [{"id": 1, "name": "Raccoon"}, {"id": 2, "name": "cat-raccoons"}],
    }
    empty = {"images": [], "annotations": [], "categories": []}
    records = prep.build_records(
        empty, tmp_path, {}, {}, tmp_path, raccoon_xmls=[xml_dir / "r1.xml"],
        roboflow=(rf, tmp_path / "rf"),
    )  # fmt: skip
    assert [r["source"] for r in records] == ["raccoon", "roboflow"]
    assert records[0]["file"] == tmp_path / "raccoon_dataset-master" / "images" / "r1.jpg"
    assert records[0]["boxes"] == [("raccoon", [10.0, 20.0, 100.0, 200.0])]
    assert records[1]["boxes"] == [("raccoon", [1.0, 2.0, 3.0, 4.0])]  # "cat-raccoons" dropped

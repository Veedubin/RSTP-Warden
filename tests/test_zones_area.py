"""Zones: `kind: ignore | area`, zone-name rules, rule references, runtime frame size.

Ignore zones drop detections centred in a blocked cell, judged in the decoded frame's
pixels. Area zones never drop anything; their active cells name a region that events
and rules refer to.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rtsp_warden.config import CameraConfig, GridZoneConfig, load_config, validate_camera_zones
from rtsp_warden.detectors.base import Detection
from rtsp_warden.detectors.grid_mask import GridMask
from rtsp_warden.detectors.registry import (
    CameraDetectorBundle,
    build_area_masks_from_config,
    build_detectors_for_camera,
    build_grid_masks_from_config,
)
from rtsp_warden.detectors.runner import DetectorRunner, _FrameJob
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.routes._common import templates
from rtsp_warden.web.routes.zones import _zone_to_dict

URL = "rtsp://u:p@h/m"
RIGHT_HALF = [(c, r) for c in range(8, 16) for r in range(16)]


def _zone(name: str, kind: str = "ignore", cells=(), **extra: object) -> dict:
    """One raw zone entry as it appears in config.yaml (16x16, saved at 1920x1080)."""
    return {
        "name": name,
        "kind": kind,
        "grid_cols": 16,
        "grid_rows": 16,
        "blocked_cells": [list(cell) for cell in cells],
        "frame_width": 1920,
        "frame_height": 1080,
        **extra,
    }


def _rule(name: str, zones: list[str]) -> dict:
    """One raw rule entry; `phone` is the action name the web fixture defines."""
    return {"name": name, "zones": zones, "actions": ["phone"]}


def _camera(zones: list[dict], rules: list[dict] | None = None, detectors=None) -> CameraConfig:
    return CameraConfig.model_validate(
        {
            "name": "yard",
            "main_url": URL,
            "zones": zones,
            "rules": rules or [],
            "detectors": detectors or [],
        }
    )


# ---------------------------------------------------------------------------
# Config: kind, names, uniqueness, rule references
# ---------------------------------------------------------------------------


def test_zone_kind_defaults_to_ignore() -> None:
    zone = GridZoneConfig(name="road", frame_width=1920, frame_height=1080)
    assert zone.kind == "ignore"


def test_zone_kind_area_is_accepted() -> None:
    zone = GridZoneConfig(name="yard", kind="area", frame_width=1920, frame_height=1080)
    assert zone.kind == "area"


def test_unknown_zone_kind_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'ignore' or 'area'"):
        GridZoneConfig(name="yard", kind="include", frame_width=1920, frame_height=1080)


def test_zone_name_is_stripped() -> None:
    zone = GridZoneConfig(name="  yard  ", frame_width=1920, frame_height=1080)
    assert zone.name == "yard"


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("   ", "zone name must not be empty"),
        ("front/yard", "zone name must not contain '/'"),
        ("x" * 65, "zone name must be at most 64 characters"),
    ],
)
def test_bad_zone_names_are_rejected(name: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        GridZoneConfig(name=name, frame_width=1920, frame_height=1080)


def test_zone_name_of_64_characters_is_accepted() -> None:
    zone = GridZoneConfig(name="x" * 64, frame_width=1920, frame_height=1080)
    assert len(zone.name) == 64


def test_zone_names_must_be_unique_per_camera() -> None:
    with pytest.raises(ValidationError, match="zone name 'road' is used more than once"):
        _camera([_zone("road"), _zone("road", kind="area")])


def test_rule_may_name_an_area_zone() -> None:
    cam = _camera([_zone("drive", kind="area")], [_rule("night", ["drive"])])
    assert cam.rules[0].zones == ["drive"]


def test_rule_may_name_a_disabled_area_zone() -> None:
    cam = _camera([_zone("drive", kind="area", enabled=False)], [_rule("night", ["drive"])])
    assert cam.zones[0].enabled is False


def test_rule_without_zones_needs_no_zone() -> None:
    cam = _camera([], [_rule("any", [])])
    assert cam.rules[0].zones == []


def test_rule_naming_an_unknown_zone_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _camera([_zone("drive", kind="area")], [_rule("night", ["porch"])])
    text = str(excinfo.value)
    assert "camera 'yard': rule 'night' names zone 'porch'" in text
    assert "which is not a zone of this camera (area zones: 'drive')" in text
    assert "u:p@" not in text


def test_rule_naming_an_ignore_zone_is_rejected() -> None:
    expected = "rule 'night' names zone 'road', which is an ignore zone"
    with pytest.raises(ValidationError, match=expected):
        _camera([_zone("road")], [_rule("night", ["road"])])


def test_validate_camera_zones_works_on_plain_lists() -> None:
    zones = [GridZoneConfig(name="drive", kind="area", frame_width=1, frame_height=1)]
    validate_camera_zones(zones, [])
    with pytest.raises(ValueError, match="zone name 'drive' is used more than once"):
        validate_camera_zones(zones + zones, [])


def test_zone_errors_at_load_do_not_echo_the_camera_url(tmp_path: Path) -> None:
    """The SystemExit text names the zone and never prints the camera URL."""
    path = tmp_path / "config.yaml"
    camera = {"name": "yard", "main_url": URL, "zones": [_zone("road"), _zone("road")]}
    path.write_text(yaml.safe_dump({"cameras": [camera]}), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        load_config(path)
    text = str(excinfo.value)
    assert "zone name 'road' is used more than once" in text
    assert "u:p@" not in text


# ---------------------------------------------------------------------------
# Registry: ignore zones become grid_masks, area zones become area_masks
# ---------------------------------------------------------------------------


def test_build_grid_masks_skips_area_zones() -> None:
    zones = [GridZoneConfig(**_zone("road")), GridZoneConfig(**_zone("drive", kind="area"))]
    assert [gm.name for gm in build_grid_masks_from_config(zones)] == ["road"]


def test_build_area_masks_keeps_enabled_areas_in_config_order() -> None:
    zones = [
        GridZoneConfig(**_zone("road")),
        GridZoneConfig(**_zone("porch", kind="area", cells=[(1, 1)])),
        GridZoneConfig(**_zone("old", kind="area", enabled=False)),
        GridZoneConfig(**_zone("drive", kind="area")),
    ]
    areas = build_area_masks_from_config(zones)
    assert [name for name, _ in areas] == ["porch", "drive"]
    assert all(isinstance(gm, GridMask) and gm.name == name for name, gm in areas)
    assert areas[0][1].is_cell_blocked(1, 1)


def test_camera_bundle_splits_ignore_and_area_zones(tmp_path: Path) -> None:
    cam = _camera(
        [_zone("road", cells=[(0, 0)]), _zone("drive", kind="area"), _zone("porch", kind="area")],
        detectors=[{"type": "motion"}],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=tmp_path)
    assert [gm.name for gm in bundle.grid_masks] == ["road"]
    assert [name for name, _ in bundle.area_masks] == ["drive", "porch"]


def test_camera_bundle_area_masks_default_to_empty() -> None:
    assert CameraDetectorBundle().area_masks == []


# ---------------------------------------------------------------------------
# Runner: ignore zones are judged in the decoded frame's pixels
# ---------------------------------------------------------------------------


def _jpeg(width: int, height: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


class _BoxDetector:
    """Returns one person box per frame, in the pixels of the frame it is given."""

    name = "box"
    kind = "person"

    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        return [Detection(kind="person", confidence=0.9, bbox=self.bbox, ts_unix=ts_unix)]


def _run_one_frame(
    grid_masks: list[GridMask], bbox: tuple[int, int, int, int], width: int, height: int
) -> list[Detection]:
    """Push one black width x height JPEG through a runner synchronously; return the sink input."""
    got: list[list[Detection]] = []
    runner = DetectorRunner(
        detectors=(_BoxDetector(bbox),),
        grid_masks=grid_masks,
        result_sinks=[lambda camera, stream, dets: got.append(dets)],
        swallow_exceptions=False,
    )
    job = _FrameJob(camera="yard", stream="main", jpeg_bytes=_jpeg(width, height), ts_unix=1.0)
    runner._process_job(job)
    assert len(got) == 1
    return got[0]


@pytest.mark.parametrize(
    ("width", "height", "right_box", "left_box"),
    [
        (320, 180, (220, 70, 40, 40), (40, 70, 40, 40)),
        (640, 360, (440, 140, 80, 80), (80, 140, 80, 80)),
    ],
)
def test_runner_judges_ignore_zones_in_decoded_frame_pixels(
    width: int, height: int, right_box: tuple, left_box: tuple
) -> None:
    """A zone saved at 1920x1080 with the right half blocked drops right-half boxes of a
    smaller tap frame and keeps left-half ones."""
    masks = build_grid_masks_from_config([GridZoneConfig(**_zone("right", cells=RIGHT_HALF))])
    assert _run_one_frame(masks, right_box, width, height) == []
    kept = _run_one_frame(masks, left_box, width, height)
    assert [d.bbox for d in kept] == [left_box]


def test_area_zones_never_filter_detections(tmp_path: Path) -> None:
    every_cell = [(c, r) for c in range(16) for r in range(16)]
    cam = _camera([_zone("drive", kind="area", cells=every_cell)], detectors=[{"type": "motion"}])
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=tmp_path)
    assert bundle.grid_masks == []
    kept = _run_one_frame(bundle.grid_masks, (220, 70, 40, 40), 320, 180)
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# Web: the editor's kind control, save/delete write-back and rule protection
# ---------------------------------------------------------------------------


@pytest.fixture
def zones_config(tmp_path: Path) -> Path:
    """config.yaml with one camera: ignore zone `road`, area zones `drive` and `lawn`,
    and rule `night` that requires `drive`."""
    raw = {
        "cameras": [
            {
                "name": "yard",
                "main_url": URL,
                "proxy": {"enabled": False},
                "zones": [
                    _zone("road", cells=[(0, 0)]),
                    _zone("drive", kind="area", cells=[(1, 1)]),
                    _zone("lawn", kind="area"),
                ],
                "rules": [_rule("night", ["drive"])],
            }
        ],
        # Ignored by AppConfig until the actions task defines `actions`; then it backs `phone`.
        "actions": [{"name": "phone", "type": "ntfy", "url": "https://ntfy.example", "topic": "t"}],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def admin(db_with_user: str, zones_config: Path) -> TestClient:
    """Logged-in admin client whose app writes back to zones_config."""
    app = create_app(
        WebSettings(),
        cfg=load_config(zones_config),
        runtime_provider=lambda: None,
        config_path=zones_config,
    )
    client = TestClient(app)
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return client


def _save(client: TestClient, **fields: object):
    """POST the zone editor form for camera `yard` (16x16 at 1920x1080 unless overridden)."""
    token = client.cookies.get("warden_csrf", "")
    data = {
        "grid_cols": "16",
        "grid_rows": "16",
        "frame_width": "1920",
        "frame_height": "1080",
        "csrf_token": token,
        **fields,
    }
    return client.post(
        "/cameras/yard/zones", data=data, headers={"X-CSRF-Token": token}, follow_redirects=False
    )


def _delete(client: TestClient, zone_name: str):
    token = client.cookies.get("warden_csrf", "")
    return client.post(
        f"/cameras/yard/zones/{zone_name}/delete",
        data={"csrf_token": token},
        headers={"X-CSRF-Token": token},
        follow_redirects=False,
    )


def _zones_on_disk(path: Path) -> dict[str, dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {zone["name"]: zone for zone in raw["cameras"][0].get("zones", [])}


def _zones_in_memory(client: TestClient) -> dict[str, str]:
    return {zone.name: zone.kind for zone in client.app.state.cfg.cameras[0].zones}


def test_zone_to_dict_keeps_kind() -> None:
    zone = GridZoneConfig(**_zone("lawn", kind="area", cells=[(2, 3)]))
    assert _zone_to_dict(zone) == {
        "name": "lawn",
        "kind": "area",
        "grid_cols": 16,
        "grid_rows": 16,
        "blocked_cells": [[2, 3]],
        "frame_width": 1920,
        "frame_height": 1080,
        "enabled": True,
    }


def test_save_zone_stores_kind_area(admin: TestClient, zones_config: Path) -> None:
    r = _save(admin, zone_name="porch", kind="area", blocked_cell=["3,4"])
    assert r.status_code == 303
    on_disk = _zones_on_disk(zones_config)
    assert on_disk["porch"]["kind"] == "area"
    assert on_disk["porch"]["blocked_cells"] == [[3, 4]]
    assert _zones_in_memory(admin)["porch"] == "area"


def test_save_zone_without_kind_is_an_ignore_zone(admin: TestClient, zones_config: Path) -> None:
    r = _save(admin, zone_name="porch")
    assert r.status_code == 303
    assert _zones_on_disk(zones_config)["porch"]["kind"] == "ignore"


def test_save_zone_keeps_the_kind_of_other_zones(admin: TestClient, zones_config: Path) -> None:
    assert _save(admin, zone_name="porch").status_code == 303
    on_disk = _zones_on_disk(zones_config)
    assert (on_disk["road"]["kind"], on_disk["drive"]["kind"], on_disk["lawn"]["kind"]) == (
        "ignore",
        "area",
        "area",
    )


def test_save_zone_can_turn_an_unused_area_into_an_ignore_zone(
    admin: TestClient, zones_config: Path
) -> None:
    assert _save(admin, zone_name="lawn", kind="ignore").status_code == 303
    assert _zones_on_disk(zones_config)["lawn"]["kind"] == "ignore"


def test_save_zone_rejects_an_unknown_kind(admin: TestClient, zones_config: Path) -> None:
    before = zones_config.read_text(encoding="utf-8")
    r = _save(admin, zone_name="porch", kind="include")
    assert r.status_code == 422
    assert "kind must be 'ignore' or 'area'" in r.text
    assert zones_config.read_text(encoding="utf-8") == before


def test_save_zone_rejects_a_slash_in_the_name(admin: TestClient, zones_config: Path) -> None:
    before = zones_config.read_text(encoding="utf-8")
    r = _save(admin, zone_name="front/porch")
    assert r.status_code == 422
    assert "zone name must not contain '/'" in r.text
    assert zones_config.read_text(encoding="utf-8") == before
    assert "front/porch" not in _zones_in_memory(admin)


def test_save_zone_refuses_to_make_a_rule_zone_an_ignore_zone(
    admin: TestClient, zones_config: Path
) -> None:
    before = zones_config.read_text(encoding="utf-8")
    r = _save(admin, zone_name="drive", kind="ignore", blocked_cell=["1,1"])
    assert r.status_code == 422
    assert "rule 'night' names zone 'drive', which is an ignore zone" in r.text
    assert zones_config.read_text(encoding="utf-8") == before
    assert _zones_in_memory(admin)["drive"] == "area"


def test_delete_zone_named_by_a_rule_is_refused(admin: TestClient, zones_config: Path) -> None:
    before = zones_config.read_text(encoding="utf-8")
    r = _delete(admin, "drive")
    assert r.status_code == 422
    assert "rule 'night' names zone 'drive', which is not a zone of this camera" in r.text
    assert zones_config.read_text(encoding="utf-8") == before
    assert "drive" in _zones_in_memory(admin)


def test_delete_zone_not_named_by_a_rule_still_works(admin: TestClient, zones_config: Path) -> None:
    assert _delete(admin, "lawn").status_code == 303
    assert "lawn" not in _zones_on_disk(zones_config)
    assert "lawn" not in _zones_in_memory(admin)


def test_save_zone_without_kind_keeps_an_existing_zones_kind(
    admin: TestClient, zones_config: Path
) -> None:
    """Until the editor carries the kind control (after the RW-2 rebase), a save from it sends
    no kind: an existing area zone must stay an area zone."""
    assert _save(admin, zone_name="drive", blocked_cell=["1,1", "2,2"]).status_code == 303
    on_disk = _zones_on_disk(zones_config)
    assert on_disk["drive"]["kind"] == "area"
    assert on_disk["drive"]["blocked_cells"] == [[1, 1], [2, 2]]
    assert _zones_in_memory(admin)["drive"] == "area"


def test_zone_kind_partial_preselects_area() -> None:
    html = templates.get_template("partials/zone_kind.html").render(zone_kind="area")
    assert 'name="kind" value="area" checked' in html
    assert 'name="kind" value="ignore" checked' not in html


def test_zone_kind_partial_defaults_to_ignore() -> None:
    html = templates.get_template("partials/zone_kind.html").render()
    assert 'name="kind" value="ignore" checked' in html
    assert 'name="kind" value="area" checked' not in html


def _read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import rtsp_warden.web.routes.zones as zones_mod

    def fail(path: Path, data: object) -> None:
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(zones_mod, "_locked_write_yaml", fail)


def test_save_zone_write_failure_names_the_path(
    admin: TestClient, zones_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9: a read-only config.yaml is reported with its path, never as a 500."""
    _read_only(monkeypatch)
    r = _save(admin, zone_name="porch")
    assert r.status_code == 503
    assert str(zones_config) in r.text
    assert "Read-only file system" in r.text


def test_save_zone_write_failure_is_shown_to_htmx(
    admin: TestClient, zones_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The editor posts with htmx, which only swaps 2xx answers: the message comes back as 200."""
    _read_only(monkeypatch)
    token = admin.cookies.get("warden_csrf", "")
    r = admin.post(
        "/cameras/yard/zones",
        data={"zone_name": "porch", "grid_cols": "16", "grid_rows": "16", "csrf_token": token},
        headers={"X-CSRF-Token": token, "HX-Request": "true"},
        follow_redirects=False,
    )
    assert r.status_code == 200
    assert str(zones_config) in r.text
    assert "until the next restart" in r.text


def test_delete_zone_write_failure_names_the_path(
    admin: TestClient, zones_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _read_only(monkeypatch)
    r = _delete(admin, "lawn")
    assert r.status_code == 503
    assert str(zones_config) in r.text

"""Registry branch for ``type: onnx`` and the runtime's ``models_dir`` wiring.

Nothing here loads a model: building a detector reads only the descriptor.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from rtsp_warden import app as app_module
from rtsp_warden.config import AppConfig, CameraConfig, RuntimeConfig
from rtsp_warden.detectors.builtin.onnx import OnnxDetector
from rtsp_warden.detectors.registry import (
    DEFAULT_ONNX_MODEL,
    CameraDetectorBundle,
    DetectorSpec,
    build_detector,
    build_detector_with_sensitivity,
    build_detectors_for_camera,
)
from rtsp_warden.detectors.sensitivity import apply_sensitivity_to_confidence
from tests.helpers_onnx import write_model_dir, yolox_output

LN2 = math.log(2.0)
INPUT = (64, 64)


def _tiny_model(models_dir: Path) -> None:
    rows = {19: [0.5, 0.5, LN2, LN2, 0.9, 0.8, 0.1]}
    write_model_dir(models_dir, yolox_output(INPUT, 2, rows))


def test_registry_onnx_defaults_to_yolox_s(tmp_path: Path) -> None:
    det = build_detector_with_sensitivity(
        DetectorSpec(type="onnx"), "cam", models_dir=tmp_path / "models"
    )

    assert isinstance(det, OnnxDetector)
    assert det.descriptor.name == DEFAULT_ONNX_MODEL == "yolox-s"
    assert det.input_width == 640
    assert det.device == "auto"
    assert det.min_confidence == pytest.approx(0.5)  # camera sensitivity 50
    assert det.classes is None
    assert det.models_dir == tmp_path / "models"
    assert det.loaded is False  # nothing downloaded at build time


def test_registry_onnx_uses_spec_and_camera_settings(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _tiny_model(models_dir)
    spec = DetectorSpec(type="onnx", model="tiny", device="cpu", min_confidence=0.6)

    det = build_detector_with_sensitivity(
        spec,
        "cam",
        camera_sensitivity=90.0,
        camera_detect_classes=["person"],
        models_dir=models_dir,
    )

    assert isinstance(det, OnnxDetector)
    assert det.descriptor.name == "tiny"
    assert det.device == "cpu"
    assert det.min_confidence == pytest.approx(0.6)  # spec wins over sensitivity
    assert det.classes == ["person"]


def test_registry_onnx_maps_camera_sensitivity_without_min_confidence(tmp_path: Path) -> None:
    det = build_detector_with_sensitivity(
        DetectorSpec(type="onnx", model="yolox-nano"),
        "cam",
        camera_sensitivity=80.0,
        models_dir=tmp_path,
    )

    assert isinstance(det, OnnxDetector)
    assert det.input_width == 416
    assert det.min_confidence == pytest.approx(apply_sensitivity_to_confidence(80.0))


def test_build_detectors_for_camera_passes_models_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _tiny_model(models_dir)
    cam = CameraConfig(
        name="cam",
        main_url="rtsp://u:p@h/m",
        detect_classes=["car"],
        detectors=[DetectorSpec(type="onnx", model="tiny")],
    )

    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models_dir)

    assert len(bundle.detectors) == 1
    det = bundle.detectors[0]
    assert isinstance(det, OnnxDetector)
    assert det.descriptor.name == "tiny"
    assert det.models_dir == models_dir
    assert det.classes == ["car"]


def test_unknown_model_is_skipped_not_raised(tmp_path: Path) -> None:
    cam = CameraConfig(
        name="cam",
        main_url="rtsp://u:p@h/m",
        detectors=[DetectorSpec(type="onnx", model="no-such-model")],
    )

    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=tmp_path)

    assert bundle.detectors == []


def test_legacy_build_detector_handles_onnx() -> None:
    det = build_detector(DetectorSpec(type="onnx"), "cam")

    assert isinstance(det, OnnxDetector)
    assert det.descriptor.name == "yolox-s"
    assert det.models_dir == RuntimeConfig().models_dir


def test_app_runtime_passes_runtime_models_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path | None] = []

    def fake_build(
        camera_cfg: CameraConfig, base_specs: list[DetectorSpec], *, models_dir: Path | None = None
    ) -> CameraDetectorBundle:
        seen.append(models_dir)
        return CameraDetectorBundle()

    monkeypatch.setattr(app_module, "build_detectors_for_camera", fake_build)
    cfg = AppConfig(
        cameras=[
            CameraConfig(
                name="cam",
                main_url="rtsp://u:p@h/m",
                detectors=[DetectorSpec(type="motion")],
            )
        ],
        runtime=RuntimeConfig(models_dir=tmp_path / "models"),
    )
    rt = app_module.AppRuntime(cfg=cfg)

    rt.build()
    after_build = len(seen)
    rt.rebuild_camera_detectors("cam")

    assert after_build >= 1
    assert len(seen) > after_build
    assert set(seen) == {tmp_path / "models"}


# --- RW-5: per-slot classes -----------------------------------------------------------------------

URL = "rtsp://u:p@h/m"


def _labelled_model(models_dir: Path, name: str, labels: list[str]) -> None:
    write_model_dir(models_dir, yolox_output(INPUT, len(labels), {}), name=name, labels=labels)


def test_spec_classes_intersect_with_camera_detect_classes(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _labelled_model(models, "wild", ["cat", "fox", "raccoon", "person"])
    cam = CameraConfig(
        name="yard",
        main_url=URL,
        detect_classes=["fox", "person", "car"],
        detectors=[DetectorSpec(type="onnx", model="wild", classes=["cat", "fox", "raccoon"])],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models)
    assert bundle.detectors[0].classes == ["fox"]


def test_spec_classes_alone_filter_the_model(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _labelled_model(models, "wild", ["cat", "fox", "raccoon", "person"])
    cam = CameraConfig(
        name="yard",
        main_url=URL,
        detectors=[DetectorSpec(type="onnx", model="wild", classes=["raccoon"])],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models)
    assert bundle.detectors[0].classes == ["raccoon"]


def test_empty_intersection_builds_a_silent_detector(tmp_path: Path) -> None:
    """(review focus) Nothing in common means "report nothing", never a startup error."""
    models = tmp_path / "models"
    _labelled_model(models, "wild", ["cat", "fox"])
    cam = CameraConfig(
        name="yard",
        main_url=URL,
        detect_classes=["person"],
        detectors=[DetectorSpec(type="onnx", model="wild", classes=["fox"])],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models)
    det = bundle.detectors[0]
    assert isinstance(det, OnnxDetector) and det.classes == []
    det.setup()
    assert det.process(np.zeros((64, 64, 3), dtype=np.uint8), 1.0) == []


def test_overlapping_classes_load_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """(review focus) Two models allowed to report one label are accepted, with a warning."""
    models = tmp_path / "models"
    _labelled_model(models, "a", ["dog", "cat"])
    _labelled_model(models, "b", ["dog", "fox"])
    cam = CameraConfig(
        name="yard",
        main_url=URL,
        detectors=[
            DetectorSpec(type="onnx", model="a", classes=["dog"]),
            DetectorSpec(type="onnx", model="b"),
        ],
    )
    with caplog.at_level("WARNING", logger="rtsp_warden.detectors.registry"):
        bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models)
    assert len(bundle.detectors) == 2
    assert "both report 'dog'" in caplog.text
    assert "'fox'" not in caplog.text

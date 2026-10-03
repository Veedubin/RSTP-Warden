"""runtime.models_dir and validation of detect_classes / rules[].labels against model labels."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rtsp_warden.config import AppConfig, CameraConfig, RuntimeConfig, load_config
from rtsp_warden.detectors.model_registry import (
    ModelNotFound,
    camera_label_universe,
    camera_model_labels,
)

URL = "rtsp://u:p@h/m"


@pytest.fixture(autouse=True)
def default_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The default models_dir is an empty temp dir, so a developer's cache never matters."""
    models = tmp_path / "default-models"
    monkeypatch.setenv("WARDEN_MODELS_DIR", str(models))
    return models


def cam(**fields: object) -> dict:
    return {"name": "yard", "main_url": URL, **fields}


def app(*cameras: dict, models_dir: Path | None = None) -> dict:
    raw: dict = {"cameras": list(cameras)}
    if models_dir is not None:
        raw["runtime"] = {"models_dir": str(models_dir)}
    return raw


def rule(name: str, labels: list[str]) -> dict:
    return {"name": name, "labels": labels, "actions": []}


def write_model(models_dir: Path, name: str, label_lines: list[str], **fields: object) -> None:
    d = models_dir / name
    d.mkdir(parents=True, exist_ok=True)
    desc: dict[str, object] = {
        "name": name,
        "file": f"{name}.onnx",
        "labels": "labels.txt",
        "input_size": [64, 64],
        "postprocess": "yolox",
    }
    desc.update(fields)
    (d / "model.yaml").write_text(yaml.safe_dump(desc), encoding="utf-8")
    (d / "labels.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")


# --- runtime.models_dir --------------------------------------------------------------------


def test_models_dir_default_comes_from_the_environment(default_models: Path) -> None:
    assert RuntimeConfig().models_dir == default_models
    assert AppConfig.model_validate(app(cam())).runtime.models_dir == default_models


def test_models_dir_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert RuntimeConfig(models_dir="~/models").models_dir == tmp_path / "models"


def test_models_dir_round_trips_through_a_dump(tmp_path: Path) -> None:
    cfg = AppConfig.model_validate(app(cam(), models_dir=tmp_path / "m"))
    dumped = cfg.model_dump(mode="json")
    assert dumped["runtime"]["models_dir"] == str(tmp_path / "m")
    assert AppConfig.model_validate(dumped).runtime.models_dir == tmp_path / "m"


# --- detect_classes --------------------------------------------------------------------------


def test_detect_classes_known_labels_pass() -> None:
    cfg = AppConfig.model_validate(
        app(cam(detectors=[{"type": "onnx"}], detect_classes=["person", "cat"]))
    )
    assert cfg.cameras[0].detect_classes == ["person", "cat"]


def test_detect_classes_unknown_label_lists_the_model_labels() -> None:
    with pytest.raises(ValidationError) as ei:
        AppConfig.model_validate(
            app(cam(detectors=[{"type": "onnx"}], detect_classes=["person", "unicorn"]))
        )
    msg = str(ei.value)
    assert "camera 'yard': unknown detect_classes ['unicorn']" in msg
    assert "yolox-s labels: person, bicycle, car," in msg
    assert "toothbrush" in msg


def test_detect_classes_typo_suggests_the_closest_label() -> None:
    with pytest.raises(ValidationError, match=r"did you mean 'persn' -> 'person'\?"):
        AppConfig.model_validate(app(cam(detectors=[{"type": "onnx"}], detect_classes=["persn"])))


@pytest.mark.parametrize("detectors", [[], [{"type": "motion"}], [{"type": "dnn"}]])
def test_detect_classes_not_checked_without_an_onnx_detector(detectors: list[dict]) -> None:
    cfg = AppConfig.model_validate(app(cam(detectors=detectors, detect_classes=["unicorn"])))
    assert cfg.cameras[0].detect_classes == ["unicorn"]


@pytest.mark.parametrize("classes", [None, []])
def test_detect_classes_none_and_empty_pass(classes: list[str] | None) -> None:
    AppConfig.model_validate(app(cam(detectors=[{"type": "onnx"}], detect_classes=classes)))


def test_disabled_onnx_detector_still_defines_the_labels() -> None:
    disabled = [{"type": "onnx", "enabled": False}]
    AppConfig.model_validate(app(cam(detectors=disabled, detect_classes=["cat"])))
    with pytest.raises(ValidationError, match="unknown detect_classes"):
        AppConfig.model_validate(app(cam(detectors=disabled, detect_classes=["unicorn"])))


def test_several_models_validate_against_the_union(tmp_path: Path) -> None:
    models = tmp_path / "m"
    write_model(models, "critters", ["raccoon", "fox"])
    detectors = [{"type": "onnx", "model": "yolox-nano"}, {"type": "onnx", "model": "critters"}]

    AppConfig.model_validate(
        app(cam(detectors=detectors, detect_classes=["person", "raccoon"]), models_dir=models)
    )
    with pytest.raises(ValidationError) as ei:
        AppConfig.model_validate(
            app(cam(detectors=detectors, detect_classes=["unicorn"]), models_dir=models)
        )
    msg = str(ei.value)
    assert "yolox-nano labels: person," in msg
    assert "critters labels: raccoon, fox" in msg


def test_unknown_model_is_a_config_error_naming_the_available_ones() -> None:
    with pytest.raises(ValidationError) as ei:
        AppConfig.model_validate(app(cam(detectors=[{"type": "onnx", "model": "yolox-m"}])))
    msg = str(ei.value)
    assert "camera 'yard': unknown model 'yolox-m'; available: yolox-nano, yolox-s" in msg


def test_unknown_postprocess_in_a_user_model_is_a_config_error(tmp_path: Path) -> None:
    write_model(tmp_path, "ssd-model", ["a"], postprocess="ssd")
    with pytest.raises(ValidationError, match="unknown postprocess 'ssd'; supported: yolox"):
        AppConfig.model_validate(
            app(cam(detectors=[{"type": "onnx", "model": "ssd-model"}]), models_dir=tmp_path)
        )


def test_validation_never_creates_models_dir_or_needs_the_model_file(tmp_path: Path) -> None:
    models = tmp_path / "never-created"
    AppConfig.model_validate(
        app(cam(detectors=[{"type": "onnx"}], detect_classes=["dog"]), models_dir=models)
    )
    assert not models.exists()


# --- rules[].labels ------------------------------------------------------------------------------


def test_rule_labels_accept_model_labels_and_motion() -> None:
    cfg = AppConfig.model_validate(
        app(cam(detectors=[{"type": "onnx"}], rules=[rule("r1", ["person", "motion"])]))
    )
    assert cfg.cameras[0].rules[0].labels == ["person", "motion"]


def test_rule_labels_unknown_label_fails() -> None:
    with pytest.raises(ValidationError, match=r"unknown rule 'r1' labels \['unicorn'\]"):
        AppConfig.model_validate(
            app(cam(detectors=[{"type": "onnx"}], rules=[rule("r1", ["unicorn"])]))
        )


def test_rule_labels_not_checked_without_an_onnx_detector() -> None:
    AppConfig.model_validate(app(cam(detectors=[{"type": "motion"}], rules=[rule("r1", ["x"])])))


# --- error hygiene and load_config --------------------------------------------------------------


def test_validation_error_does_not_echo_the_input() -> None:
    with pytest.raises(ValidationError) as ei:
        AppConfig.model_validate(app(cam(detectors=[{"type": "onnx"}], detect_classes=["unicorn"])))
    msg = str(ei.value)
    assert "input_value" not in msg
    assert "@h/m" not in msg


def test_nested_camera_error_does_not_echo_a_camera_url() -> None:
    with pytest.raises(ValidationError) as ei:
        AppConfig.model_validate({"cameras": [{"name": "yard", "sub_url": "rtsp://u:p@h/s"}]})
    msg = str(ei.value)
    assert "cameras.0.main_url" in msg
    assert "rtsp://" not in msg


def test_load_config_turns_an_unknown_label_into_system_exit(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(app(cam(detectors=[{"type": "onnx"}], detect_classes=["unicorn"]))),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as ei:
        load_config(path)
    assert str(ei.value).startswith("Config validation failed:\n")
    assert "unknown detect_classes ['unicorn']" in str(ei.value)


# --- camera_model_labels / camera_label_universe -----------------------------------------------


def test_camera_label_universe_is_none_without_onnx(tmp_path: Path) -> None:
    c = CameraConfig.model_validate(cam(detectors=[{"type": "motion"}]))
    assert camera_model_labels(c, tmp_path) == {}
    assert camera_label_universe(c, tmp_path) is None


def test_camera_label_universe_default_model_is_coco(tmp_path: Path) -> None:
    c = CameraConfig.model_validate(cam(detectors=[{"type": "onnx"}]))
    assert list(camera_model_labels(c, tmp_path)) == ["yolox-s"]
    universe = camera_label_universe(c, tmp_path)
    assert universe is not None
    assert len(universe) == 80
    assert {"person", "car", "dog"} <= universe


def test_camera_model_labels_keeps_first_use_order_without_duplicates(tmp_path: Path) -> None:
    write_model(tmp_path, "critters", ["raccoon"])
    c = CameraConfig.model_validate(
        cam(
            detectors=[
                {"type": "onnx"},
                {"type": "onnx", "model": "critters"},
                {"type": "onnx", "model": "yolox-s", "enabled": False},
            ]
        )
    )
    per_model = camera_model_labels(c, tmp_path)
    assert list(per_model) == ["yolox-s", "critters"]
    assert per_model["critters"] == ["raccoon"]


def test_camera_label_universe_unknown_model_raises(tmp_path: Path) -> None:
    c = CameraConfig.model_validate(cam(detectors=[{"type": "onnx", "model": "nope"}]))
    with pytest.raises(ModelNotFound):
        camera_label_universe(c, tmp_path)

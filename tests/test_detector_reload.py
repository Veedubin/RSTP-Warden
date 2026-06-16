"""Tests for detector rebuild: build_detectors_for_camera, sensitivity, class filter integration."""

from __future__ import annotations

from rtsp_warden.config import CameraConfig, DetectorSpec, GridZoneConfig
from rtsp_warden.detectors.class_filter import effective_classes
from rtsp_warden.detectors.grid_mask import GridMask
from rtsp_warden.detectors.registry import (
    CameraDetectorBundle,
    build_detector_with_sensitivity,
    build_detectors_for_camera,
    build_grid_masks_from_config,
)
from rtsp_warden.detectors.roi import Mask
from rtsp_warden.detectors.sensitivity import (
    apply_sensitivity_to_confidence,
    apply_sensitivity_to_motion,
    apply_sensitivity_to_nms,
)

# ---------------------------------------------------------------------------
# build_detectors_for_camera
# ---------------------------------------------------------------------------


def test_build_detectors_respects_spec_enabled() -> None:
    """Disabled specs are skipped entirely."""
    cam = CameraConfig(
        name="test_cam",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        detectors=[
            DetectorSpec(type="motion", enabled=False),
        ],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors)
    # No enabled detectors -> empty bundle
    assert len(bundle.detectors) == 0


def test_build_detectors_applies_sensitivity_to_motion() -> None:
    """Camera sensitivity is applied to MotionDetector varThreshold."""
    cam = CameraConfig(
        name="test_cam",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        sensitivity=80.0,
        detectors=[
            DetectorSpec(type="motion", enabled=True),
        ],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors)
    assert len(bundle.detectors) == 1
    det = bundle.detectors[0]
    assert det.name == "motion"
    # sensitivity=80 -> varThreshold = apply_sensitivity_to_motion(80)
    assert det.var_threshold == apply_sensitivity_to_motion(80.0)


def test_build_detectors_applies_sensitivity_to_dnn() -> None:
    """Camera sensitivity is applied to DNN confidence and NMS."""
    cam = CameraConfig(
        name="test_cam",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        sensitivity=75.0,
        detectors=[
            DetectorSpec(type="dnn", enabled=True, config={}),
        ],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors)
    assert len(bundle.detectors) == 1
    det = bundle.detectors[0]
    assert det.name == "dnn"
    expected_conf = apply_sensitivity_to_confidence(75.0)
    expected_nms = apply_sensitivity_to_nms(75.0)
    assert abs(det.confidence - expected_conf) < 0.01
    assert abs(det.nms_threshold - expected_nms) < 0.01


def test_build_detectors_intersects_detect_classes() -> None:
    """Camera detect_classes intersects with detector's allowed_classes."""
    cam = CameraConfig(
        name="test_cam",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        detect_classes=["person", "dog"],
        detectors=[
            DetectorSpec(
                type="dnn",
                enabled=True,
                config={"classes": ["person", "car", "truck"]},
            ),
        ],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors)
    det = bundle.detectors[0]
    # Intersection of ["person", "dog"] and ["person", "car", "truck"]
    # -> ["person"]
    assert det.allowed_classes == ["person"]


def test_build_detectors_creates_grid_masks() -> None:
    """GridMasks are built from camera zone config."""
    cam = CameraConfig(
        name="test_cam",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        zones=[
            GridZoneConfig(
                name="exclude_road",
                grid_cols=8,
                grid_rows=8,
                blocked_cells={(3, 4), (3, 5)},
                frame_width=1920,
                frame_height=1080,
            ),
        ],
        detectors=[
            DetectorSpec(type="motion", enabled=True),
        ],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors)
    assert len(bundle.grid_masks) == 1
    assert bundle.grid_masks[0].name == "exclude_road"
    assert bundle.grid_masks[0].is_cell_blocked(3, 4)
    assert bundle.grid_masks[0].is_cell_blocked(3, 5)


def test_build_detectors_skips_disabled_zones() -> None:
    """Disabled zones are not included in grid masks."""
    cam = CameraConfig(
        name="test_cam",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        zones=[
            GridZoneConfig(
                name="active_zone",
                grid_cols=8,
                grid_rows=8,
                blocked_cells={(0, 0)},
                frame_width=1920,
                frame_height=1080,
                enabled=True,
            ),
            GridZoneConfig(
                name="disabled_zone",
                grid_cols=8,
                grid_rows=8,
                blocked_cells={(1, 1)},
                frame_width=1920,
                frame_height=1080,
                enabled=False,
            ),
        ],
        detectors=[
            DetectorSpec(type="motion", enabled=True),
        ],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors)
    assert len(bundle.grid_masks) == 1
    assert bundle.grid_masks[0].name == "active_zone"


def test_build_detectors_detect_classes_with_none_detector() -> None:
    """Camera detect_classes with DNN detector that has no config.classes."""
    cam = CameraConfig(
        name="test_cam",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        detect_classes=["person"],
        detectors=[
            DetectorSpec(type="dnn", enabled=True, config={}),
        ],
    )
    bundle = build_detectors_for_camera(cam, cam.detectors)
    det = bundle.detectors[0]
    # Camera classes with no detector classes -> use camera's list
    assert det.allowed_classes == ["person"]


# ---------------------------------------------------------------------------
# build_detector_with_sensitivity
# ---------------------------------------------------------------------------


def test_build_detector_with_sensitivity_motion_default() -> None:
    """MotionDetector with default sensitivity uses camera_sensitivity mapping."""
    spec = DetectorSpec(type="motion", enabled=True)
    det = build_detector_with_sensitivity(spec, "cam1", camera_sensitivity=80.0)
    assert det.name == "motion"
    assert det.var_threshold == apply_sensitivity_to_motion(80.0)


def test_build_detector_with_sensitivity_motion_spec_override() -> None:
    """MotionDetector with spec.sensitivity uses that value."""
    spec = DetectorSpec(type="motion", enabled=True, sensitivity=0.3)
    det = build_detector_with_sensitivity(spec, "cam1", camera_sensitivity=80.0)
    assert det.name == "motion"
    # Spec override: sensitivity=0.3, var_threshold should be None
    assert det.var_threshold is None
    assert abs(det.sensitivity - 0.3) < 0.01


def test_build_detector_with_sensitivity_person() -> None:
    """PersonDetector uses camera sensitivity for min_confidence when not set."""
    spec = DetectorSpec(type="person", enabled=True)
    det = build_detector_with_sensitivity(spec, "cam1", camera_sensitivity=75.0)
    expected_conf = apply_sensitivity_to_confidence(75.0)
    assert abs(det.min_confidence - expected_conf) < 0.01


def test_build_detector_with_sensitivity_person_spec_override() -> None:
    """PersonDetector with spec.min_confidence uses that value."""
    spec = DetectorSpec(type="person", enabled=True, min_confidence=0.8)
    det = build_detector_with_sensitivity(spec, "cam1", camera_sensitivity=75.0)
    assert det.min_confidence == 0.8


# ---------------------------------------------------------------------------
# build_grid_masks_from_config
# ---------------------------------------------------------------------------


def test_build_grid_masks_empty_zones() -> None:
    """No zones produces empty list."""
    result = build_grid_masks_from_config([])
    assert result == []


def test_build_grid_masks_from_config_creates_masks() -> None:
    """GridZoneConfig is converted to GridMask correctly."""
    zones = [
        GridZoneConfig(
            name="test_zone",
            grid_cols=16,
            grid_rows=16,
            blocked_cells={(0, 0), (15, 15)},
            frame_width=1920,
            frame_height=1080,
        ),
    ]
    result = build_grid_masks_from_config(zones)
    assert len(result) == 1
    assert result[0].name == "test_zone"
    assert result[0].grid_cols == 16
    assert result[0].is_cell_blocked(0, 0)
    assert result[0].is_cell_blocked(15, 15)


# ---------------------------------------------------------------------------
# CameraDetectorBundle
# ---------------------------------------------------------------------------


def test_camera_detector_bundle_defaults() -> None:
    """CameraDetectorBundle has sensible defaults."""
    bundle = CameraDetectorBundle()
    assert bundle.detectors == []
    assert bundle.masks == []
    assert bundle.roi is None
    assert bundle.grid_masks == []


def test_camera_detector_bundle_with_data() -> None:
    """CameraDetectorBundle holds detectors, masks, roi, and grid_masks."""
    mask = Mask(polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
    gm = GridMask(grid_cols=4, grid_rows=4, frame_width=400, frame_height=400)
    bundle = CameraDetectorBundle(
        detectors=[],
        masks=[mask],
        roi=None,
        grid_masks=[gm],
    )
    assert len(bundle.masks) == 1
    assert len(bundle.grid_masks) == 1


# ---------------------------------------------------------------------------
# effective_classes integration with DNN
# ---------------------------------------------------------------------------


def test_effective_classes_empty_intersection_logged() -> None:
    """Empty intersection between camera and detector classes produces []."""
    result = effective_classes(["person"], ["car", "truck"])
    assert result == []


def test_effective_classes_none_none() -> None:
    """Both None means no filter."""
    result = effective_classes(None, None)
    assert result is None


def test_effective_classes_camera_only() -> None:
    """Only camera classes set -> use camera's list."""
    result = effective_classes(["person", "dog"], None)
    assert result == ["person", "dog"]


def test_effective_classes_detector_only() -> None:
    """Only detector classes set -> use detector's list."""
    result = effective_classes(None, ["car", "truck"])
    assert result == ["car", "truck"]

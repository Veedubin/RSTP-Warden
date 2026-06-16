"""Camera route handlers for the rtsp-warden web UI.

Provides camera list, camera detail, camera status partial
for htmx auto-refresh, per-camera retention policy management,
sensitivity adjustment, detection class configuration, and
per-detector enable/disable toggles.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from ...config import AppConfig, CameraConfig, DetectorSpec, RetentionConfig
from ..auth_depends import CurrentUser, require_admin, require_user
from ..config_lock import _locked_write_yaml
from ..paths import TEMPLATES_DIR
from ..services.cameras import get_camera_by_name, get_camera_detectors, list_cameras
from ..services.recordings import list_recordings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 80 COCO class names used by YOLOv4-tiny DNN detectors.
COCO_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Category groupings for the detection-classes checkbox UI.
_CLASS_CATEGORIES: dict[str, list[str]] = {
    "person": ["person"],
    "pet": ["cat", "dog"],
    "vehicle": [
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ],
    "critter": [
        "bird",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "other": [
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ],
}


def _group_classes_for_template() -> list[tuple[str, list[str]]]:
    """Group COCO classes into categories for the detection-classes template.

    Returns:
        List of (category_name, class_names) tuples.
    """
    return [(cat, classes) for cat, classes in _CLASS_CATEGORIES.items()]


def _get_cfg(request: Request) -> AppConfig:
    """Get the AppConfig from app state, raising 503 if unavailable."""
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not loaded")
    return cfg


def _get_config_path(request: Request) -> Path | None:
    """Try to determine the config file path from app state.

    Returns None if the path is not available (in-memory config only).
    """
    config_path = getattr(request.app.state, "config_path", None)
    if config_path is not None:
        return Path(config_path)
    return None


def _find_camera_config(cfg: AppConfig, name: str) -> CameraConfig | None:
    """Find a CameraConfig object by name."""
    for cam in cfg.cameras:
        if cam.name == name:
            return cam
    return None


@router.get("", response_class=HTMLResponse)
async def cameras_list(request: Request, user=Depends(require_user)) -> HTMLResponse:
    """Render the camera grid page."""
    cfg = _get_cfg(request)
    cameras = list_cameras(cfg)
    return _templates.TemplateResponse(
        request,
        "cameras/list.html",
        {
            "request": request,
            "cameras": cameras,
        },
    )


@router.get("/{name}", response_class=HTMLResponse)
async def camera_detail(request: Request, name: str, user=Depends(require_user)) -> HTMLResponse:
    """Render a single camera detail page."""
    cfg = _get_cfg(request)
    cam = get_camera_by_name(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Build MJPEG and snapshot URLs for proxy-enabled cameras
    mjpeg_url = ""
    snapshot_url = ""
    if cam["has_proxy"] and cam["proxy_mode"] == "mjpeg":
        host = cam.get("bind_host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = cam["proxy_port"]
        mjpeg_url = f"http://{host}:{port}/mjpeg"
        snapshot_url = f"http://{host}:{port}/snapshot.jpg"

    # Recent recordings for this camera (last 10)
    try:
        recent_recordings, _ = list_recordings(camera_name=name, limit=10)
    except Exception:
        recent_recordings = []

    # Retention info for the detail page
    cam_config = _find_camera_config(cfg, name)
    from ...retention_resolver import resolve_retention

    effective_retention = (
        resolve_retention(cam_config, cfg.retention) if cam_config else cfg.retention
    )
    has_per_camera_retention = cam_config.retention is not None if cam_config else False

    # Sensitivity and detect_classes for the detail page
    sensitivity = cam_config.sensitivity if cam_config else 50.0
    detect_classes = cam_config.detect_classes if cam_config else None

    return _templates.TemplateResponse(
        request,
        "cameras/detail.html",
        {
            "request": request,
            "camera": cam,
            "mjpeg_url": mjpeg_url,
            "snapshot_url": snapshot_url,
            "recent_recordings": recent_recordings,
            "effective_retention": effective_retention,
            "has_per_camera_retention": has_per_camera_retention,
            "global_retention": cfg.retention,
            "sensitivity": sensitivity,
            "detect_classes": detect_classes,
        },
    )


@router.get("/{name}/status", response_class=HTMLResponse)
async def camera_status(request: Request, name: str, user=Depends(require_user)) -> HTMLResponse:
    """Return a partial camera card for htmx auto-refresh."""
    cfg = _get_cfg(request)
    cam = get_camera_by_name(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    return _templates.TemplateResponse(
        request,
        "partials/camera_card.html",
        {
            "request": request,
            "camera": cam,
        },
    )


@router.get("/{name}/settings", response_class=HTMLResponse)
async def camera_settings(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> HTMLResponse:
    """Render the read-only camera settings page (admin-only).

    Displays the camera's full configuration and a banner explaining
    that changes require editing config.yaml and restarting the server.
    """
    cfg = _get_cfg(request)
    # Find the raw CameraConfig object (not the dict from get_camera_by_name)
    cam_config: CameraConfig | None = None
    for cam in cfg.cameras:
        if cam.name == name:
            cam_config = cam
            break

    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Build a display-friendly dict (redact URLs)
    from ...status_model import redact_rtsp_url

    settings_data = {
        "name": cam_config.name,
        "main_url_redacted": redact_rtsp_url(cam_config.main_url),
        "sub_url_redacted": redact_rtsp_url(cam_config.sub_url),
        "record_enabled": cam_config.record.enabled,
        "record_output_dir": str(cam_config.record.output_dir),
        "record_container": cam_config.record.main.container,
        "record_chunk_seconds": cam_config.record.main.chunk_seconds,
        "record_transport": cam_config.record.main.rtsp_transport,
        "proxy_enabled": cam_config.proxy.enabled,
        "proxy_mode": cam_config.proxy.mode,
        "proxy_stream": cam_config.proxy.stream,
        "proxy_bind_host": cam_config.proxy.bind_host,
        "proxy_port": cam_config.proxy.port,
        "proxy_fps": cam_config.proxy.fps,
    }

    # Retention settings
    retention = cam_config.record.retention
    settings_data["retention_max_days"] = retention.max_days
    settings_data["retention_max_gb"] = retention.max_gb
    settings_data["retention_keep_last_n"] = retention.keep_last_n

    return _templates.TemplateResponse(
        request,
        "cameras/settings.html",
        {
            "request": request,
            "camera_name": name,
            "settings": settings_data,
        },
    )


@router.get("/{name}/detectors", response_class=HTMLResponse)
async def cameras_detectors_partial(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_user),
) -> HTMLResponse:
    """htmx partial -- refresh detector list for one camera.

    Returns the partial template HTML directly.
    """
    cfg = _get_cfg(request)
    detectors = get_camera_detectors(cfg, name)
    has_roi = any(d["has_roi"] for d in detectors)
    has_masks = any(d["has_masks"] for d in detectors)
    num_masks = sum(1 for d in detectors if d["has_masks"])

    return _templates.TemplateResponse(
        request,
        "partials/detector_list.html",
        {
            "request": request,
            "detectors": detectors,
            "camera_name": name,
            "has_roi": has_roi,
            "has_masks": has_masks,
            "num_masks": num_masks,
        },
    )


@router.post("/{name}/retention")
async def save_camera_retention(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> RedirectResponse:
    """Save or clear per-camera retention override.

    Form fields:
        max_days, max_gb, keep_last_n, cleanup_interval_seconds

    Special form field:
        action=reset  --  clears the per-camera override (falls back to global)

    On success, redirects to the camera detail page.
    """
    cfg = _get_cfg(request)
    cam_config = _find_camera_config(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    config_path = _get_config_path(request)
    form = await request.form()

    # Handle "reset to global" action
    action = form.get("action")
    if action == "reset":
        cam_config.retention = None
        if config_path is not None:
            _persist_camera_retention(config_path, cfg)
        return RedirectResponse(url=f"/cameras/{name}", status_code=303)

    # Parse retention fields from form
    max_days_raw = form.get("max_days")
    max_gb_raw = form.get("max_gb")
    keep_last_n_raw = form.get("keep_last_n")
    cleanup_interval_raw = form.get("cleanup_interval_seconds")

    max_days: int | None = int(max_days_raw) if max_days_raw and str(max_days_raw).strip() else None
    max_gb: float | None = float(max_gb_raw) if max_gb_raw and str(max_gb_raw).strip() else None
    keep_last_n: int = (
        int(keep_last_n_raw) if keep_last_n_raw and str(keep_last_n_raw).strip() else 0
    )
    cleanup_interval_seconds: int = (
        int(cleanup_interval_raw)
        if cleanup_interval_raw and str(cleanup_interval_raw).strip()
        else 300
    )

    cam_config.retention = RetentionConfig(
        max_days=max_days,
        max_gb=max_gb,
        keep_last_n=keep_last_n,
        cleanup_interval_seconds=cleanup_interval_seconds,
    )

    if config_path is not None:
        _persist_camera_retention(config_path, cfg)

    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


@router.post("/{name}/reload")
async def reload_camera_detectors(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> JSONResponse:
    """Hot-reload detectors for a camera.

    Rebuilds the detector runner for the named camera using the current
    in-memory config and atomically swaps it into the active runtime.
    No server restart required.

    Returns:
        JSON response with status and camera name.
    """
    from ...app import AppRuntime

    app_rt: AppRuntime | None = getattr(request.app.state, "runtime", None)
    if app_rt is None:
        raise HTTPException(status_code=503, detail="Server runtime not initialized")

    cfg = _get_cfg(request)
    cam_config = _find_camera_config(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    try:
        app_rt.rebuild_camera_detectors(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.error("failed to reload detectors for camera %s: %s", name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload detectors: {exc}") from exc

    return JSONResponse({"ok": True, "camera": name})


@router.get("/{name}/sensitivity", response_class=HTMLResponse)
async def camera_sensitivity_page(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Render the sensitivity adjustment page for a camera (admin-only).

    Shows a slider (0-100) and per-detector mapping preview.
    """
    cfg = _get_cfg(request)
    cam_config = _find_camera_config(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Build per-detector mapping preview
    from ...detectors.sensitivity import (
        apply_sensitivity_to_confidence,
        apply_sensitivity_to_motion,
        apply_sensitivity_to_nms,
    )

    sensitivity = cam_config.sensitivity
    detector_mappings: list[dict[str, str]] = []
    for spec in cam_config.detectors:
        if spec.type == "motion":
            var_th = apply_sensitivity_to_motion(sensitivity)
            detector_mappings.append(
                {
                    "type": "motion",
                    "param": "varThreshold",
                    "value": str(var_th),
                }
            )
        elif spec.type in ("person", "vehicle"):
            conf = apply_sensitivity_to_confidence(sensitivity)
            detector_mappings.append(
                {
                    "type": spec.type,
                    "param": "min_confidence",
                    "value": f"{conf:.2f}",
                }
            )
        elif spec.type == "dnn":
            conf = apply_sensitivity_to_confidence(sensitivity)
            nms = apply_sensitivity_to_nms(sensitivity)
            detector_mappings.append(
                {
                    "type": "dnn",
                    "param": "confidence",
                    "value": f"{conf:.2f}",
                }
            )
            detector_mappings.append(
                {
                    "type": "dnn",
                    "param": "nms_threshold",
                    "value": f"{nms:.2f}",
                }
            )

    return _templates.TemplateResponse(
        request,
        "cameras/sensitivity.html",
        {
            "request": request,
            "camera_name": name,
            "sensitivity": int(sensitivity),
            "detector_mappings": detector_mappings,
        },
    )


@router.post("/{name}/sensitivity")
async def save_camera_sensitivity(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> RedirectResponse:
    """Save camera sensitivity value (admin-only).

    Form fields:
        sensitivity: int 0-100
        action: optional "save_and_reload" to also rebuild detectors

    On success, redirects to the camera detail page (or sensitivity page
    if save_and_reload fails).
    """
    cfg = _get_cfg(request)
    cam_config = _find_camera_config(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    form = await request.form()

    # Parse sensitivity value
    sensitivity_raw = form.get("sensitivity")
    try:
        sensitivity = int(sensitivity_raw)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="sensitivity must be an integer 0-100") from exc

    if not 0 <= sensitivity <= 100:
        raise HTTPException(status_code=422, detail="sensitivity must be between 0 and 100")

    cam_config.sensitivity = float(sensitivity)

    # Persist to config.yaml
    config_path = _get_config_path(request)
    if config_path is not None:
        _persist_camera_field(config_path, cfg, "sensitivity", cam_config.sensitivity)

    # Handle "save and reload" action
    action = form.get("action")
    if action == "save_and_reload":
        _try_rebuild_detectors(request, name)

    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


@router.get("/{name}/detection-classes", response_class=HTMLResponse)
async def camera_detection_classes_page(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Render the detection class selection page for a camera (admin-only).

    Shows checkboxes for all 80 COCO classes, grouped by category.
    """
    cfg = _get_cfg(request)
    cam_config = _find_camera_config(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    active_classes = cam_config.detect_classes or []
    active_classes_set = set(active_classes)
    class_groups = _group_classes_for_template()

    return _templates.TemplateResponse(
        request,
        "cameras/detection_classes.html",
        {
            "request": request,
            "camera_name": name,
            "all_classes": COCO_CLASSES,
            "active_classes": active_classes,
            "active_classes_set": active_classes_set,
            "class_groups": class_groups,
        },
    )


@router.post("/{name}/detection-classes")
async def save_camera_detection_classes(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> RedirectResponse:
    """Save camera detection class selection (admin-only).

    Form fields:
        class_<name>=on for each selected class
        action: optional "save_and_reload" to also rebuild detectors

    On success, redirects to the camera detail page.
    """
    cfg = _get_cfg(request)
    cam_config = _find_camera_config(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    form = await request.form()

    # Build the list of selected classes from form checkboxes
    selected: list[str] = []
    for cls_name in COCO_CLASSES:
        if form.get(f"class_{cls_name}") == "on":
            selected.append(cls_name)

    cam_config.detect_classes = selected if selected else []

    # Persist to config.yaml
    config_path = _get_config_path(request)
    if config_path is not None:
        _persist_camera_field(config_path, cfg, "detect_classes", cam_config.detect_classes)

    # Handle "save and reload" action
    action = form.get("action")
    if action == "save_and_reload":
        _try_rebuild_detectors(request, name)

    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


@router.post("/{name}/detectors/{det_type}/enabled")
async def toggle_detector_enabled(
    request: Request,
    name: str,
    det_type: str,
    user: CurrentUser = Depends(require_admin),
) -> RedirectResponse:
    """Toggle a detector's enabled flag on a camera (admin-only).

    Form fields:
        enabled: "true" or "false"

    After toggling, rebuilds the detector runner so changes take effect
    immediately, then redirects to the camera detail page.
    """
    cfg = _get_cfg(request)
    cam_config = _find_camera_config(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    form = await request.form()
    enabled_raw = form.get("enabled", "false")
    enabled = str(enabled_raw).lower() in ("true", "1", "on")

    # Update all matching detector specs
    matched = False
    for spec in cam_config.detectors:
        if spec.type == det_type:
            spec.enabled = enabled
            matched = True

    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"No detector of type {det_type!r} found on camera {name!r}",
        )

    # Persist to config.yaml
    config_path = _get_config_path(request)
    if config_path is not None:
        _persist_detectors(config_path, cfg)

    # Rebuild detectors so change takes effect immediately
    _try_rebuild_detectors(request, name)

    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


def _try_rebuild_detectors(request: Request, camera_name: str) -> None:
    """Attempt to rebuild camera detectors; log warning on failure.

    Non-fatal: if the runtime is not available, the config change is
    still persisted and will take effect on next server restart.
    """
    from ...app import AppRuntime

    app_rt: AppRuntime | None = getattr(request.app.state, "runtime", None)
    if app_rt is None:
        log.warning(
            "runtime not available; detector rebuild skipped for camera %s",
            camera_name,
        )
        return

    try:
        app_rt.rebuild_camera_detectors(camera_name)
    except ValueError as exc:
        log.warning("failed to rebuild detectors for camera %s: %s", camera_name, exc)
    except Exception as exc:
        log.error(
            "failed to rebuild detectors for camera %s: %s",
            camera_name,
            exc,
            exc_info=True,
        )


def _persist_camera_retention(config_path: Path, cfg: AppConfig) -> None:
    """Serialize current AppConfig back to config.yaml on disk.

    Uses _locked_write_yaml for crash-safe, locked writes.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras_data = data.get("cameras", [])
    for i, cam_dict in enumerate(cameras_data):
        cam_name = cam_dict.get("name")
        if cam_name is None:
            continue
        # Find the matching CameraConfig object
        for cam_cfg in cfg.cameras:
            if cam_cfg.name == cam_name:
                if cam_cfg.retention is not None:
                    cameras_data[i]["retention"] = cam_cfg.retention.model_dump(exclude_none=True)
                elif "retention" in cameras_data[i]:
                    del cameras_data[i]["retention"]
                break
    data["cameras"] = cameras_data
    _locked_write_yaml(config_path, data)


def _persist_camera_field(
    config_path: Path, cfg: AppConfig, field_name: str, value: object
) -> None:
    """Persist a single camera-level field to config.yaml.

    Updates only the specified field for each camera, preserving all
    other fields in the file.

    Args:
        config_path: Path to config.yaml.
        cfg: Current AppConfig (source of truth in memory).
        field_name: Field name on CameraConfig to persist.
        value: Value to write for the field.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras_data = data.get("cameras", [])
    for cam_dict in cameras_data:
        cam_name = cam_dict.get("name")
        if cam_name is None:
            continue
        for cam_cfg in cfg.cameras:
            if cam_cfg.name == cam_name:
                cam_dict[field_name] = value
                break
    data["cameras"] = cameras_data
    _locked_write_yaml(config_path, data)


def _persist_detectors(config_path: Path, cfg: AppConfig) -> None:
    """Serialize current detector configs back to config.yaml on disk.

    Writes each camera's full detectors list (including enabled flags)
    to config.yaml using _locked_write_yaml for crash-safe writes.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras_data = data.get("cameras", [])
    for cam_dict in cameras_data:
        cam_name = cam_dict.get("name")
        if cam_name is None:
            continue
        for cam_cfg in cfg.cameras:
            if cam_cfg.name == cam_name:
                cam_dict["detectors"] = _detectors_to_list(cam_cfg.detectors)
                break
    data["cameras"] = cameras_data
    _locked_write_yaml(config_path, data)


def _detectors_to_list(detectors: list[DetectorSpec]) -> list[dict]:
    """Convert a list of DetectorSpec to YAML-serializable dicts.

    Preserves all fields including the enabled flag.
    """
    result: list[dict] = []
    for det in detectors:
        d: dict = {"type": det.type, "enabled": det.enabled}
        if det.interval_seconds != 1.0:
            d["interval_seconds"] = det.interval_seconds
        if det.config:
            d["config"] = det.config
        if det.min_area is not None:
            d["min_area"] = det.min_area
        if det.sensitivity is not None:
            d["sensitivity"] = det.sensitivity
        if det.min_confidence is not None:
            d["min_confidence"] = det.min_confidence
        if det.min_size is not None:
            d["min_size"] = det.min_size
        if det.scale_factor is not None:
            d["scale_factor"] = det.scale_factor
        if det.min_neighbors is not None:
            d["min_neighbors"] = det.min_neighbors
        if det.import_path is not None:
            d["import_path"] = det.import_path
        if det.roi is not None:
            d["roi"] = list(det.roi)
        if det.masks is not None:
            d["masks"] = [list(m) for m in det.masks]
        result.append(d)
    return result

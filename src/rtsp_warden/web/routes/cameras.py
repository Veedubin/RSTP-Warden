"""Camera route handlers for the rtsp-warden web UI.

Provides camera list, camera detail, camera status partial
for htmx auto-refresh, per-camera retention policy management,
sensitivity adjustment, detection class configuration, and
per-detector enable/disable toggles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from ...config import AppConfig, DetectorSpec, RetentionConfig
from ..auth_depends import CurrentUser, require_admin, require_user
from ..config_lock import _locked_write_yaml
from ..services.cameras import get_camera_by_name, get_camera_detectors, list_cameras
from ..services.preview import MJPEG_CONTENT_TYPE, find_hub, mjpeg_frames
from ._common import find_camera, get_cfg, get_config_path, templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")


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


@router.get("", response_class=HTMLResponse)
async def cameras_list(request: Request, user=Depends(require_user)) -> HTMLResponse:
    """Render the camera grid page."""
    cfg = get_cfg(request)
    cameras = list_cameras(cfg, request.app.state.runtime_provider())
    return templates.TemplateResponse(
        request,
        "cameras/list.html",
        {
            "request": request,
            "cameras": cameras,
        },
    )


@router.get("/{name}/snapshot.jpg")
async def camera_snapshot(request: Request, name: str, user=Depends(require_user)) -> Response:
    """Latest JPEG frame from the in-process hub."""
    hub = find_hub(request.app.state.runtime_provider(), name)
    if hub is None:
        raise HTTPException(status_code=503, detail="No live preview for this camera")
    jpeg, _fid, _ts = hub.snapshot()
    if not jpeg:
        raise HTTPException(status_code=503, detail="No frame yet")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{name}/live.mjpeg")
async def camera_live(request: Request, name: str, user=Depends(require_user)) -> StreamingResponse:
    """Same-origin MJPEG stream; replaces links to the side-server on 127.0.0.1."""
    hub = find_hub(request.app.state.runtime_provider(), name)
    if hub is None:
        raise HTTPException(status_code=503, detail="No live preview for this camera")
    return StreamingResponse(
        mjpeg_frames(hub),
        media_type=MJPEG_CONTENT_TYPE,
        headers={"Cache-Control": "no-store"},
    )


# htmx stops an "every Ns" poll when a response has this status, then swaps in the
# (empty) body, so the polling element disappears.
HTMX_STOP_POLLING = 286


def _missing_camera(request: Request, name: str) -> Response:
    """Answer a status poll or page request for a camera that is not configured.

    An htmx poll gets 286 with an empty body: htmx removes the polling card or row and
    stops polling, so a deleted camera does not poll a 404 forever. Anything else is 404.
    """
    if request.headers.get("hx-request") == "true":
        return Response(status_code=HTMX_STOP_POLLING)
    raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")


@router.get("/{name}", response_class=HTMLResponse)
async def camera_detail(
    request: Request, name: str, user: CurrentUser = Depends(require_user)
) -> HTMLResponse:
    """Render a single camera detail page.

    Admin-only controls (edit, zones, sensitivity, classes, the retention form) are
    rendered only for admins; viewers would get 403 on every one of them.
    """
    from ...retention_resolver import resolve_retention

    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
    cam = get_camera_by_name(cfg, name, request.app.state.runtime_provider())
    if cam_config is None or cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Same-origin preview URLs served by this app from the in-process FrameHub.
    mjpeg_url = ""
    snapshot_url = ""
    if cam["has_proxy"] and cam["proxy_mode"] == "mjpeg":
        mjpeg_url = f"/cameras/{name}/live.mjpeg"
        snapshot_url = f"/cameras/{name}/snapshot.jpg"

    return templates.TemplateResponse(
        request,
        "cameras/detail.html",
        {
            "request": request,
            "camera": cam,
            "is_admin": user.role == "admin",
            "mjpeg_url": mjpeg_url,
            "snapshot_url": snapshot_url,
            "effective_retention": resolve_retention(cam_config, cfg.retention),
            "has_per_camera_retention": cam_config.retention is not None,
            "global_retention": cfg.retention,
            "sensitivity": cam_config.sensitivity,
            "detect_classes": cam_config.detect_classes,
            "zone_count": len(cam_config.zones),
        },
    )


@router.get("/{name}/status", response_class=HTMLResponse)
async def camera_status(
    request: Request, name: str, user: CurrentUser = Depends(require_user)
) -> Response:
    """Return the camera card partial; every card polls this every 5 s."""
    cfg = get_cfg(request)
    cam = get_camera_by_name(cfg, name, request.app.state.runtime_provider())
    if cam is None:
        return _missing_camera(request, name)
    return templates.TemplateResponse(
        request,
        "partials/camera_card.html",
        {"request": request, "camera": cam},
    )


@router.get("/{name}/status-row", response_class=HTMLResponse)
async def camera_status_row(
    request: Request, name: str, user: CurrentUser = Depends(require_user)
) -> Response:
    """Return the detail page's status ``<tr>``; the row polls this every 5 s."""
    cfg = get_cfg(request)
    cam = get_camera_by_name(cfg, name, request.app.state.runtime_provider())
    if cam is None:
        return _missing_camera(request, name)
    return templates.TemplateResponse(
        request,
        "partials/status_row.html",
        {"request": request, "camera": cam},
    )


@router.get("/{name}/settings")
async def camera_settings_redirect(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> RedirectResponse:
    """Old read-only settings page; camera settings are now edited at /cameras/{name}/edit."""
    if find_camera(get_cfg(request), name) is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")
    return RedirectResponse(url=f"/cameras/{quote(name, safe='')}/edit", status_code=303)


@router.get("/{name}/detectors", response_class=HTMLResponse)
async def cameras_detectors_partial(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_user),
) -> HTMLResponse:
    """htmx partial -- refresh detector list for one camera.

    Returns the partial template HTML directly.
    """
    cfg = get_cfg(request)
    detectors = get_camera_detectors(cfg, name)
    has_roi = any(d["has_roi"] for d in detectors)
    has_masks = any(d["has_masks"] for d in detectors)
    num_masks = sum(1 for d in detectors if d["has_masks"])

    return templates.TemplateResponse(
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
    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    config_path = get_config_path(request)
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

    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
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
    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
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

    return templates.TemplateResponse(
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
    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
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
    config_path = get_config_path(request)
    if config_path is not None:
        _persist_camera_field(config_path, name, "sensitivity", cam_config.sensitivity)

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
    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    active_classes = cam_config.detect_classes or []
    active_classes_set = set(active_classes)
    class_groups = _group_classes_for_template()

    return templates.TemplateResponse(
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
    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
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
    config_path = get_config_path(request)
    if config_path is not None:
        _persist_camera_field(config_path, name, "detect_classes", cam_config.detect_classes)

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
    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
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
    config_path = get_config_path(request)
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
    config_path: Path, camera_name: str, field_name: str, value: object
) -> None:
    """Persist one camera-level field for one camera to config.yaml.

    Args:
        config_path: Path to config.yaml.
        camera_name: The camera whose field changed.
        field_name: Field name on CameraConfig to persist.
        value: Value to write for the field.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras_data = data.get("cameras", [])
    for cam_dict in cameras_data:
        if cam_dict.get("name") == camera_name:
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

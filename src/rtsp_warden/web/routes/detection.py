"""Detection routes for one camera (RW-3 Task 15).

Everything under ``/cameras/{name}`` that configures detection: the Detection panel
(``detect_fps``, tracking, detector list, class filter, zones link, rules and "Fire
test event"), index-keyed detector toggles, the sensitivity and detection-classes
pages, the per-camera retention override and the detector hot reload. These handlers
lived in ``web/routes/cameras.py`` before RW-3 Task 15.

htmx requests (``HX-Request: true``) get HTML fragments back; plain form posts get a
303 to the camera page, as before.

Config writes and detector rebuilds block (file IO, an ONNX session load, joining the
old runner's worker), so every async handler runs them with ``run_in_threadpool``:
nothing here may stall the event loop that also serves live previews and /healthz.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool

from ...config import DETECT_FPS_MAX, DETECT_FPS_MIN, AppConfig, CameraConfig, RetentionConfig
from ...detectors.sensitivity import (
    apply_sensitivity_to_confidence,
    apply_sensitivity_to_motion,
    apply_sensitivity_to_nms,
)
from ..auth_depends import CurrentUser, require_admin, require_user
from ..services.detection import (
    _persist_camera_field,
    _persist_camera_retention,
    _persist_detector_entry,
    camera_badge,
    class_groups_for,
    detector_rows,
    fire_test_event,
    label_choices,
    runtime_detection_status,
    write_failed_message,
)
from ..services.preview import (
    MJPEG_CONTENT_TYPE,
    box_annotator,
    boxes_max_age,
    find_hub,
    live_boxes_for,
    mjpeg_frames,
)
from ._common import find_camera, get_cfg, get_config_path, templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")


# --- helpers -----------------------------------------------------------------------------


def _is_htmx(request: Request) -> bool:
    """True for requests sent by htmx (they carry ``HX-Request: true``)."""
    return request.headers.get("HX-Request") == "true"


def _camera_or_404(cfg: AppConfig, name: str) -> CameraConfig:
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")
    return cam


def _display_runtime(request: Request) -> Any:
    """The runtime used for display (``app.state.runtime_provider()``), or None."""
    provider = getattr(request.app.state, "runtime_provider", None)
    return provider() if callable(provider) else None


def _form_float(form: Any, key: str) -> float:
    raw = str(form.get(key, "")).strip()
    try:
        value = float(raw)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"{key} must be a number") from None
    if not math.isfinite(value):
        raise HTTPException(status_code=422, detail=f"{key} must be a number")
    return value


def _form_int(form: Any, key: str) -> int:
    raw = str(form.get(key, "")).strip()
    try:
        return int(raw)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"{key} must be a whole number") from None


def _detector_list_context(
    request: Request,
    cfg: AppConfig,
    cam: CameraConfig,
    user: CurrentUser,
    message: str | None = None,
) -> dict[str, Any]:
    rows = detector_rows(cfg, _display_runtime(request), cam.name)
    return {
        "request": request,
        "camera_name": cam.name,
        "detectors": rows,
        "has_roi": any(d["has_roi"] for d in rows),
        "has_masks": any(d["has_masks"] for d in rows),
        "num_masks": sum(1 for d in rows if d["has_masks"]),
        "is_admin": user.role == "admin",
        "list_message": message,
        "detect_fps": cam.detect_fps,
    }


def _detector_list_response(
    request: Request,
    cfg: AppConfig,
    cam: CameraConfig,
    user: CurrentUser,
    message: str | None = None,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/detector_list.html",
        _detector_list_context(request, cfg, cam, user, message),
    )


def _panel_response(
    request: Request,
    cfg: AppConfig,
    cam: CameraConfig,
    user: CurrentUser,
    *,
    message: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    context = _detector_list_context(request, cfg, cam, user)
    context.update(
        {
            "cam": cam,
            "status": runtime_detection_status(_display_runtime(request), cam.name),
            "detection_badge": camera_badge(_display_runtime(request), cam.name),
            "detect_fps_min": DETECT_FPS_MIN,
            "detect_fps_max": DETECT_FPS_MAX,
            "zone_count": len(cam.zones),
            "message": message,
            "error": error,
        }
    )
    return templates.TemplateResponse(request, "partials/detection_panel.html", context)


def _try_rebuild_detectors(request: Request, camera_name: str) -> bool:
    """Attempt to rebuild camera detectors; log warning on failure.

    Non-fatal: if the runtime is not available, the config change is
    still persisted and will take effect on next server restart.
    Returns True when the rebuild ran without raising.
    """
    from ...app import AppRuntime

    app_rt: AppRuntime | None = getattr(request.app.state, "runtime", None)
    if app_rt is None:
        log.warning(
            "runtime not available; detector rebuild skipped for camera %s",
            camera_name,
        )
        return False

    try:
        app_rt.rebuild_camera_detectors(camera_name)
    except ValueError as exc:
        log.warning("failed to rebuild detectors for camera %s: %s", camera_name, exc)
        return False
    except Exception as exc:
        log.error(
            "failed to rebuild detectors for camera %s: %s",
            camera_name,
            exc,
            exc_info=True,
        )
        return False
    return True


# --- Detection panel ---------------------------------------------------------------------


@router.get("/{name}/detection", response_class=HTMLResponse)
async def detection_panel(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_user),
) -> HTMLResponse:
    """htmx partial: the Detection panel body for one camera."""
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    return _panel_response(request, cfg, cam, user)


@router.post("/{name}/detection", response_model=None)
async def save_detection_settings(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Save ``detect_fps``, ``track_grace_seconds`` and ``min_track_frames`` (admin-only).

    Only changed fields are written to config.yaml, then the camera's detectors are
    rebuilt once. A new ``detect_fps`` changes the tap rate, an ffmpeg argument, so the
    rebuild restarts that camera's ingest.
    """
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    form = await request.form()

    detect_fps = _form_float(form, "detect_fps")
    grace = _form_float(form, "track_grace_seconds")
    min_frames = _form_int(form, "min_track_frames")
    if not DETECT_FPS_MIN <= detect_fps <= DETECT_FPS_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"detect_fps must be between {DETECT_FPS_MIN:g} and {DETECT_FPS_MAX:g}",
        )
    if grace <= 0:
        raise HTTPException(status_code=422, detail="track_grace_seconds must be > 0")
    if min_frames < 1:
        raise HTTPException(status_code=422, detail="min_track_frames must be >= 1")
    for index, spec in enumerate(cam.detectors):
        explicit = spec.fps is not None and not getattr(spec, "_fps_from_interval", False)
        if explicit and spec.fps > detect_fps:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"detectors[{index}] ({spec.type}) has fps {spec.fps:g}, above the new "
                    f"detect_fps {detect_fps:g}; lower that detector's fps in config.yaml first"
                ),
            )

    changes: dict[str, float | int] = {}
    if detect_fps != cam.detect_fps:
        changes["detect_fps"] = detect_fps
    if grace != cam.track_grace_seconds:
        changes["track_grace_seconds"] = grace
    if min_frames != cam.min_track_frames:
        changes["min_track_frames"] = min_frames

    message: str | None = None
    error: str | None = None
    if not changes:
        message = "No changes to save."
    else:
        config_path = get_config_path(request)
        if config_path is not None:

            def persist(path: Path = config_path) -> None:
                for key, value in changes.items():
                    _persist_camera_field(path, name, key, value)

            try:
                await run_in_threadpool(persist)
            except OSError as exc:
                log.warning("camera %s: detection settings not saved: %s", name, exc)
                error = write_failed_message(config_path, exc)
        for key, value in changes.items():
            setattr(cam, key, value)
        # One rebuild covers every field: it rebuilds the tracker and, when the tap the
        # camera needs changed (detect_fps), asks the supervisor to restart the ingest
        # itself (R18, Task 12). Calling request_restart_camera here would restart twice.
        if not await run_in_threadpool(_try_rebuild_detectors, request, name):
            message = "Saved. The change applies after the next restart."
        elif "detect_fps" in changes:
            message = f"Saved. This camera's ingest restarts to use {detect_fps:g} fps."
        else:
            message = "Saved. Detectors reloaded."

    if _is_htmx(request):
        return _panel_response(request, cfg, cam, user, message=message, error=error)
    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


@router.get("/{name}/detectors", response_class=HTMLResponse)
async def cameras_detectors_partial(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_user),
) -> HTMLResponse:
    """htmx partial: the detector table of one camera (polled every 10 s by the panel)."""
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    return _detector_list_response(request, cfg, cam, user)


@router.post("/{name}/detectors/{index}/enabled", response_model=None)
async def toggle_detector_enabled(
    request: Request,
    name: str,
    index: int,
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Enable or disable one detector, addressed by its index in config.yaml (admin-only).

    Form fields:
        enabled: "true" or "false"

    Patches only ``cameras[name].detectors[index].enabled`` in config.yaml (409 when
    that entry is gone or has another type, i.e. the file changed since it was
    loaded), updates the in-memory spec, then rebuilds the camera's detectors.
    """
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    if not 0 <= index < len(cam.detectors):
        raise HTTPException(status_code=404, detail=f"Camera {name!r} has no detector #{index}")
    spec = cam.detectors[index]

    form = await request.form()
    enabled = str(form.get("enabled", "false")).lower() in ("true", "1", "on")

    message: str | None = None
    config_path = get_config_path(request)
    if config_path is not None:
        try:
            written = await run_in_threadpool(
                _persist_detector_entry,
                config_path,
                name,
                index,
                {"enabled": enabled},
                expected_type=spec.type,
            )
        except OSError as exc:
            log.warning("camera %s: detector #%d toggle not saved: %s", name, index, exc)
            message = write_failed_message(config_path, exc)
        else:
            if not written:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"config.yaml no longer has a {spec.type} detector at index {index} "
                        f"for camera {name!r}; restart rtsp-warden to load the edited file"
                    ),
                )

    spec.enabled = enabled
    await run_in_threadpool(_try_rebuild_detectors, request, name)

    if _is_htmx(request):
        return _detector_list_response(request, cfg, cam, user, message)
    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


@router.post("/{name}/detectors/{index}/fps", response_model=None)
async def set_detector_fps(
    request: Request,
    name: str,
    index: int,
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Set one detector's own rate, addressed by its index in config.yaml (admin-only).

    Form fields:
        fps: a number above 0 and at most the camera's ``detect_fps``; empty means
            "run at the camera's detect_fps" (the key is written as null)

    Patches only ``cameras[name].detectors[index].fps`` in config.yaml (409 when that
    entry is gone or has another type), updates the in-memory spec, then rebuilds the
    camera's detectors. A per-detector rate is a hot reload (spec 6): the frame tap
    keeps the camera's detect_fps, so the ingest is not restarted.
    """
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    if not 0 <= index < len(cam.detectors):
        raise HTTPException(status_code=404, detail=f"Camera {name!r} has no detector #{index}")
    spec = cam.detectors[index]

    form = await request.form()
    fps: float | None = None
    if str(form.get("fps", "")).strip():
        fps = _form_float(form, "fps")
        if not 0 < fps <= cam.detect_fps:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"fps must be above 0 and at most the camera's detect_fps {cam.detect_fps:g}"
                ),
            )

    message: str | None = None
    config_path = get_config_path(request)
    if config_path is not None:
        try:
            written = await run_in_threadpool(
                _persist_detector_entry,
                config_path,
                name,
                index,
                {"fps": fps},
                expected_type=spec.type,
            )
        except OSError as exc:
            log.warning("camera %s: detector #%d fps not saved: %s", name, index, exc)
            message = write_failed_message(config_path, exc)
        else:
            if not written:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"config.yaml no longer has a {spec.type} detector at index {index} "
                        f"for camera {name!r}; restart rtsp-warden to load the edited file"
                    ),
                )

    spec.fps = fps
    spec._fps_from_interval = False  # an explicit value now, never clamped as converted
    await run_in_threadpool(_try_rebuild_detectors, request, name)

    if _is_htmx(request):
        return _detector_list_response(request, cfg, cam, user, message)
    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


@router.post("/{name}/rules/test", response_class=HTMLResponse)
def fire_rules_test_event(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Fire a synthetic ``person`` test event through the camera's rules (admin-only).

    Uses ``AppRuntime.dispatch_event`` (real rules, cooldown bypassed, real action
    queue, no clip). A plain ``def``: FastAPI runs it in its threadpool, so the
    database insert and the thumbnail write stay off the event loop.
    """
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    runtime = getattr(request.app.state, "runtime", None)
    result = fire_test_event(cam, dispatch=getattr(runtime, "dispatch_event", None))
    return templates.TemplateResponse(
        request,
        "partials/rules_test_result.html",
        {"request": request, "camera_name": name, "result": result, "rule_count": len(cam.rules)},
    )


# --- retention and reload (moved from cameras.py) ---------------------------------------


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

    On success, redirects to the camera detail page. A config.yaml that cannot be
    written answers 503 with the reason; the override stays active in memory.
    """
    cfg = get_cfg(request)
    cam_config = _camera_or_404(cfg, name)

    config_path = get_config_path(request)
    form = await request.form()

    # Handle "reset to global" action
    action = form.get("action")
    if action == "reset":
        cam_config.retention = None
    else:
        # Parse retention fields from form
        max_days_raw = form.get("max_days")
        max_gb_raw = form.get("max_gb")
        keep_last_n_raw = form.get("keep_last_n")
        cleanup_interval_raw = form.get("cleanup_interval_seconds")

        max_days: int | None = (
            int(max_days_raw) if max_days_raw and str(max_days_raw).strip() else None
        )
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
        try:
            await run_in_threadpool(_persist_camera_retention, config_path, cfg)
        except OSError as exc:
            raise HTTPException(
                status_code=503, detail=write_failed_message(config_path, exc)
            ) from exc

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
    _camera_or_404(cfg, name)

    try:
        await run_in_threadpool(app_rt.rebuild_camera_detectors, name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.error("failed to reload detectors for camera %s: %s", name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload detectors: {exc}") from exc

    return JSONResponse({"ok": True, "camera": name})


# --- sensitivity (moved from cameras.py) ------------------------------------------------


def _sensitivity_mappings(cam: CameraConfig) -> list[dict[str, str]]:
    """Per-detector preview of what the camera sensitivity sets."""
    sensitivity = cam.sensitivity
    mappings: list[dict[str, str]] = []
    for spec in cam.detectors:
        if spec.type == "motion":
            var_th = apply_sensitivity_to_motion(sensitivity)
            mappings.append({"type": "motion", "param": "varThreshold", "value": str(var_th)})
        elif spec.type in ("person", "vehicle"):
            conf = apply_sensitivity_to_confidence(sensitivity)
            mappings.append({"type": spec.type, "param": "min_confidence", "value": f"{conf:.2f}"})
        elif spec.type == "dnn":
            conf = apply_sensitivity_to_confidence(sensitivity)
            nms = apply_sensitivity_to_nms(sensitivity)
            mappings.append({"type": "dnn", "param": "confidence", "value": f"{conf:.2f}"})
            mappings.append({"type": "dnn", "param": "nms_threshold", "value": f"{nms:.2f}"})
        elif spec.type == "onnx":
            if spec.min_confidence is not None:
                value = f"{spec.min_confidence:.2f} (set in config.yaml)"
            else:
                value = f"{apply_sensitivity_to_confidence(sensitivity):.2f}"
            mappings.append({"type": "onnx", "param": "min_confidence", "value": value})
    return mappings


def _sensitivity_page(
    request: Request, cam: CameraConfig, *, error: str | None = None
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "cameras/sensitivity.html",
        {
            "request": request,
            "camera_name": cam.name,
            "sensitivity": int(cam.sensitivity),
            "detector_mappings": _sensitivity_mappings(cam),
            "error": error,
        },
    )


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
    cam = _camera_or_404(cfg, name)
    return _sensitivity_page(request, cam)


@router.post("/{name}/sensitivity", response_model=None)
async def save_camera_sensitivity(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Save camera sensitivity value (admin-only).

    Form fields:
        sensitivity: int 0-100
        action: optional "save_and_reload" to also rebuild detectors

    On success, redirects to the camera detail page. When config.yaml cannot be
    written, the page is shown again with the reason (the value stays in memory).
    """
    cfg = get_cfg(request)
    cam_config = _camera_or_404(cfg, name)

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
    error: str | None = None
    config_path = get_config_path(request)
    if config_path is not None:
        try:
            await run_in_threadpool(
                _persist_camera_field, config_path, name, "sensitivity", cam_config.sensitivity
            )
        except OSError as exc:
            error = write_failed_message(config_path, exc)

    # Handle "save and reload" action
    if form.get("action") == "save_and_reload":
        await run_in_threadpool(_try_rebuild_detectors, request, name)

    if error is not None:
        return _sensitivity_page(request, cam_config, error=error)
    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


# --- detection classes (moved from cameras.py) -------------------------------------------


def _classes_page(
    request: Request, cfg: AppConfig, cam: CameraConfig, *, error: str | None = None
) -> HTMLResponse:
    labels = label_choices(cam, cfg.runtime.models_dir)
    all_mode = cam.detect_classes is None
    active = list(labels) if all_mode else list(cam.detect_classes or [])
    return templates.TemplateResponse(
        request,
        "cameras/detection_classes.html",
        {
            "request": request,
            "camera_name": cam.name,
            "all_classes": labels,
            "active_classes": active,
            "active_classes_set": set(active),
            "class_groups": class_groups_for(labels),
            "classes_mode": "all" if all_mode else "custom",
            "has_onnx": any(spec.type == "onnx" for spec in cam.detectors),
            "error": error,
        },
    )


@router.get("/{name}/detection-classes", response_class=HTMLResponse)
async def camera_detection_classes_page(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Render the detection class selection page for a camera (admin-only).

    The checkboxes are the labels of the camera's ``onnx`` models (the default
    model's COCO labels when it has none), grouped by category. "All labels"
    (``detect_classes: null``) is a radio choice of its own.
    """
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    return _classes_page(request, cfg, cam)


@router.post("/{name}/detection-classes", response_model=None)
async def save_camera_detection_classes(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Save camera detection class selection (admin-only).

    Form fields:
        classes_mode: "all" stores ``detect_classes: null`` (every label); anything
            else, or no field, stores the ticked labels (possibly ``[]``)
        class_<label>=on for each selected label; names the model does not have
            are ignored
        action: optional "save_and_reload" to also rebuild detectors

    On success, redirects to the camera detail page.
    """
    cfg = get_cfg(request)
    cam_config = _camera_or_404(cfg, name)

    form = await request.form()
    selected: list[str] | None
    if form.get("classes_mode") == "all":
        selected = None
    else:
        labels = label_choices(cam_config, cfg.runtime.models_dir)
        selected = [label for label in labels if form.get(f"class_{label}") == "on"]

    cam_config.detect_classes = selected

    # Persist to config.yaml
    error: str | None = None
    config_path = get_config_path(request)
    if config_path is not None:
        try:
            await run_in_threadpool(
                _persist_camera_field, config_path, name, "detect_classes", selected
            )
        except OSError as exc:
            error = write_failed_message(config_path, exc)

    # Handle "save and reload" action
    if form.get("action") == "save_and_reload":
        await run_in_threadpool(_try_rebuild_detectors, request, name)

    if error is not None:
        return _classes_page(request, cfg, cam_config, error=error)
    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


@router.get("/{name}/live-boxes.mjpeg")
async def camera_live_boxes(
    request: Request, name: str, user=Depends(require_user)
) -> StreamingResponse:
    """Same-origin MJPEG stream with the tracker's current boxes drawn on each frame.

    The plain stream stays at /cameras/{name}/live.mjpeg (web/routes/cameras.py). The runner
    is looked up again for every frame because detector rebuilds replace it; frames pass
    through unannotated while nothing fresh is tracked. Works without a loaded config, like
    live.mjpeg; then the stale limit is the default.
    """
    provider = request.app.state.runtime_provider
    hub = find_hub(provider(), name)
    if hub is None:
        raise HTTPException(status_code=503, detail="No live preview for this camera")
    cfg = getattr(request.app.state, "cfg", None)
    cam = find_camera(cfg, name) if cfg is not None else None
    annotate = box_annotator(lambda: live_boxes_for(provider(), name), max_age_s=boxes_max_age(cam))
    return StreamingResponse(
        mjpeg_frames(hub, annotate=annotate),
        media_type=MJPEG_CONTENT_TYPE,
        headers={"Cache-Control": "no-store"},
    )

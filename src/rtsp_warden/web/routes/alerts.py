"""Alert notifier route handlers for the rtsp-warden web UI.

Provides admin-only views for listing notifiers, sending test
notifications, and managing alert configuration.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates

from ...config import AppConfig
from ..auth_depends import CurrentUser, require_admin
from ..paths import TEMPLATES_DIR

router = APIRouter(prefix="/alerts")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _get_cfg(request: Request) -> AppConfig:
    """Get AppConfig from app state, raising 500 if missing."""
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="AppConfig not available")
    return cfg


def _get_alert_manager(request: Request):
    """Get AlertManager from app state, or None if not set."""
    return getattr(request.app.state, "alert_manager", None)


@router.get("", response_class=HTMLResponse)
async def alerts_list(request: Request, user: CurrentUser = Depends(require_admin)) -> HTMLResponse:
    """Render the alerts list page (admin-only)."""
    cfg = _get_cfg(request)
    notifiers = cfg.alerts.notifiers

    return _templates.TemplateResponse(
        request,
        "alerts/list.html",
        {
            "request": request,
            "notifiers": notifiers,
            "cfg": cfg,
        },
    )


@router.get("/{name}/test", response_class=JSONResponse)
async def test_notifier(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> JSONResponse:
    """Send a test notification to the named notifier (htmx-friendly JSON)."""
    manager = _get_alert_manager(request)
    if manager is None:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "AlertManager not initialized"},
        )

    try:
        result = await manager.test_notifier(name)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=f"No notifier named {name!r}") from err
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )

    return JSONResponse(
        content={
            "success": result.success,
            "notifier_name": result.notifier_name,
            "error": result.error,
            "http_status": result.http_status,
        },
    )


@router.get("/new", response_class=HTMLResponse)
async def new_notifier_form(
    request: Request, user: CurrentUser = Depends(require_admin)
) -> HTMLResponse:
    """Render the new notifier form."""
    return _templates.TemplateResponse(
        request,
        "alerts/new.html",
        {
            "request": request,
            "error": None,
        },
    )


@router.get("/{name}/edit", response_class=HTMLResponse)
async def edit_notifier_form(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Render the edit notifier form."""
    cfg = _get_cfg(request)
    notifier = None
    for n in cfg.alerts.notifiers:
        if n.name == name:
            notifier = n
            break

    if notifier is None:
        raise HTTPException(status_code=404, detail=f"No notifier named {name!r}")

    return _templates.TemplateResponse(
        request,
        "alerts/edit.html",
        {
            "request": request,
            "notifier": notifier,
            "error": None,
        },
    )

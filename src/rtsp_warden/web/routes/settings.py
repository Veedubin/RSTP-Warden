"""System settings route handler for the rtsp-warden web UI.

Displays read-only system information: version, ports, auth status,
database type, camera/user counts, environment variables, and disk space.
Admin-only access.
"""

from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from ... import __version__
from ...db import list_users
from ..auth_depends import CurrentUser, require_admin
from ..paths import TEMPLATES_DIR

router = APIRouter(prefix="/settings")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("", response_class=HTMLResponse)
async def settings_page(
    request: Request, user: CurrentUser = Depends(require_admin)
) -> HTMLResponse:
    """Render the system settings / info page (admin-only)."""
    cfg = getattr(request.app.state, "cfg", None)

    # Count cameras and users
    camera_count = len(cfg.cameras) if cfg else 0
    try:
        user_count = len(list_users())
    except Exception:
        user_count = 0

    # Read relevant env vars
    env_vars = {
        "WARDEN_AUTH_ENABLED": os.getenv("WARDEN_AUTH_ENABLED", "(not set)"),
        "WARDEN_WEB_HOST": os.getenv("WARDEN_WEB_HOST", "(not set)"),
        "WARDEN_WEB_PORT": os.getenv("WARDEN_WEB_PORT", "(not set)"),
        "WARDEN_DB_URL": os.getenv("WARDEN_DB_URL", "(not set)"),
        "WARDEN_HTTPS": os.getenv("WARDEN_HTTPS", "(not set)"),
    }

    # Disk space at recordings dir
    disk_info = ""
    if cfg:
        try:
            rec_dir = cfg.cameras[0].record.output_dir if cfg.cameras else None
            if rec_dir:
                usage = shutil.disk_usage(str(rec_dir))
                disk_info = f"{usage.free // (1024**3)} GB free of {usage.total // (1024**3)} GB"
        except Exception:
            disk_info = "Unable to determine"

    # Database type from DB URL
    db_url = os.getenv("WARDEN_DB_URL", "")
    db_type = "PostgreSQL" if "postgresql" in db_url else "SQLite" if db_url else "SQLite (default)"

    # Auth status
    auth_enabled = os.getenv("WARDEN_AUTH_ENABLED", "true").lower() not in ("false", "0", "no")

    # Web settings
    web_host = os.getenv("WARDEN_WEB_HOST", "0.0.0.0")
    web_port = os.getenv("WARDEN_WEB_PORT", "8080")

    return _templates.TemplateResponse(
        request,
        "settings/form.html",
        {
            "request": request,
            "version": __version__,
            "camera_count": camera_count,
            "user_count": user_count,
            "auth_enabled": auth_enabled,
            "db_type": db_type,
            "web_host": web_host,
            "web_port": web_port,
            "env_vars": env_vars,
            "disk_info": disk_info,
        },
    )

"""User management route handlers for the rtsp-warden web UI.

Provides admin-only CRUD for users: list, create, reset password,
delete, and toggle admin status.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from ... import auth
from ...db import (
    create_user,
    delete_user,
    get_user_by_id,
    get_user_by_username,
    list_users,
    set_user_admin,
    update_user_password,
)
from ..auth_depends import CurrentUser, require_admin
from ..csrf import check_csrf_form
from ..paths import TEMPLATES_DIR

router = APIRouter(prefix="/users")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Valid username: 3-32 chars, alphanumeric + underscore
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")
_MIN_PASSWORD_LEN = 8


@router.get("", response_class=HTMLResponse)
async def users_list(request: Request, user: CurrentUser = Depends(require_admin)) -> HTMLResponse:
    """Render the user list page (admin-only)."""
    all_users = list_users()
    return _templates.TemplateResponse(
        request,
        "users/list.html",
        {
            "request": request,
            "users": all_users,
            "current_user_id": user.user_id,
        },
    )


@router.get("/new", response_class=HTMLResponse)
async def new_user_form(
    request: Request, user: CurrentUser = Depends(require_admin)
) -> HTMLResponse:
    """Render the new user creation form."""
    return _templates.TemplateResponse(
        request,
        "users/new.html",
        {
            "request": request,
            "error": None,
            "username": "",
            "is_admin": False,
            "generated_password": None,
        },
    )


@router.post("/new", response_class=HTMLResponse)
async def create_new_user(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
    is_admin: str = Form(""),
    csrf_token: str = Form(""),
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Process the new user form submission."""
    if not check_csrf_form(request, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    errors: list[str] = []

    # Validate username
    username = username.strip()
    if not _USERNAME_RE.match(username):
        errors.append(
            "Username must be 3-32 characters, using only letters, digits, and underscores."
        )

    # Check for duplicate username
    if username and get_user_by_username(username) is not None:
        errors.append(f"Username {username!r} already exists.")

    # Validate / generate password
    generated_password: str | None = None
    if password.strip():
        if len(password) < _MIN_PASSWORD_LEN:
            errors.append(f"Password must be at least {_MIN_PASSWORD_LEN} characters.")
    else:
        # Auto-generate a 16-char alphanumeric password
        generated_password = auth.generate_admin_password()
        password = generated_password

    if errors:
        return _templates.TemplateResponse(
            request,
            "users/new.html",
            {
                "request": request,
                "error": " ".join(errors),
                "username": username,
                "is_admin": is_admin == "on",
                "generated_password": None,
            },
        )

    # Hash and create user
    pw_hash = auth.hash_password(password)
    admin_flag = is_admin == "on"
    new = create_user(username=username, password_hash=pw_hash, is_admin=admin_flag)

    return _templates.TemplateResponse(
        request,
        "users/new.html",
        {
            "request": request,
            "error": None,
            "username": new.username,
            "is_admin": admin_flag,
            "generated_password": generated_password or password,
        },
    )


@router.get("/{user_id}/reset-password", response_class=HTMLResponse)
async def reset_password_form(
    request: Request,
    user_id: int,
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Render the reset password confirmation form."""
    target = get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    return _templates.TemplateResponse(
        request,
        "users/reset_password.html",
        {
            "request": request,
            "target_user": target,
            "new_password": None,
        },
    )


@router.post("/{user_id}/reset-password", response_class=HTMLResponse)
async def reset_password_submit(
    request: Request,
    user_id: int,
    csrf_token: str = Form(""),
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Generate a new password for the target user and display it once."""
    if not check_csrf_form(request, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    target = get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    new_password = auth.generate_admin_password()
    pw_hash = auth.hash_password(new_password)
    update_user_password(user_id, pw_hash)

    # Re-fetch to get updated data
    target = get_user_by_id(user_id)

    return _templates.TemplateResponse(
        request,
        "users/reset_password.html",
        {
            "request": request,
            "target_user": target,
            "new_password": new_password,
        },
    )


@router.post("/{user_id}/delete")
async def delete_user_route(
    request: Request,
    user_id: int,
    csrf_token: str = Form(""),
    user: CurrentUser = Depends(require_admin),
) -> RedirectResponse:
    """Delete a user. Refuses to delete the logged-in admin."""
    if not check_csrf_form(request, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    # Refuse to delete self
    if user_id == user.user_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")

    success = delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return RedirectResponse(url="/users", status_code=303)


@router.post("/{user_id}/toggle-admin")
async def toggle_admin_route(
    request: Request,
    user_id: int,
    csrf_token: str = Form(""),
    user: CurrentUser = Depends(require_admin),
) -> RedirectResponse:
    """Toggle a user's admin status. Refuses to demote self."""
    if not check_csrf_form(request, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    # Refuse to demote self
    if user_id == user.user_id:
        raise HTTPException(status_code=400, detail="You cannot change your own admin status.")

    target = get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    new_admin = target.role != "admin"
    set_user_admin(user_id, new_admin)

    return RedirectResponse(url="/users", status_code=303)

"""User management route handlers for the rtsp-warden web UI.

Provides admin-only CRUD for users: list, create, reset password,
delete, and toggle admin status.

The create and reset-password pages are plain HTML forms because their
response is a whole page. Toggle-admin and delete are htmx forms on the users
list: an htmx request gets ``partials/users_table.html`` back (200, errors
shown inside it), which the page swaps over ``#users-table``. Without htmx
they redirect to ``/users`` (303) with a flash message, or fail with 400/404.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

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
from ._common import is_htmx, set_flash, templates

router = APIRouter(prefix="/users")

# Valid username: 3-32 chars, alphanumeric + underscore
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")
_MIN_PASSWORD_LEN = 8


def _users_table(
    request: Request,
    user: CurrentUser,
    *,
    notice: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Render the users table partial that htmx swaps over ``#users-table``."""
    return templates.TemplateResponse(
        request,
        "partials/users_table.html",
        {
            "request": request,
            "users": list_users(),
            "current_user_id": user.user_id,
            "notice": notice,
            "error": error,
        },
    )


@router.get("", response_class=HTMLResponse)
async def users_list(request: Request, user: CurrentUser = Depends(require_admin)) -> HTMLResponse:
    """Render the user list page (admin-only)."""
    all_users = list_users()
    return templates.TemplateResponse(
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
    return templates.TemplateResponse(
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
        return templates.TemplateResponse(
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

    return templates.TemplateResponse(
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

    return templates.TemplateResponse(
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

    return templates.TemplateResponse(
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
) -> Response:
    """Delete a user. Refuses to delete the logged-in admin."""
    if not check_csrf_form(request, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    htmx = is_htmx(request)

    # Refuse to delete self
    if user_id == user.user_id:
        message = "You cannot delete your own account."
        if htmx:
            return _users_table(request, user, error=message)
        raise HTTPException(status_code=400, detail=message)

    target = get_user_by_id(user_id)
    if target is None or not delete_user(user_id):
        if htmx:
            return _users_table(request, user, error="User not found.")
        raise HTTPException(status_code=404, detail="User not found")

    message = f"User {target.username} deleted."
    if htmx:
        return _users_table(request, user, notice=message)
    response = RedirectResponse(url="/users", status_code=303)
    set_flash(response, message, "success")
    return response


@router.post("/{user_id}/toggle-admin")
async def toggle_admin_route(
    request: Request,
    user_id: int,
    csrf_token: str = Form(""),
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Toggle a user's admin status. Refuses to demote self."""
    if not check_csrf_form(request, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    htmx = is_htmx(request)

    # Refuse to demote self
    if user_id == user.user_id:
        message = "You cannot change your own admin status."
        if htmx:
            return _users_table(request, user, error=message)
        raise HTTPException(status_code=400, detail=message)

    target = get_user_by_id(user_id)
    if target is None:
        if htmx:
            return _users_table(request, user, error="User not found.")
        raise HTTPException(status_code=404, detail="User not found")

    new_admin = target.role != "admin"
    set_user_admin(user_id, new_admin)

    if new_admin:
        message = f"{target.username} is now an admin."
    else:
        message = f"{target.username} is no longer an admin."
    if htmx:
        return _users_table(request, user, notice=message)
    response = RedirectResponse(url="/users", status_code=303)
    set_flash(response, message, "success")
    return response

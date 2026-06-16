"""FastAPI dependency injection helpers for authentication.

Provides ``require_user`` and ``require_admin`` as ``Depends()`` targets
for route-level access control.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from ..auth import CurrentUser
from .auth_bridge import get_current_user_from_request


async def get_current_user(request: Request) -> CurrentUser | None:
    """Resolve the current user from the request, or return None."""
    return get_current_user_from_request(request)


async def require_user(
    user: CurrentUser | None = Depends(get_current_user),
) -> CurrentUser:
    """Dependency that requires an authenticated user.

    Raises 401 if no valid session or bearer token is present.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": 'Bearer realm="warden"'},
        )
    return user


async def require_admin(user: CurrentUser = Depends(require_user)) -> CurrentUser:
    """Dependency that requires an authenticated admin user.

    Raises 403 if the user is authenticated but not an admin.
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user

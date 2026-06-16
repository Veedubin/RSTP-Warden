"""Security middleware installer for the FastAPI app.

Adds CSRF and context middleware in the correct order so that
CSRF validation happens before user resolution on each request.
"""

from __future__ import annotations

from fastapi import FastAPI

from .context import ContextMiddleware
from .csrf import CSRFMiddleware


def install_security(app: FastAPI) -> None:
    """Add CSRF and context middleware to the app.

    Middleware order in Starlette: the last ``add_middleware`` call wraps
    the outermost layer and processes the request *first*. We add
    ContextMiddleware first (it becomes the inner layer) so that
    CSRFMiddleware processes the request before context resolution.

    Request flow:  CSRFMiddleware -> ContextMiddleware -> route handler
    Response flow: route handler -> ContextMiddleware -> CSRFMiddleware
    """
    # ContextMiddleware is added first → becomes inner layer
    # (processes request after CSRF, processes response before CSRF)
    app.add_middleware(ContextMiddleware)
    # CSRFMiddleware is added second → becomes outer layer
    # (processes request first, processes response last)
    app.add_middleware(CSRFMiddleware)

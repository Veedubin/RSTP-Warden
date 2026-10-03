"""Actions page: list the configured actions and test one.

Actions are defined in ``config.yaml`` under ``actions:``; this page never edits them.
``GET /actions`` lists each action with its type, the rules that use it, and its run
history from ``action_runs`` (last run, last status, failure count).

``POST /actions/{name}/test`` builds the real action class from its spec and calls
``Action.test()``. Both handlers are plain ``def`` so FastAPI runs them in its
threadpool: a slow notification endpoint (up to the action's 10 s timeout) never
blocks the event loop that serves the live previews. A test writes no ``action_runs``
row. Nothing here renders a URL, topic, token or header, and an exception is reported
by its class name only, because ``str(exc)`` from httpx contains the request URL.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from ...actions.base import ActionResult
from ...actions.factory import build_action
from ...config import ActionSpec, AppConfig
from ...db.schema import action_stats, as_utc
from ..auth_depends import CurrentUser, require_admin
from ._common import get_cfg, templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/actions")


def _rules_using(cfg: AppConfig, name: str) -> list[str]:
    """Return ``"<camera>/<rule>"`` for every camera rule that names this action."""
    return [
        f"{cam.name}/{rule.name}"
        for cam in cfg.cameras
        for rule in cam.rules
        if name in rule.actions
    ]


def action_rows(cfg: AppConfig, stats: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build one display row per configured action, in config order.

    ``stats`` is the result of ``db.schema.action_stats()``. An action with no runs
    has no last run, no last status and zero failures. A naive ``last_run`` (SQLite)
    is read as UTC and shown in the server's local time. Rows never carry a URL,
    topic, token or header: those can hold secrets.
    """
    rows: list[dict[str, Any]] = []
    for spec in cfg.actions:
        st = stats.get(spec.name, {})
        last_run = as_utc(st.get("last_run"))
        rows.append(
            {
                "name": spec.name,
                "type": spec.type,
                "used_by": _rules_using(cfg, spec.name),
                "last_run": last_run.astimezone().strftime("%Y-%m-%d %H:%M") if last_run else None,
                "last_run_iso": last_run.isoformat() if last_run else None,
                "last_status": st.get("last_status"),
                "failures": int(st.get("failures") or 0),
            }
        )
    return rows


@router.get("", response_class=HTMLResponse)
def actions_list(request: Request, user: CurrentUser = Depends(require_admin)) -> HTMLResponse:
    """Render the Actions page (admin only)."""
    cfg = get_cfg(request)
    stats_error = False
    try:
        stats = action_stats()
    except Exception as exc:  # the page still lists config.yaml when the DB read fails
        log.warning("actions page: could not read action run history (%s)", type(exc).__name__)
        stats = {}
        stats_error = True
    return templates.TemplateResponse(
        request,
        "actions/list.html",
        {"request": request, "rows": action_rows(cfg, stats), "stats_error": stats_error},
    )


def _find_spec(cfg: AppConfig, name: str) -> ActionSpec | None:
    """Return the action spec with this name, or None."""
    for spec in cfg.actions:
        if spec.name == name:
            return spec
    return None


@router.post("/{name}/test", response_class=HTMLResponse)
def test_action(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> HTMLResponse:
    """Send a test notification through one action and return the result fragment."""
    cfg = get_cfg(request)
    spec = _find_spec(cfg, name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"No action named {name!r}")
    try:
        result = build_action(spec).test()
    except Exception as exc:  # an action should return a failed result, but never 500 here
        log.warning("action %r: test raised %s", name, type(exc).__name__)
        result = ActionResult(ok=False, error=f"unexpected error ({type(exc).__name__})")
    return templates.TemplateResponse(
        request,
        "partials/action_test_result.html",
        {"request": request, "name": name, "result": result},
    )

"""Build action instances from the ``actions:`` config list."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

from ..config import AppriseActionSpec, NtfyActionSpec, WebhookActionSpec
from .apprise import AppriseAction
from .base import Action
from .ntfy import NtfyAction
from .webhook import WebhookAction

if TYPE_CHECKING:
    from ..config import ActionSpec, AppConfig


def build_action(
    spec: ActionSpec,
    *,
    client_factory: Callable[[], httpx.Client] = httpx.Client,
) -> Action:
    """Return the action for one spec. ``client_factory`` is used by the HTTP actions."""
    if isinstance(spec, NtfyActionSpec):
        return NtfyAction(spec, client_factory=client_factory)
    if isinstance(spec, WebhookActionSpec):
        return WebhookAction(spec, client_factory=client_factory)
    if isinstance(spec, AppriseActionSpec):
        return AppriseAction(spec)
    raise ValueError(f"unknown action type: {getattr(spec, 'type', None)!r}")


def build_actions(cfg: AppConfig) -> dict[str, Action]:
    """Map action name -> action for every entry in ``cfg.actions`` (config order)."""
    return {spec.name: build_action(spec) for spec in cfg.actions}

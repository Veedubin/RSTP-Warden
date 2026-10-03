"""Actions (ntfy, webhook, Apprise) and the rule engine that decides when they fire.

Replaces the old ``rtsp_warden.alerts`` package. Actions are synchronous.
"""

from __future__ import annotations

from .apprise import AppriseAction
from .base import (
    Action,
    ActionPayload,
    ActionResult,
    placeholder_jpeg,
    send_test,
    synthetic_payload,
)
from .factory import build_action, build_actions
from .ntfy import NtfyAction
from .rules import RuleDecision, RuleEngine, RuleMatch, in_window
from .webhook import WebhookAction

__all__ = [
    "Action",
    "ActionPayload",
    "ActionResult",
    "AppriseAction",
    "NtfyAction",
    "RuleDecision",
    "RuleEngine",
    "RuleMatch",
    "WebhookAction",
    "build_action",
    "build_actions",
    "in_window",
    "placeholder_jpeg",
    "send_test",
    "synthetic_payload",
]

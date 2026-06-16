"""Alerts package -- notifier backends and dispatch manager."""

from __future__ import annotations

from .apprise import AppriseNotifier
from .base import NotificationResult, Notifier, NotifierSendError
from .factory import build_notifier
from .manager import AlertManager
from .ntfy import NtfyNotifier
from .webhook import WebhookNotifier

__all__ = [
    "AlertManager",
    "AppriseNotifier",
    "NtfyNotifier",
    "Notifier",
    "NotifierSendError",
    "NotificationResult",
    "WebhookNotifier",
    "build_notifier",
]

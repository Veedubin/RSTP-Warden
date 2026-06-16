"""Alerts package -- notifier backends and dispatch manager."""

from __future__ import annotations

from .base import NotificationResult, Notifier, NotifierSendError
from .factory import build_notifier
from .manager import AlertManager
from .ntfy import NtfyNotifier
from .webhook import WebhookNotifier

__all__ = [
    "AlertManager",
    "NtfyNotifier",
    "Notifier",
    "NotifierSendError",
    "NotificationResult",
    "WebhookNotifier",
    "build_notifier",
]

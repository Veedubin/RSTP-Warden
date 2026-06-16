"""Factory function to build notifier instances from config specs."""

from __future__ import annotations

from ..config import NotifierSpec
from .base import Notifier
from .ntfy import NtfyNotifier
from .webhook import WebhookNotifier


def build_notifier(spec: NotifierSpec) -> Notifier:
    """Build a Notifier instance from a NotifierSpec config.

    Args:
        spec: The notifier configuration from AppConfig.alerts.notifiers.

    Returns:
        A Notifier implementation (NtfyNotifier or WebhookNotifier).

    Raises:
        ValueError: If the notifier type is unknown.
    """
    if spec.type == "ntfy":
        return NtfyNotifier(
            name=spec.name,
            url=spec.url,
            topic=spec.topic,
            token=spec.token,
            priority=spec.priority,
        )
    elif spec.type == "webhook":
        return WebhookNotifier(
            name=spec.name,
            url=spec.url,
            method=spec.method,
            headers=spec.headers,
        )
    else:
        raise ValueError(f"unknown notifier type: {spec.type}")

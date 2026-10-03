"""Apprise action: one ``Apprise.notify`` call per event, thumbnail attached."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import (
    ActionPayload,
    ActionResult,
    exception_text,
    notification_body,
    notification_title,
    send_test,
)

if TYPE_CHECKING:
    from ..config import AppriseActionSpec


class AppriseAction:
    """Deliver through every URL in ``urls`` (mailto://, tgram://, mqtt://, ...).

    ``apprise.Apprise`` is looked up at call time so tests can monkeypatch it.
    Error text never contains a URL: Apprise URLs embed credentials.
    """

    def __init__(self, spec: AppriseActionSpec) -> None:
        self.name: str = spec.name
        self.type: str = "apprise"
        self._urls = list(spec.urls)

    def send(self, payload: ActionPayload, attachment: Path | None = None) -> ActionResult:
        try:
            import apprise

            apobj = apprise.Apprise()
            for index, url in enumerate(self._urls, start=1):
                if not apobj.add(url):
                    return ActionResult(
                        ok=False, error=f"Apprise could not load URL #{index}; check its scheme"
                    )
            attach = str(attachment) if attachment is not None and attachment.is_file() else None
            delivered = apobj.notify(
                body=notification_body(payload),
                title=notification_title(payload),
                attach=attach,
            )
        except Exception as exc:
            return ActionResult(ok=False, error=exception_text(exc))
        if delivered:
            return ActionResult(ok=True)
        return ActionResult(ok=False, error="Apprise reported a failed delivery")

    def test(self) -> ActionResult:
        return send_test(self)

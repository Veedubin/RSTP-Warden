"""Tests for the alerts framework: notifiers, manager, and factory."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from rtsp_warden.alerts import AlertManager, NtfyNotifier, WebhookNotifier, build_notifier
from rtsp_warden.alerts.base import NotificationResult
from rtsp_warden.config import AlertsConfig, NtifySpec, WebhookSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


async def _send_with_mock(notifier: Any, event: dict[str, Any], captures: dict[str, Any]) -> Any:
    """Send an event using a notifier with a mock transport, capturing request details."""
    transport = _capturing_handler(captures)
    notifier._client = httpx.AsyncClient(transport=transport)
    result = await notifier.send(event)
    await notifier.close()
    return result


async def _test_with_mock(notifier: Any, captures: dict[str, Any]) -> Any:
    """Run test() using a notifier with a mock transport, capturing request details."""
    transport = _capturing_handler(captures)
    notifier._client = httpx.AsyncClient(transport=transport)
    result = await notifier.test()
    await notifier.close()
    return result


def _make_event(
    camera_name: str = "front-door",
    event_type: str = "motion",
    severity: str = "info",
    message: str = "motion detected",
) -> dict[str, Any]:
    return {
        "id": 1,
        "camera_name": camera_name,
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "metadata_json": "{}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _mock_handler(status_code: int = 200) -> httpx.MockTransport:
    """Create a MockTransport that returns the given status code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=status_code, request=request)

    return httpx.MockTransport(handler)


def _capturing_handler(captures: dict[str, Any]) -> httpx.MockTransport:
    """Create a MockTransport that captures request details and returns 200.

    httpx lowercases all header names, so captured headers are lowercase.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        captures["url"] = str(request.url)
        captures["method"] = request.method
        # Store headers with original casing via a case-insensitive view
        captures["headers"] = httpx.Headers(request.headers)
        captures["content"] = request.content.decode()
        return httpx.Response(status_code=200, request=request)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# NtfyNotifier tests
# ---------------------------------------------------------------------------


class TestNtfyNotifier:
    """Tests for NtfyNotifier."""

    def test_ntfy_builds_correct_url(self) -> None:
        """NtfyNotifier with url=https://ntfy.sh, topic=alerts -> posts to https://ntfy.sh/alerts."""
        captures: dict[str, Any] = {}
        n = NtfyNotifier(name="test", url="https://ntfy.sh", topic="alerts")
        _run(_send_with_mock(n, _make_event(), captures))
        assert captures["url"] == "https://ntfy.sh/alerts"

    def test_ntfy_sends_title_priority_tags(self) -> None:
        """POST includes correct Title, Priority, Tags headers."""
        captures: dict[str, Any] = {}
        n = NtfyNotifier(name="test", url="https://ntfy.sh", topic="alerts", priority=4)
        _run(_send_with_mock(n, _make_event(event_type="person", severity="warn"), captures))
        headers: httpx.Headers = captures["headers"]
        assert "title" in headers
        assert "warn" in headers["title"]
        assert headers["priority"] == "4"
        assert headers["tags"] == "person"
        # Body is the message text
        assert "motion detected" in captures["content"]

    def test_ntfy_sends_authorization_header(self) -> None:
        """When token is set, Authorization: Bearer <token> header is included."""
        captures: dict[str, Any] = {}
        n = NtfyNotifier(name="test", url="https://ntfy.sh", topic="alerts", token="tk_123")
        _run(_send_with_mock(n, _make_event(), captures))
        assert captures["headers"]["authorization"] == "Bearer tk_123"

    def test_ntfy_test_method(self) -> None:
        """test() sends a 'This is a test from rtsp-warden' message."""
        captures: dict[str, Any] = {}
        n = NtfyNotifier(name="test", url="https://ntfy.sh", topic="alerts")
        _run(_test_with_mock(n, captures))
        assert "This is a test from rtsp-warden" in captures["content"]
        assert "test" in captures["headers"]["tags"]


# ---------------------------------------------------------------------------
# WebhookNotifier tests
# ---------------------------------------------------------------------------


class TestWebhookNotifier:
    """Tests for WebhookNotifier."""

    def test_webhook_posts_json(self) -> None:
        """Body is JSON of event dict."""
        captures: dict[str, Any] = {}
        n = WebhookNotifier(name="test", url="https://example.com/hook")
        event = _make_event()
        _run(_send_with_mock(n, event, captures))
        assert captures["method"] == "POST"
        # JSON body should contain the event keys
        import json

        body = json.loads(captures["content"])
        assert body["camera_name"] == "front-door"
        assert body["event_type"] == "motion"

    def test_webhook_with_headers(self) -> None:
        """Custom headers are passed through."""
        captures: dict[str, Any] = {}
        n = WebhookNotifier(
            name="test",
            url="https://example.com/hook",
            headers={"X-Custom": "value1"},
        )
        _run(_send_with_mock(n, _make_event(), captures))
        assert captures["headers"]["x-custom"] == "value1"

    def test_webhook_with_put_method(self) -> None:
        """PUT method works correctly."""
        captures: dict[str, Any] = {}
        n = WebhookNotifier(name="test", url="https://example.com/hook", method="PUT")
        _run(_send_with_mock(n, _make_event(), captures))
        assert captures["method"] == "PUT"


# ---------------------------------------------------------------------------
# AlertManager tests
# ---------------------------------------------------------------------------


class TestAlertManager:
    """Tests for AlertManager dispatch logic."""

    def _make_manager(
        self,
        notifiers: list[NtifySpec | WebhookSpec],
        enabled: bool = True,
        on_send: Any = None,
    ) -> AlertManager:
        cfg = AlertsConfig(enabled=enabled, notifiers=notifiers)
        mgr = AlertManager(cfg, on_send=on_send)
        _run(mgr.start())
        # Inject mock httpx clients into all notifiers so HTTP calls succeed
        for n in mgr._notifiers:
            n._client = httpx.AsyncClient(transport=_mock_handler(200))
        return mgr

    def test_manager_filters_disabled_notifiers(self) -> None:
        """Disabled notifier is not called (returns error='disabled')."""
        spec = WebhookSpec(
            name="disabled-one",
            type="webhook",
            url="https://example.com/hook",
            enabled=False,
        )
        mgr = self._make_manager([spec])
        results = _run(mgr.dispatch_event(_make_event()))
        assert len(results) == 1
        assert results[0].notifier_name == "disabled-one"
        assert results[0].success is False
        assert "disabled" in (results[0].error or "")

    def test_manager_filters_by_severity(self) -> None:
        """Info event doesn't go to a notifier that only allows warn/error."""
        spec = WebhookSpec(
            name="strict",
            type="webhook",
            url="https://example.com/hook",
            severities=["warn", "error"],
        )
        mgr = self._make_manager([spec])
        results = _run(mgr.dispatch_event(_make_event(severity="info")))
        assert len(results) == 1
        assert results[0].success is False
        assert "severity" in (results[0].error or "").lower()

    def test_manager_debounces_recent_sends(self) -> None:
        """Same (camera, type) within min_interval_seconds -> skipped."""
        spec = WebhookSpec(
            name="debounced",
            type="webhook",
            url="https://example.com/hook",
            min_interval_seconds=30,
            severities=["info", "warn", "error"],
        )
        mgr = self._make_manager([spec])
        event = _make_event(camera_name="yard", event_type="motion", severity="warn")
        # First dispatch should succeed
        results1 = _run(mgr.dispatch_event(event))
        assert results1[0].success is True
        # Immediate second dispatch should be debounced
        results2 = _run(mgr.dispatch_event(event))
        assert results2[0].success is False
        assert "debounced" in (results2[0].error or "")

    def test_manager_dispatches_in_parallel(self) -> None:
        """2 notifiers, both called."""
        spec1 = WebhookSpec(
            name="hook1",
            type="webhook",
            url="https://example.com/hook1",
            severities=["info", "warn", "error"],
        )
        spec2 = WebhookSpec(
            name="hook2",
            type="webhook",
            url="https://example.com/hook2",
            severities=["info", "warn", "error"],
        )
        mgr = self._make_manager([spec1, spec2])
        results = _run(mgr.dispatch_event(_make_event(severity="warn")))
        assert len(results) == 2
        # Both should succeed (mock returns 200)
        names = {r.notifier_name for r in results}
        assert "hook1" in names
        assert "hook2" in names

    def test_manager_handles_notifier_exception(self) -> None:
        """One notifier raising doesn't break the others."""
        spec_ok = WebhookSpec(
            name="ok-notifier",
            type="webhook",
            url="https://example.com/hook",
            severities=["info", "warn", "error"],
        )
        spec_bad = NtifySpec(
            name="bad-notifier",
            type="ntfy",
            url="https://ntfy.sh",
            topic="test",
            severities=["info", "warn", "error"],
        )
        mgr = self._make_manager([spec_bad, spec_ok])

        # Make the "bad" notifier raise by replacing its send method
        for n in mgr._notifiers:
            if n.name == "bad-notifier":

                async def _bad_send(event: dict) -> NotificationResult:
                    raise RuntimeError("connection refused")

                n.send = _bad_send  # type: ignore[assignment]

        results = _run(mgr.dispatch_event(_make_event(severity="info")))
        assert len(results) == 2
        # The bad one should have success=False with error
        bad_result = [r for r in results if r.notifier_name == "bad-notifier"][0]
        assert bad_result.success is False
        assert "connection refused" in (bad_result.error or "")
        # The ok one should still succeed
        ok_result = [r for r in results if r.notifier_name == "ok-notifier"][0]
        assert ok_result.success is True


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    """Tests for build_notifier factory function."""

    def test_build_ntfy_notifier(self) -> None:
        """build_notifier with NtifySpec returns NtfyNotifier."""
        spec = NtifySpec(
            name="phone",
            type="ntfy",
            url="https://ntfy.sh",
            topic="alerts",
            token="tk_123",
            priority=4,
        )
        notifier = build_notifier(spec)
        assert isinstance(notifier, NtfyNotifier)
        assert notifier.name == "phone"
        assert notifier._topic == "alerts"
        assert notifier._token == "tk_123"
        assert notifier._priority == 4

    def test_build_webhook_notifier(self) -> None:
        """build_notifier with WebhookSpec returns WebhookNotifier."""
        spec = WebhookSpec(
            name="slack",
            type="webhook",
            url="https://hooks.slack.com/services/abc",
            headers={"X-Custom": "val"},
            method="POST",
        )
        notifier = build_notifier(spec)
        assert isinstance(notifier, WebhookNotifier)
        assert notifier.name == "slack"

    def test_build_unknown_type_raises(self) -> None:
        """build_notifier with unknown type raises ValueError."""
        # Create a minimal spec and bypass type checks to test factory guard
        spec = NtifySpec(name="x", type="ntfy", url="https://ntfy.sh")
        # Mutate type to test the factory guard
        object.__setattr__(spec, "type", "unknown")
        with pytest.raises(ValueError, match="unknown notifier type"):
            build_notifier(spec)

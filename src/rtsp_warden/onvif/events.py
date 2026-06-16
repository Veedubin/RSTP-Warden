"""ONVIF event subscription via PullPoint notification mechanism.

Provides OnvifEventSubscriber for creating pull-point subscriptions on
ONVIF cameras and periodically pulling event messages (motion, tampering,
analytics). Uses handcrafted SOAP envelopes over httpx -- same pattern as
onvif/ptz.py. Does NOT require zeep.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

from .discovery import OnvifError
from .ptz import OnvifClient

log = logging.getLogger(__name__)

# ONVIF Events SOAP namespaces
SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
WSA_NS = "http://schemas.xmlsoap.org/ws/2004/08/addressing"
ONVIF_EVENTS_NS = "http://www.onvif.org/ver10/events/wsdl"
ONVIF_SCHEMA_NS = "http://www.onvif.org/ver10/schema"
WSNT_NS = "http://docs.oasis-open.org/wsn/b-2"
WSNT_TOPIC_NS = "http://docs.oasis-open.org/wsn/t-1"

NS_EVENTS = {
    "soap": SOAP_NS,
    "wsa": WSA_NS,
    "tev": ONVIF_EVENTS_NS,
    "tt": ONVIF_SCHEMA_NS,
    "wsnt": WSNT_NS,
    "wstop": WSNT_TOPIC_NS,
}


class OnvifEventType(str, Enum):
    """ONVIF event type classification."""

    MOTION = "motion"
    TAMPER = "tamper"
    ANALYTIC = "analytic"
    UNKNOWN = "unknown"


# Topic-to-event-type mapping (case-insensitive substring match)
_TOPIC_MAP: list[tuple[str, OnvifEventType]] = [
    ("motion", OnvifEventType.MOTION),
    ("cellmotion", OnvifEventType.MOTION),
    ("motionalarm", OnvifEventType.MOTION),
    ("tamper", OnvifEventType.TAMPER),
    ("tampering", OnvifEventType.TAMPER),
    ("imagingalarm", OnvifEventType.TAMPER),
    ("videoloss", OnvifEventType.TAMPER),
    ("ruleengine", OnvifEventType.ANALYTIC),
    ("analytics", OnvifEventType.ANALYTIC),
    ("linecross", OnvifEventType.ANALYTIC),
    ("intrusion", OnvifEventType.ANALYTIC),
]

# Event type to severity mapping for AlertManager integration
_EVENT_SEVERITY: dict[OnvifEventType, str] = {
    OnvifEventType.MOTION: "info",
    OnvifEventType.TAMPER: "warn",
    OnvifEventType.ANALYTIC: "info",
    OnvifEventType.UNKNOWN: "info",
}


@dataclass(slots=True)
class OnvifEvent:
    """A single ONVIF event message received from a pull-point subscription.

    Attributes:
        camera_name: Name of the camera that produced the event.
        event_type: Classified event type (motion, tamper, analytic, unknown).
        timestamp: Time the event was received or reported by the camera.
        raw_topic: Full ONVIF topic expression string.
        raw_message: Parsed key-value pairs from the notification message.
    """

    camera_name: str
    event_type: OnvifEventType
    timestamp: datetime
    raw_topic: str
    raw_message: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_topic(
        cls,
        camera_name: str,
        raw_topic: str,
        timestamp: datetime | None = None,
        raw_message: dict[str, Any] | None = None,
    ) -> OnvifEvent:
        """Create an OnvifEvent by classifying the ONVIF topic string.

        Args:
            camera_name: Source camera name.
            raw_topic: ONVIF topic expression (e.g. tns1:VideoSource/MotionAlarm).
            timestamp: Event time (defaults to now UTC).
            raw_message: Parsed notification data.
        """
        topic_lower = raw_topic.lower()
        event_type = OnvifEventType.UNKNOWN
        for keyword, mapped_type in _TOPIC_MAP:
            if keyword in topic_lower:
                event_type = mapped_type
                break

        return cls(
            camera_name=camera_name,
            event_type=event_type,
            timestamp=timestamp or datetime.now(tz=timezone.utc),
            raw_topic=raw_topic,
            raw_message=raw_message or {},
        )

    def to_alert_event(self) -> dict[str, Any]:
        """Convert to an event dict compatible with AlertManager.dispatch_event.

        Returns:
            Dict with keys: camera_name, event_type, severity, message.
        """
        severity = _EVENT_SEVERITY.get(self.event_type, "info")
        return {
            "camera_name": self.camera_name,
            "event_type": self.event_type.value,
            "severity": severity,
            "message": f"ONVIF {self.event_type.value} event: {self.raw_topic}",
        }


def _soap_envelope(body_content: str, action: str) -> str:
    """Build a SOAP 1.2 envelope with WS-Addressing headers for events."""
    message_id = uuid.uuid4().urn
    return f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="{SOAP_NS}"
               xmlns:wsa="{WSA_NS}">
  <soap:Header>
    <wsa:Action>{action}</wsa:Action>
    <wsa:MessageID>{message_id}</wsa:MessageID>
  </soap:Header>
  <soap:Body>
    {body_content}
  </soap:Body>
</soap:Envelope>"""


def _local_tag(tag: str) -> str:
    """Extract the local name from an XML tag (strip namespace)."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _wsse_auth(username: str | None, password: str | None) -> httpx.DigestAuth | None:
    """Build WSSE auth headers for ONVIF (HTTP digest auth)."""
    if username is None or password is None:
        return None
    return httpx.DigestAuth(username, password)


# --- SOAP envelope builders ---


def _build_subscribe_envelope(timeout_seconds: int = 60) -> str:
    """Build CreatePullPointSubscription SOAP envelope."""

    termination = f"PT{timeout_seconds}S"
    body = (
        f'<tev:CreatePullPointSubscription xmlns:tev="{ONVIF_EVENTS_NS}">'
        f"<tev:InitialTerminationTime>{termination}</tev:InitialTerminationTime>"
        f"</tev:CreatePullPointSubscription>"
    )
    action = f"{ONVIF_EVENTS_NS}/EventPortType/CreatePullPointSubscriptionRequest"
    return _soap_envelope(body, action)


def _build_pull_messages_envelope(
    subscription_ref: str,
    timeout_seconds: int = 10,
    message_limit: int = 10,
) -> str:
    """Build PullMessages SOAP envelope."""
    body = (
        f'<tev:PullMessages xmlns:tev="{ONVIF_EVENTS_NS}">'
        f"<tev:Timeout>PT{timeout_seconds}S</tev:Timeout>"
        f"<tev:MessageLimit>{message_limit}</tev:MessageLimit>"
        f"</tev:PullMessages>"
    )
    action = f"{ONVIF_EVENTS_NS}/PullPointSubscription/PullMessagesRequest"
    return _soap_envelope(body, action)


def _build_unsubscribe_envelope() -> str:
    """Build Unsubscribe SOAP envelope."""
    body = f'<wsnt:Unsubscribe xmlns:wsnt="{WSNT_NS}"/>'
    action = f"{WSNT_NS}/UnsubscribeRequest"
    return _soap_envelope(body, action)


# --- XML response parsers ---


def _parse_subscription_ref(response_text: str) -> str | None:
    """Extract the subscription reference address from CreatePullPointSubscriptionResponse.

    Returns the subscription address string, or None on parse failure.
    """
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError:
        return None

    # Look for SubscriptionReference > Address
    for elem in root.iter():
        local = _local_tag(elem.tag)
        if local == "SubscriptionReference":
            # Find Address child
            for child in elem:
                if _local_tag(child.tag) == "Address" and child.text:
                    return child.text.strip()
        # Some cameras put the address directly
        if local == "Address" and elem.text:
            # Check if parent is SubscriptionReference
            parent_found = False
            for parent in root.iter():
                for c in parent:
                    if c is elem and _local_tag(parent.tag) == "SubscriptionReference":
                        parent_found = True
                        break
            if parent_found:
                return elem.text.strip()

    return None


def _parse_pull_messages(response_text: str) -> list[dict[str, Any]]:
    """Parse a PullMessagesResponse into a list of notification dicts.

    Each dict has keys: 'topic' (str), 'timestamp' (str|None), 'data' (dict).
    """
    notifications: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError:
        return notifications

    # Find NotificationMessage elements
    for elem in root.iter():
        local = _local_tag(elem.tag)
        if local == "NotificationMessage":
            msg: dict[str, Any] = {"topic": "", "timestamp": None, "data": {}}
            for child in elem:
                child_local = _local_tag(child.tag)
                if child_local == "Topic":
                    msg["topic"] = child.text.strip() if child.text else ""
                elif child_local == "UtcTime":
                    msg["timestamp"] = child.text.strip() if child.text else None
                elif child_local == "Message":
                    # Parse SimpleItem / Data inside Message
                    msg["data"] = _parse_message_data(child)
                elif child_local == "Source":
                    msg["data"].update(_parse_simple_items(child))
                elif child_local == "Data":
                    msg["data"].update(_parse_simple_items(child))
            notifications.append(msg)

    return notifications


def _parse_simple_items(parent: ET.Element) -> dict[str, Any]:
    """Extract name/value pairs from SimpleItem elements."""
    result: dict[str, Any] = {}
    for elem in parent.iter():
        local = _local_tag(elem.tag)
        if local == "SimpleItem":
            name = elem.get("Name", "")
            value = elem.get("Value", "")
            if name:
                result[name] = value
    return result


def _parse_message_data(parent: ET.Element) -> dict[str, Any]:
    """Parse data items from a NotificationMessage > Message element."""
    result: dict[str, Any] = {}
    for child in parent:
        child_local = _local_tag(child.tag)
        if child_local == "Source":
            result.update(_parse_simple_items(child))
        elif child_local == "Data":
            result.update(_parse_simple_items(child))
        elif child_local == "Key":
            result.update(_parse_simple_items(child))
    return result


def _check_soap_fault(response_text: str, operation: str) -> None:
    """Check a SOAP response for fault elements and raise OnvifError if found."""
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError:
        return

    for elem in root.iter():
        if _local_tag(elem.tag) == "Fault":
            reason = ""
            for child in elem.iter():
                if _local_tag(child.tag) == "Text" and child.text:
                    reason = child.text
                    break
                if _local_tag(child.tag) == "faultstring" and child.text:
                    reason = child.text
                    break
            if reason:
                raise OnvifError(f"{operation} SOAP fault: {reason}")
            raise OnvifError(f"{operation} SOAP fault (unknown reason)")


class OnvifEventSubscriber:
    """PullPoint subscription client for ONVIF event services.

    Creates a PullPoint subscription on the camera's event service,
    then periodically calls PullMessages to retrieve motion, tampering,
    and other ONVIF events. Maps ONVIF event topics to rtsp-warden
    event types and fires a callback for each received event.

    Args:
        client: OnvifClient instance for the target camera.
        camera_name: Logical camera name for event attribution.
        topics: List of ONVIF topic expressions to subscribe to.
        callback: Async callable receiving OnvifEvent for each message.
        poll_interval_seconds: Seconds between PullMessages calls.
        subscription_timeout_seconds: PullPoint subscription lifetime in seconds.
        timeout_seconds: Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        client: OnvifClient,
        camera_name: str,
        topics: list[str] | None = None,
        callback: Callable[[OnvifEvent], Awaitable[None]] | None = None,
        poll_interval_seconds: float = 10.0,
        subscription_timeout_seconds: int = 60,
        timeout_seconds: int = 10,
    ) -> None:
        self._client = client
        self._camera_name = camera_name
        self._topics = topics or []
        self._callback = callback
        self._poll_interval = poll_interval_seconds
        self._subscription_timeout = subscription_timeout_seconds
        self._timeout_seconds = timeout_seconds

        # Runtime state
        self._subscription_ref: str | None = None
        self._event_xaddr: str | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._running = False
        self._last_event_time: datetime | None = None
        self._transport: httpx.BaseTransport | None = None

    @property
    def camera_name(self) -> str:
        """Camera name this subscriber is bound to."""
        return self._camera_name

    @property
    def subscription_ref(self) -> str | None:
        """Current subscription reference address, or None if not subscribed."""
        return self._subscription_ref

    @property
    def is_running(self) -> bool:
        """Whether the polling loop is currently active."""
        return self._running

    @property
    def last_event_time(self) -> datetime | None:
        """Timestamp of the last received event, or None."""
        return self._last_event_time

    def _make_client(self, timeout: float | None = None) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with optional transport override."""
        kwargs: dict[str, Any] = {"timeout": timeout or self._timeout_seconds}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    async def start(self) -> None:
        """Create a PullPoint subscription and start the polling loop.

        Raises OnvifError if subscription fails.
        """
        await self.subscribe()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info(
            "ONVIF event subscriber started for camera %s (ref=%s)",
            self._camera_name,
            self._subscription_ref,
        )

    async def stop(self) -> None:
        """Unsubscribe and cancel the polling task."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._subscription_ref is not None:
            try:
                await self.unsubscribe()
            except Exception:
                log.debug("Error unsubscribing camera %s", self._camera_name, exc_info=True)
        log.info("ONVIF event subscriber stopped for camera %s", self._camera_name)

    async def subscribe(self) -> str:
        """Create a PullPoint subscription. Returns the subscription reference.

        Raises OnvifError on failure.
        """
        # Resolve event service endpoint from device capabilities
        if self._event_xaddr is None:
            self._event_xaddr = await self._resolve_event_xaddr()

        envelope = _build_subscribe_envelope(timeout_seconds=self._subscription_timeout)

        resp_text = await self._send_soap(
            self._event_xaddr, envelope, "CreatePullPointSubscription"
        )
        _check_soap_fault(resp_text, "CreatePullPointSubscription")

        ref = _parse_subscription_ref(resp_text)
        if ref is None:
            raise OnvifError("No subscription reference in CreatePullPointSubscription response")

        self._subscription_ref = ref
        log.info(
            "Created PullPoint subscription for %s: %s",
            self._camera_name,
            ref,
        )
        return ref

    async def pull_messages(self) -> list[OnvifEvent]:
        """Pull pending messages from the current subscription.

        Returns a list of OnvifEvent objects. Returns empty list if not
        subscribed or on parse failure.
        """
        if self._subscription_ref is None:
            return []

        envelope = _build_pull_messages_envelope(
            subscription_ref=self._subscription_ref,
            timeout_seconds=min(self._timeout_seconds, 30),
            message_limit=10,
        )

        resp_text = await self._send_soap(self._subscription_ref, envelope, "PullMessages")
        _check_soap_fault(resp_text, "PullMessages")

        raw_msgs = _parse_pull_messages(resp_text)
        events: list[OnvifEvent] = []
        for msg in raw_msgs:
            topic = msg.get("topic", "")
            ts_str = msg.get("timestamp")
            ts = _parse_utc_time(ts_str) if ts_str else datetime.now(tz=timezone.utc)
            data = msg.get("data", {})
            event = OnvifEvent.from_topic(
                camera_name=self._camera_name,
                raw_topic=topic,
                timestamp=ts,
                raw_message=data,
            )
            events.append(event)

        return events

    async def unsubscribe(self) -> None:
        """Terminate the current PullPoint subscription."""
        if self._subscription_ref is None:
            return

        envelope = _build_unsubscribe_envelope()
        try:
            resp_text = await self._send_soap(self._subscription_ref, envelope, "Unsubscribe")
            _check_soap_fault(resp_text, "Unsubscribe")
        except Exception:
            log.debug("Unsubscribe error for %s", self._camera_name, exc_info=True)
        finally:
            self._subscription_ref = None

    async def _poll_once(self) -> list[OnvifEvent]:
        """Send a single PullMessages request and fire callbacks for events."""
        events = await self.pull_messages()
        if events and self._callback is not None:
            for event in events:
                self._last_event_time = event.timestamp
                try:
                    await self._callback(event)
                except Exception:
                    log.debug("Callback error for %s event", self._camera_name, exc_info=True)
        return events

    async def _poll_loop(self) -> None:
        """Poll for messages at the configured interval until cancelled."""
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except OnvifError:
                log.warning(
                    "PullPoint error for %s, will retry",
                    self._camera_name,
                    exc_info=True,
                )
            except Exception:
                log.warning(
                    "Unexpected error in poll loop for %s",
                    self._camera_name,
                    exc_info=True,
                )

            await asyncio.sleep(self._poll_interval)

    async def _resolve_event_xaddr(self) -> str:
        """Resolve the event service URL from device GetCapabilities."""
        ptz = self._client.ptz
        device_xaddr = ptz.device_xaddr

        body = (
            '<tds:GetCapabilities xmlns:tds="http://www.onvif.org/ver10/device/wsdl">'
            "<tds:Category>All</tds:Category>"
            "</tds:GetCapabilities>"
        )
        action = "http://www.onvif.org/ver10/device/wsdl/GetCapabilities"
        envelope = _soap_envelope(body, action)

        resp_text = await self._send_soap(device_xaddr, envelope, "GetCapabilities")

        xaddr = _parse_event_capability(resp_text)
        if xaddr is None:
            # Fallback: derive from device URL
            xaddr = _derive_event_xaddr(device_xaddr)
        return xaddr

    async def _send_soap(
        self,
        url: str,
        envelope: str,
        operation: str,
    ) -> str:
        """Send a SOAP request and return the response text.

        Raises OnvifError on HTTP errors.
        """
        ptz = self._client.ptz
        try:
            async with self._make_client() as client:
                resp = await client.post(
                    url,
                    content=envelope,
                    headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                    auth=_wsse_auth(ptz.username, ptz.password),
                )
                resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OnvifError(f"{operation} timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise OnvifError(f"{operation} auth failed: invalid credentials") from exc
            raise OnvifError(f"{operation} HTTP error: {exc}") from exc
        except httpx.RequestError as exc:
            raise OnvifError(f"{operation} request failed: {exc}") from exc

        return resp.text


def _parse_event_capability(response_text: str) -> str | None:
    """Extract the Events.XAddr from a GetCapabilities response."""
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError:
        return None

    # Search for XAddr within Events capabilities
    for elem in root.iter():
        local = _local_tag(elem.tag)
        if local == "XAddr":
            parent = _find_parent_containing(root, elem, "Events")
            if parent is not None and elem.text:
                return elem.text.strip()

    return None


def _find_parent_containing(
    root: ET.Element, target: ET.Element, keyword: str
) -> ET.Element | None:
    """Find a parent element whose tag contains the given keyword."""
    for parent in root.iter():
        for child in parent:
            if child is target:
                if keyword.lower() in parent.tag.lower():
                    return parent
                for grandparent in root.iter():
                    for pc in grandparent:
                        if pc is parent and keyword.lower() in grandparent.tag.lower():
                            return grandparent
                return None
    return None


def _derive_event_xaddr(device_xaddr: str) -> str:
    """Derive a likely event service URL from the device service URL."""
    # http://192.168.1.100/onvif/device_service -> http://192.168.1.100/onvif/event_service
    return device_xaddr.replace("device_service", "event_service")


def _parse_utc_time(ts_str: str) -> datetime:
    """Parse a UTC timestamp string into a datetime object."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f+00:00",
        "%Y-%m-%dT%H:%M:%S+00:00",
    ):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Fallback: return now
    return datetime.now(tz=timezone.utc)


# Module-level registry of active subscribers (for web route status)
_active_subscribers: dict[str, OnvifEventSubscriber] = {}


def register_subscriber(camera_name: str, subscriber: OnvifEventSubscriber) -> None:
    """Register an active subscriber for status tracking."""
    _active_subscribers[camera_name] = subscriber


def unregister_subscriber(camera_name: str) -> None:
    """Remove a subscriber from the active registry."""
    _active_subscribers.pop(camera_name, None)


def get_active_subscribers() -> dict[str, OnvifEventSubscriber]:
    """Return a copy of the active subscriber registry."""
    return dict(_active_subscribers)


def get_subscription_states() -> list[dict[str, Any]]:
    """Return subscription state dicts for all active subscribers.

    Each dict has: camera_name, is_running, subscription_ref, last_event_time.
    """
    states: list[dict[str, Any]] = []
    for name, sub in _active_subscribers.items():
        states.append(
            {
                "camera_name": name,
                "is_running": sub.is_running,
                "subscription_ref": sub.subscription_ref,
                "last_event_time": (
                    sub.last_event_time.isoformat() if sub.last_event_time else None
                ),
            }
        )
    return states

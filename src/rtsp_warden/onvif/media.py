"""ONVIF Media service: find a camera's RTSP stream URIs for the add-camera form.

``discover_stream_uris`` tries the common ONVIF ports in order. On each port it asks the
device service for the camera clock (unauthenticated), then calls GetCapabilities, Media
GetProfiles and GetStreamUri for the first two profiles, signing every call with a
WS-UsernameToken whose Created time follows the camera clock (HTTP Digest is offered as
well, for firmware that wants it). Cameras behind a port forward report their LAN address
in every XAddr and URI, so each one is pointed back at the host the camera was reached on.
"""

from __future__ import annotations

import ipaddress
import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from xml.sax.saxutils import escape

import httpx

from .discovery import OnvifError
from .soap import (
    DEVICE_NS,
    MEDIA_NS,
    OnvifAuthError,
    check_soap_fault,
    find_local,
    local_tag,
    soap_envelope,
    wsse_header,
)

log = logging.getLogger(__name__)

DEFAULT_PORTS: tuple[int, ...] = (80, 8080, 888, 2020)
DEVICE_SERVICE_PATH = "/onvif/device_service"

_HOST_ERROR = "host must be a host name or IP address, without a port, path or credentials"
_FORBIDDEN_HOST_CHARS = frozenset("/?#@[]\\%")

_GET_SYSTEM_DATE_AND_TIME = "<tds:GetSystemDateAndTime/>"
_GET_CAPABILITIES = "<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>"
_GET_PROFILES = "<trt:GetProfiles/>"


@dataclass(slots=True)
class StreamUris:
    """What GetStreamUri reported, already pointed at the host the camera was reached on."""

    host: str  # the host the camera was reached on (IPv6 without brackets)
    port: int  # ONVIF port that answered
    main: str | None  # rtsp URI of the first profile, host rewritten, no userinfo
    sub: str | None  # rtsp URI of the second profile; None when absent or same as main
    profiles: list[str]  # profile tokens in camera order


def _normalize_host(host: str) -> str:
    """Return ``host`` without IPv6 brackets; raise ValueError for anything else."""
    value = host.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value or any(ch in _FORBIDDEN_HOST_CHARS or ch.isspace() for ch in value):
        raise ValueError(_HOST_ERROR)
    if ":" in value:
        try:
            ipaddress.IPv6Address(value)
        except ValueError:
            raise ValueError(_HOST_ERROR) from None
    return value


def _netloc(host: str, port: int | None) -> str:
    literal = f"[{host}]" if ":" in host else host
    return literal if port is None else f"{literal}:{port}"


def rewrite_host(uri: str, host: str) -> str:
    """Point ``uri`` at ``host``: keep scheme, port, path and query; drop any userinfo.

    Raises ValueError for an invalid ``host`` or a ``uri`` without scheme and host. The
    messages never echo either argument (a URI may carry credentials).
    """
    target = _normalize_host(host)
    parts = urlsplit(uri.strip())
    if not parts.scheme or not parts.hostname:
        raise ValueError("URI has no scheme or host")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("URI has an invalid port") from None
    return urlunsplit((parts.scheme, _netloc(target, port), parts.path, parts.query, ""))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _get_stream_uri_body(token: str) -> str:
    return (
        "<trt:GetStreamUri>"
        "<trt:StreamSetup>"
        "<tt:Stream>RTP-Unicast</tt:Stream>"
        "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>"
        "</trt:StreamSetup>"
        f"<trt:ProfileToken>{escape(token)}</trt:ProfileToken>"
        "</trt:GetStreamUri>"
    )


def _parse_camera_utc(root: ET.Element) -> datetime | None:
    """Read SystemDateAndTime/UTCDateTime; None when absent or not a valid date."""
    utc = find_local(root, "UTCDateTime")
    if utc is None:
        return None
    values: list[int] = []
    for name in ("Year", "Month", "Day", "Hour", "Minute", "Second"):
        elem = find_local(utc, name)
        text = "" if elem is None else (elem.text or "").strip()
        if not text.isdigit():
            return None
        values.append(int(text))
    year, month, day, hour, minute, second = values
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None


def _media_xaddr(root: ET.Element) -> str | None:
    """The XAddr directly under a Capabilities ``Media`` element, if any."""
    for elem in root.iter():
        if local_tag(elem.tag) != "Media":
            continue
        for child in elem:
            text = (child.text or "").strip()
            if local_tag(child.tag) == "XAddr" and text:
                return text
    return None


def _profile_tokens(root: ET.Element) -> list[str]:
    tokens: list[str] = []
    for elem in root.iter():
        if local_tag(elem.tag) == "Profiles":
            token = (elem.get("token") or "").strip()
            if token and token not in tokens:
                tokens.append(token)
    return tokens


@dataclass(slots=True)
class _PortSession:
    """One ONVIF port on one host: shared client, credentials and camera clock offset."""

    client: httpx.AsyncClient
    host: str
    port: int
    username: str
    password: str
    clock: Callable[[], datetime]
    offset: timedelta = timedelta(0)

    def url(self, path: str = DEVICE_SERVICE_PATH, query: str = "") -> str:
        return urlunsplit(("http", _netloc(self.host, self.port), path, query, ""))

    async def post(
        self, url: str, action: str, body_xml: str, header_xml: str = ""
    ) -> httpx.Response:
        return await self.client.post(
            url,
            content=soap_envelope(body_xml, header_xml=header_xml).encode("utf-8"),
            headers={"Content-Type": f'application/soap+xml; charset=utf-8; action="{action}"'},
        )

    async def sync_clock(self) -> None:
        """Set ``offset`` = camera UTC - local UTC from GetSystemDateAndTime.

        The call is unauthenticated (ONVIF allows it). A 401 or a fault leaves the offset
        at zero; an HTTP error page without a date means this port is not ONVIF.
        """
        action = f"{DEVICE_NS}/GetSystemDateAndTime"
        resp = await self.post(self.url(), action, _GET_SYSTEM_DATE_AND_TIME)
        if resp.status_code == 401:
            return
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            raise OnvifError(f"HTTP {resp.status_code}, not an ONVIF reply") from None
        if find_local(root, "Fault") is not None:
            return
        camera_now = _parse_camera_utc(root)
        if camera_now is None:
            if resp.status_code >= 400:
                raise OnvifError(f"HTTP {resp.status_code}, not an ONVIF reply")
            return
        self.offset = camera_now - self.clock()

    async def call(self, url: str, action: str, body_xml: str) -> ET.Element:
        """POST one signed request; return the parsed reply or raise OnvifError."""
        operation = action.rsplit("/", 1)[-1]
        header = ""
        if self.username:
            now = self.clock() + self.offset
            header = wsse_header(self.username, self.password, now=now)
        resp = await self.post(url, action, body_xml, header)
        if resp.status_code == 401:
            raise OnvifAuthError(f"{operation}: HTTP 401")
        try:
            check_soap_fault(resp.text)
        except OnvifAuthError:
            raise
        except OnvifError as exc:
            raise OnvifError(f"{operation}: {exc}") from exc
        if resp.status_code >= 400:
            raise OnvifError(f"{operation}: HTTP {resp.status_code}")
        try:
            return ET.fromstring(resp.content)
        except ET.ParseError:
            raise OnvifError(f"{operation}: reply is not XML") from None

    def media_url(self, xaddr: str | None) -> str:
        """The Media XAddr's path and query on the host and port that answered."""
        if not xaddr:
            return self.url()
        parts = urlsplit(xaddr)
        if not parts.path:
            return self.url()
        return self.url(parts.path, parts.query)

    async def stream_uri(self, media_url: str, token: str) -> str | None:
        root = await self.call(media_url, f"{MEDIA_NS}/GetStreamUri", _get_stream_uri_body(token))
        elem = find_local(root, "Uri")
        uri = "" if elem is None else (elem.text or "").strip()
        if not uri:
            return None
        try:
            return rewrite_host(uri, self.host)
        except ValueError as exc:
            raise OnvifError(f"GetStreamUri: {exc}") from None

    async def discover(self) -> StreamUris:
        await self.sync_clock()
        caps = await self.call(self.url(), f"{DEVICE_NS}/GetCapabilities", _GET_CAPABILITIES)
        media_url = self.media_url(_media_xaddr(caps))
        profiles_root = await self.call(media_url, f"{MEDIA_NS}/GetProfiles", _GET_PROFILES)
        profiles = _profile_tokens(profiles_root)
        if not profiles:
            raise OnvifError("camera reported no media profiles")
        main = await self.stream_uri(media_url, profiles[0])
        if main is None:
            raise OnvifError(f"no stream URI for profile {profiles[0]}")
        sub: str | None = None
        if len(profiles) > 1:
            try:
                sub = await self.stream_uri(media_url, profiles[1])
            except OnvifAuthError:
                raise
            except OnvifError as exc:
                log.info("ONVIF %s:%d: no sub stream URI (%s)", self.host, self.port, exc)
        if sub == main:
            sub = None
        return StreamUris(host=self.host, port=self.port, main=main, sub=sub, profiles=profiles)


async def discover_stream_uris(
    host: str,
    username: str,
    password: str,
    *,
    ports: Sequence[int] = DEFAULT_PORTS,
    timeout_s: float = 3.0,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], datetime] | None = None,
) -> StreamUris:
    """Ask the camera at ``host`` for its RTSP stream URIs over ONVIF.

    Ports are tried in order; the first port whose device and media services answer wins.
    Connection errors, timeouts, non-ONVIF replies and non-auth faults move on to the next
    port, so a dead host costs at most ``len(ports) * timeout_s`` seconds. An auth failure
    raises ``OnvifAuthError`` (message starts with "authentication failed") at once,
    without trying other ports. Everything else raises ``OnvifError`` whose message names
    every port tried. ``clock`` returns the local UTC time (tests inject a fixed one).
    """
    try:
        target = _normalize_host(host)
    except ValueError as exc:
        raise OnvifError(str(exc)) from None
    port_list = list(ports)
    if not port_list:
        raise OnvifError("no ONVIF ports to try")
    for port in port_list:
        if not 1 <= port <= 65535:
            raise OnvifError(f"invalid ONVIF port {port}")

    client_kwargs: dict[str, Any] = {"timeout": httpx.Timeout(timeout_s)}
    if username:
        client_kwargs["auth"] = httpx.DigestAuth(username, password)
    if transport is not None:
        client_kwargs["transport"] = transport

    failures: list[str] = []
    async with httpx.AsyncClient(**client_kwargs) as client:
        for port in port_list:
            session = _PortSession(
                client=client,
                host=target,
                port=port,
                username=username,
                password=password,
                clock=clock or _utc_now,
            )
            try:
                found = await session.discover()
            except OnvifAuthError as exc:
                raise OnvifAuthError(f"authentication failed on port {port} ({exc})") from exc
            except httpx.TimeoutException:
                failures.append(f"{port} timed out")
            except httpx.RequestError:
                failures.append(f"{port} connection failed")
            except OnvifError as exc:
                failures.append(f"{port} {exc}")
            else:
                log.info("ONVIF media service for %s answered on port %d", target, port)
                return found
    tried = ", ".join(str(port) for port in port_list)
    raise OnvifError(
        f"no ONVIF media service found on {target} (tried ports {tried}: {'; '.join(failures)})"
    )

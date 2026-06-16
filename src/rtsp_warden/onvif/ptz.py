"""ONVIF PTZ client using handcrafted SOAP envelopes over httpx.

Provides continuous move and stop operations for ONVIF-compliant PTZ cameras.
Uses httpx.AsyncClient for HTTP transport; does NOT require zeep for SOAP
encoding (envelopes are handcrafted for full control and easy testing).
"""

from __future__ import annotations

import logging
import uuid
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from .discovery import OnvifError

log = logging.getLogger(__name__)

# ONVIF SOAP namespaces
SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
WSA_NS = "http://schemas.xmlsoap.org/ws/2004/08/addressing"
ONVIF_DEVICE_NS = "http://www.onvif.org/ver10/device/wsdl"
ONVIF_PTZ_NS = "http://www.onvif.org/ver20/ptz/wsdl"
ONVIF_MEDIA_NS = "http://www.onvif.org/ver10/media/wsdl"
ONVIF_SCHEMA = "http://www.onvif.org/ver10/schema"

NS = {
    "soap": SOAP_NS,
    "wsa": WSA_NS,
    "tds": ONVIF_DEVICE_NS,
    "tptz": ONVIF_PTZ_NS,
    "trt": ONVIF_MEDIA_NS,
    "tt": ONVIF_SCHEMA,
}


def _soap_envelope(body_content: str, action: str) -> str:
    """Build a SOAP 1.2 envelope with WS-Addressing headers."""
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


class OnvifPTZ:
    """Minimal PTZ controller for a single ONVIF camera.

    Requires the camera's device service URL (from discovery) and
    optionally username/password for WSSE authentication.
    """

    def __init__(
        self,
        device_xaddr: str,
        ptz_xaddr: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout_seconds: int = 10,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.device_xaddr = device_xaddr
        self._ptz_xaddr = ptz_xaddr
        self.username = username
        self.password = password
        self.timeout_seconds = timeout_seconds
        self._profile_token: str | None = None
        self._transport = transport

    def _make_client(self, timeout: float | None = None) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with optional transport override."""
        kwargs: dict[str, Any] = {"timeout": timeout or self.timeout_seconds}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    async def _get_ptz_xaddr(self) -> str:
        """Call device service's GetCapabilities to resolve PTZ service URL."""
        if self._ptz_xaddr:
            return self._ptz_xaddr

        body = f'<tds:GetCapabilities xmlns:tds="{ONVIF_DEVICE_NS}"><tds:Category>All</tds:Category></tds:GetCapabilities>'  # noqa: E501
        action = f"{ONVIF_DEVICE_NS}/GetCapabilities"
        envelope = _soap_envelope(body, action)

        try:
            async with self._make_client() as client:
                resp = await client.post(
                    self.device_xaddr,
                    content=envelope,
                    headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                    auth=_wsse_auth(self.username, self.password),
                )
                resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OnvifError(f"GetCapabilities timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise OnvifError("Authentication failed: invalid credentials") from exc
            raise OnvifError(f"GetCapabilities HTTP error: {exc}") from exc
        except httpx.RequestError as exc:
            raise OnvifError(f"GetCapabilities request failed: {exc}") from exc

        ptz_xaddr = _parse_capabilities_ptz(resp.text)
        if not ptz_xaddr:
            raise OnvifError("No PTZ capability found in GetCapabilities response")
        self._ptz_xaddr = ptz_xaddr
        return ptz_xaddr

    async def _get_profile_token(self) -> str:
        """Get the first media profile token from the device."""
        if self._profile_token:
            return self._profile_token

        ptz_xaddr = await self._get_ptz_xaddr()

        body = f'<tptz:GetProfiles xmlns:tptz="{ONVIF_PTZ_NS}"></tptz:GetProfiles>'
        action = f"{ONVIF_PTZ_NS}/GetProfiles"
        envelope = _soap_envelope(body, action)

        try:
            async with self._make_client() as client:
                resp = await client.post(
                    ptz_xaddr,
                    content=envelope,
                    headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                    auth=_wsse_auth(self.username, self.password),
                )
                resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OnvifError(f"GetProfiles timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise OnvifError("Authentication failed: invalid credentials") from exc
            raise OnvifError(f"GetProfiles HTTP error: {exc}") from exc
        except httpx.RequestError as exc:
            raise OnvifError(f"GetProfiles request failed: {exc}") from exc

        token = _parse_profile_token(resp.text)
        if not token:
            raise OnvifError("No profile token found in GetProfiles response")
        self._profile_token = token
        return token

    async def continuous_move(
        self,
        pan: float = 0,
        tilt: float = 0,
        zoom: float = 0,
        timeout_seconds: float | None = None,
    ) -> None:
        """Send a ContinuousMove request to the PTZ service.

        Args:
            pan: -1.0 (left) to 1.0 (right)
            tilt: -1.0 (down) to 1.0 (up)
            zoom: -1.0 (out) to 1.0 (in)
            timeout_seconds: optional per-request timeout override
        """
        pan = max(-1.0, min(1.0, pan))
        tilt = max(-1.0, min(1.0, tilt))
        zoom = max(-1.0, min(1.0, zoom))

        ptz_xaddr = await self._get_ptz_xaddr()
        profile_token = await self._get_profile_token()

        velocity = _build_velocity_xml(pan, tilt, zoom)
        body = (
            f'<tptz:ContinuousMove xmlns:tptz="{ONVIF_PTZ_NS}">'
            f"<tptz:ProfileToken>{profile_token}</tptz:ProfileToken>"
            f"<tptz:Velocity>{velocity}</tptz:Velocity>"
            f"</tptz:ContinuousMove>"
        )
        action = f"{ONVIF_PTZ_NS}/ContinuousMove"
        envelope = _soap_envelope(body, action)

        timeout = timeout_seconds or self.timeout_seconds
        await self._send_ptz_request(ptz_xaddr, envelope, "ContinuousMove", timeout)

    async def stop(self) -> None:
        """Stop all PTZ movement on the camera."""
        ptz_xaddr = await self._get_ptz_xaddr()
        profile_token = await self._get_profile_token()

        body = (
            f'<tptz:Stop xmlns:tptz="{ONVIF_PTZ_NS}">'
            f"<tptz:ProfileToken>{profile_token}</tptz:ProfileToken>"
            f"<tptz:PanTilt>true</tptz:PanTilt>"
            f"<tptz:Zoom>true</tptz:Zoom>"
            f"</tptz:Stop>"
        )
        action = f"{ONVIF_PTZ_NS}/Stop"
        envelope = _soap_envelope(body, action)

        await self._send_ptz_request(ptz_xaddr, envelope, "Stop", self.timeout_seconds)

    async def absolute_move(
        self,
        pan: float,
        tilt: float,
        zoom: float,
    ) -> None:
        """Move to an absolute PTZ position.

        Args:
            pan: target pan position (normalized 0-1 typically)
            tilt: target tilt position
            zoom: target zoom position
        """
        ptz_xaddr = await self._get_ptz_xaddr()
        profile_token = await self._get_profile_token()

        position = _build_position_xml(pan, tilt, zoom)
        body = (
            f'<tptz:AbsoluteMove xmlns:tptz="{ONVIF_PTZ_NS}">'
            f"<tptz:ProfileToken>{profile_token}</tptz:ProfileToken>"
            f"<tptz:Position>{position}</tptz:Position>"
            f"</tptz:AbsoluteMove>"
        )
        action = f"{ONVIF_PTZ_NS}/AbsoluteMove"
        envelope = _soap_envelope(body, action)

        await self._send_ptz_request(ptz_xaddr, envelope, "AbsoluteMove", self.timeout_seconds)

    async def _send_ptz_request(
        self,
        url: str,
        envelope: str,
        operation: str,
        timeout: float,
    ) -> None:
        """Send a SOAP request to the PTZ service and check for errors."""
        try:
            async with self._make_client(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    content=envelope,
                    headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                    auth=_wsse_auth(self.username, self.password),
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

        # Check for SOAP faults
        _check_soap_fault(resp.text, operation)


class OnvifClient:
    """High-level ONVIF client combining discovery and PTZ control."""

    def __init__(
        self,
        device_xaddr: str,
        username: str | None = None,
        password: str | None = None,
        timeout_seconds: int = 10,
    ) -> None:
        self.ptz = OnvifPTZ(
            device_xaddr=device_xaddr,
            username=username,
            password=password,
            timeout_seconds=timeout_seconds,
        )


def _build_velocity_xml(pan: float, tilt: float, zoom: float) -> str:
    """Build the PTZ Velocity XML element for ContinuousMove."""
    return (
        f'<tt:PTZSpeed xmlns:tt="{ONVIF_SCHEMA}">'
        f'<tt:PanTilt x="{pan:.4f}" y="{tilt:.4f}"/>'
        f'<tt:Zoom x="{zoom:.4f}"/>'
        f"</tt:PTZSpeed>"
    )


def _build_position_xml(pan: float, tilt: float, zoom: float) -> str:
    """Build the PTZ Position XML element for AbsoluteMove."""
    return (
        f'<tt:PTZVector xmlns:tt="{ONVIF_SCHEMA}">'
        f'<tt:PanTilt x="{pan:.4f}" y="{tilt:.4f}"/>'
        f'<tt:Zoom x="{zoom:.4f}"/>'
        f"</tt:PTZVector>"
    )


def _wsse_auth(username: str | None, password: str | None) -> httpx.DigestAuth | None:
    """Build WSSE auth headers for ONVIF.

    For v1, we use HTTP digest auth which many ONVIF cameras accept.
    Returns None if no credentials are configured.
    """
    if username is None or password is None:
        return None
    return httpx.DigestAuth(username, password)


def _parse_capabilities_ptz(response_text: str) -> str | None:
    """Extract the PTZ.XAddr from a GetCapabilities response."""
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError:
        return None

    # Search for XAddr within PTZ capabilities
    for elem in root.iter():
        local = _local_tag(elem.tag)
        if local == "XAddr":
            # Check if this XAddr is inside a PTZ element
            parent = _find_parent_containing(root, elem, "PTZ")
            if parent is not None:
                return elem.text.strip() if elem.text else None

    # Fallback: return first XAddr found (some devices have simple structure)
    for elem in root.iter():
        if _local_tag(elem.tag) == "XAddr" and elem.text:
            return elem.text.strip()

    return None


def _parse_profile_token(response_text: str) -> str | None:
    """Extract the first profile token from a GetProfiles response."""
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError:
        return None

    for elem in root.iter():
        local = _local_tag(elem.tag)
        if local == "Profiles":
            token = elem.get("token")
            if token:
                return token
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
                # Walk up one more level
                for grandparent in root.iter():
                    for pc in grandparent:
                        if pc is parent and keyword.lower() in grandparent.tag.lower():
                            return grandparent
                return None
    return None


def _local_tag(tag: str) -> str:
    """Extract the local name from an XML tag (strip namespace)."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _check_soap_fault(response_text: str, operation: str) -> None:
    """Check a SOAP response for fault elements and raise OnvifError if found."""
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError:
        return  # Can't parse; assume success

    for elem in root.iter():
        if _local_tag(elem.tag) == "Fault":
            # Extract fault reason
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

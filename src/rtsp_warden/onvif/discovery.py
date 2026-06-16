"""WS-Discovery probe for ONVIF Network Video Transmitters.

Sends a UDP multicast probe to 239.255.255.250:3702 and collects
responses from ONVIF-compliant devices on the local network.
"""

from __future__ import annotations

import logging
import select
import socket
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass

log = logging.getLogger(__name__)

# WS-Discovery multicast endpoint
MULTICAST_ADDR = "239.255.255.250"
MULTICAST_PORT = 3702

# XML namespaces used in WS-Discovery and ONVIF responses
NS = {
    "soap": "http://www.w3.org/2003/05/soap-envelope",
    "wsa": "http://schemas.xmlsoap.org/ws/2004/08/addressing",
    "wsd": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
    "dn": "http://www.onvif.org/ver10/network/wsdl",
    "d": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
}

# ONVIF NVT type identifier
NVT_TYPE = "dn:NetworkVideoTransmitter"

PROBE_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
               xmlns:wsd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
               xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <soap:Header>
    <wsa:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>
    <wsa:MessageID>{message_id}</wsa:MessageID>
    <wsa:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>
  </soap:Header>
  <soap:Body>
    <wsd:Probe>
      <wsd:Types>dn:NetworkVideoTransmitter</wsd:Types>
    </wsd:Probe>
  </soap:Body>
</soap:Envelope>"""


@dataclass(slots=True)
class DiscoveredCamera:
    """An ONVIF camera discovered via WS-Discovery."""

    xaddr: str  # Device service URL
    address: str  # IP address extracted from XAddrs
    name: str | None  # Optional friendly name from scopes
    manufacturer: str | None  # Parsed from scopes (onvif://www.onvif.org/Manufacturer)
    model: str | None  # Parsed from scopes (onvif://www.onvif.org/Model)
    types: list[str]  # ONVIF types from probe match


class OnvifError(Exception):
    """Base exception for ONVIF operations."""


class OnvifDiscovery:
    """Send a WS-Discovery probe and collect NVT responses.

    Uses UDP multicast on 239.255.255.250:3702.
    """

    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout_seconds = timeout_seconds

    def discover(self) -> list[DiscoveredCamera]:
        """Broadcast a WS-Discovery probe and return discovered cameras.

        Returns an empty list if no cameras respond within the timeout.
        Raises OnvifError if the network is unreachable.
        """
        message_id = uuid.uuid4().urn
        probe_xml = PROBE_TEMPLATE.format(message_id=message_id)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", 0))
            sock.setblocking(False)
        except OSError as exc:
            sock.close()
            raise OnvifError(f"Failed to bind discovery socket: {exc}") from exc

        try:
            sock.sendto(probe_xml.encode("utf-8"), (MULTICAST_ADDR, MULTICAST_PORT))
        except OSError as exc:
            sock.close()
            raise OnvifError(f"Failed to send discovery probe: {exc}") from exc

        cameras: dict[str, DiscoveredCamera] = {}
        deadline = _monotonic() + self.timeout_seconds

        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                break

            readable, _, _ = select.select([sock], [], [], min(remaining, 0.5))
            if not readable:
                continue

            try:
                data, addr = sock.recvfrom(65535)
            except OSError:
                continue

            camera = _parse_probe_response(data)
            if camera is None:
                continue

            # Deduplicate by address (IP)
            if camera.address not in cameras:
                cameras[camera.address] = camera

        sock.close()
        return list(cameras.values())


def _monotonic() -> float:
    """Return a monotonic clock value (for test patching)."""
    import time

    return time.monotonic()


def _parse_probe_response(data: bytes) -> DiscoveredCamera | None:
    """Parse a WS-Discovery ProbeMatches response into a DiscoveredCamera.

    Returns None if the response is not a valid NVT probe match.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None

    # Find ProbeMatches in the SOAP body
    body = root.find("soap:Body", NS)
    if body is None:
        # Try without namespace prefix (some devices use default ns)
        body = root.find(".//{http://www.w3.org/2003/05/soap-envelope}Body")
    if body is None:
        return None

    probe_match = body.find("wsd:ProbeMatches/wsd:ProbeMatch", NS)
    if probe_match is None:
        # Try alternate namespace
        for pm in root.iter():
            local = _local_tag(pm.tag)
            if local == "ProbeMatch":
                probe_match = pm
                break
    if probe_match is None:
        return None

    # Extract XAddrs
    xaddrs_elem = probe_match.find("wsd:XAddrs", NS)
    if xaddrs_elem is None:
        for child in probe_match:
            if _local_tag(child.tag) == "XAddrs":
                xaddrs_elem = child
                break
    if xaddrs_elem is None or not xaddrs_elem.text:
        return None

    xaddrs_text = xaddrs_elem.text.strip()
    # XAddrs may contain multiple space-separated URLs; take the first
    xaddr = xaddrs_text.split()[0] if xaddrs_text else ""

    # Extract Types
    types_elem = probe_match.find("wsd:Types", NS)
    types_text = types_elem.text.strip() if types_elem is not None and types_elem.text else ""
    types_list = types_text.split() if types_text else []

    # Filter: only include NetworkVideoTransmitter responses
    is_nvt = any("NetworkVideoTransmitter" in t for t in types_list)
    if not is_nvt:
        # Also check if types include any ONVIF type even if not exact NVT
        is_nvt = any("onvif" in t.lower() for t in types_list)
    if not is_nvt and not types_list:
        # Some cameras omit types; include them anyway if XAddrs present
        is_nvt = True
    if not is_nvt:
        return None

    # Extract address from XAddrs URL
    address = _extract_address(xaddr)

    # Parse scopes for metadata
    scopes_elem = probe_match.find("wsd:Scopes", NS)
    scopes_text = scopes_elem.text.strip() if scopes_elem is not None and scopes_elem.text else ""
    scopes = scopes_text.split() if scopes_text else []

    name = _parse_scope_value(scopes, "name")
    manufacturer = _parse_scope_value(scopes, "Manufacturer")
    model = _parse_scope_value(scopes, "Model")

    return DiscoveredCamera(
        xaddr=xaddr,
        address=address,
        name=name,
        manufacturer=manufacturer,
        model=model,
        types=types_list,
    )


def _local_tag(tag: str) -> str:
    """Extract the local name from an XML tag (strip namespace)."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _extract_address(xaddr: str) -> str:
    """Extract IP address from an XAddrs URL."""
    try:
        # http://192.168.1.100:80/onvif/device_service -> 192.168.1.100
        after_scheme = xaddr.split("://", 1)[1] if "://" in xaddr else xaddr
        host_port = after_scheme.split("/", 1)[0]
        host = host_port.split(":")[0]
        return host
    except (IndexError, ValueError):
        return xaddr


def _parse_scope_value(scopes: list[str], key: str) -> str | None:
    """Extract a value from ONVIF scopes like onvif://www.onvif.org/Manufacturer/Hikvision."""
    for scope in scopes:
        # Match case-insensitively for the key
        parts = scope.split("/")
        for i, part in enumerate(parts):
            if part.lower() == key.lower() and i + 1 < len(parts):
                return parts[i + 1]
    return None

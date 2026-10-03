"""Foscam HTTP CGI client (RW-4): the documented camera API, no browser plugin involved.

Foscam's web page needs an ActiveX/NPAPI plugin only to show live video; every setting on
it is a plain HTTP call to ``/cgi-bin/CGIProxy.fcgi?cmd=<name>&usr=<user>&pwd=<pass>`` (the
"IPCam CGI User Guide"), which answers an XML ``<CGI_Result>`` with a ``<result>`` code.
This module wraps the handful of commands the camera-settings page uses: device info,
the main/sub stream profiles, image tuning, mirror/flip, infrared, OSD, snapshot, reboot.

Every call opens its own ``httpx.Client`` (like the actions do) and raises ``FoscamError``
on a transport problem, a non-200 status or a non-zero result. Error texts never carry
the URL or the credentials.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import unquote

import httpx

Stream = Literal["main", "sub"]
STREAMS: tuple[str, ...] = ("main", "sub")

RESULT_TEXT: dict[int, str] = {
    -1: "the camera called the request malformed",
    -2: "the camera rejected the user name or password",
    -3: "the camera denied access to this account",
    -4: "the camera could not execute the command",
    -5: "the camera timed out",
    -7: "the camera reported an unknown error",
}

# Resolution codes as reported by a Foscam C1 V3 (firmware 2.82.2.35); only codes seen on a
# real stream are named, the rest show as "code N". Other models may map differently.
RESOLUTION_LABELS: dict[int, str] = {0: "1280x720", 3: "640x360"}

IMAGE_COMMANDS: dict[str, tuple[str, str]] = {
    "brightness": ("setBrightness", "brightness"),
    "contrast": ("setContrast", "contrast"),
    "hue": ("setHue", "hue"),
    "saturation": ("setSaturation", "saturation"),
    "sharpness": ("setSharpness", "sharpness"),
    "denoise": ("setDenoiseLevel", "level"),
}

_STREAM_PARAM_CMDS = {
    "main": (
        "getMainVideoStreamType",
        "setMainVideoStreamType",
        "getVideoStreamParam",
        "setVideoStreamParam",
    ),
    "sub": (
        "getSubVideoStreamType",
        "setSubVideoStreamType",
        "getSubVideoStreamParam",
        "setSubVideoStreamParam",
    ),
}


def result_text(code: int) -> str:
    return RESULT_TEXT.get(code, f"the camera answered result {code}")


def resolution_label(code: int) -> str:
    name = RESOLUTION_LABELS.get(code)
    return f"{name} (code {code})" if name else f"code {code}"


class FoscamError(Exception):
    """A failed CGI call. ``code`` is the camera's result code when it answered."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DeviceInfo:
    product: str
    firmware: str
    hardware: str
    name: str


@dataclass(frozen=True)
class StreamProfile:
    index: int  # 0-3: the camera keeps four profiles per stream and plays one of them
    resolution: int  # vendor code, see resolution_label()
    bit_rate: int  # bits per second
    frame_rate: int
    gop: int
    vbr: bool


@dataclass(frozen=True)
class ImageSettings:
    brightness: int
    contrast: int
    hue: int
    saturation: int
    sharpness: int
    denoise: int


@dataclass(frozen=True)
class VideoSettings:
    mirror: bool
    flip: bool
    infrared_mode: int  # 0 = automatic, 1 = manual (openInfraLed / closeInfraLed)
    osd_timestamp: bool
    osd_name: bool
    osd_position: int
    osd_temp_humid: bool = False
    osd_mask: bool = False


def _check_stream(stream: str) -> str:
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {STREAMS}, got {stream!r}")
    return stream


def _int(fields: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(fields.get(key, default))
    except (TypeError, ValueError):
        return default


class FoscamClient:
    """One camera's CGI endpoint. Safe to build per request; it keeps no connection."""

    def __init__(
        self,
        host: str,
        *,
        username: str,
        password: str,
        port: int = 88,
        timeout_s: float = 6.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.timeout_s = float(timeout_s)
        self._transport = transport

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}/cgi-bin/CGIProxy.fcgi"

    # -- transport ------------------------------------------------------------

    def _get(self, cmd: str, params: dict[str, Any]) -> httpx.Response:
        query = {"cmd": cmd, "usr": self.username, "pwd": self.password}
        query.update({k: str(v) for k, v in params.items()})
        where = f"{self.host}:{self.port}"
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
                response = client.get(self.base_url, params=query)
        except httpx.HTTPError as exc:
            raise FoscamError(
                f"cannot reach the camera at {where} ({type(exc).__name__})"
            ) from None
        if response.status_code != 200:
            raise FoscamError(f"the camera at {where} answered HTTP {response.status_code}")
        return response

    def call(self, cmd: str, **params: Any) -> dict[str, str]:
        """Run one CGI command; returns the result's child elements, percent-decoded."""
        response = self._get(cmd, params)
        try:
            root = ET.fromstring(response.text.strip())
        except ET.ParseError:
            raise FoscamError(f"the camera gave no readable answer to {cmd}") from None
        fields = {child.tag: unquote((child.text or "").strip()) for child in root}
        code = _int(fields, "result", 0)
        if code != 0:
            raise FoscamError(f"{cmd}: {result_text(code)}", code=code)
        return fields

    # -- device -----------------------------------------------------------------

    def device_info(self) -> DeviceInfo:
        f = self.call("getDevInfo")
        return DeviceInfo(
            product=f.get("productName", ""),
            firmware=f.get("firmwareVer", ""),
            hardware=f.get("hardwareVer", ""),
            name=f.get("devName", ""),
        )

    def snapshot(self) -> bytes:
        response = self._get("snapPicture2", {})
        if not response.headers.get("content-type", "").startswith("image/"):
            raise FoscamError("the camera sent no image for snapPicture2")
        return response.content

    def reboot(self) -> None:
        self.call("rebootSystem")

    # -- streams -----------------------------------------------------------------

    def stream_type(self, stream: str) -> int:
        get_type, _set_type, _get_params, _set_params = _STREAM_PARAM_CMDS[_check_stream(stream)]
        return _int(self.call(get_type), "streamType", 0)

    def set_stream_type(self, stream: str, index: int) -> None:
        _get_type, set_type, _get_params, _set_params = _STREAM_PARAM_CMDS[_check_stream(stream)]
        if not 0 <= int(index) <= 3:
            raise ValueError("stream profile index must be 0-3")
        self.call(set_type, streamType=int(index))

    def stream_profiles(self, stream: str) -> list[StreamProfile]:
        _get_type, _set_type, get_params, _set_params = _STREAM_PARAM_CMDS[_check_stream(stream)]
        f = self.call(get_params)
        return [
            StreamProfile(
                index=i,
                resolution=_int(f, f"resolution{i}"),
                bit_rate=_int(f, f"bitRate{i}"),
                frame_rate=_int(f, f"frameRate{i}"),
                gop=_int(f, f"GOP{i}"),
                vbr=_int(f, f"isVBR{i}") == 1,
            )
            for i in range(4)
        ]

    def set_stream_profile(self, stream: str, profile: StreamProfile) -> None:
        _get_type, _set_type, _get_params, set_params = _STREAM_PARAM_CMDS[_check_stream(stream)]
        if not 0 <= int(profile.index) <= 3:
            raise ValueError("stream profile index must be 0-3")
        self.call(
            set_params,
            streamType=int(profile.index),
            resolution=int(profile.resolution),
            bitRate=int(profile.bit_rate),
            frameRate=int(profile.frame_rate),
            GOP=int(profile.gop),
            isVBR=1 if profile.vbr else 0,
        )

    # -- image -------------------------------------------------------------------

    def image_settings(self) -> ImageSettings:
        f = self.call("getImageSetting")
        return ImageSettings(
            brightness=_int(f, "brightness"),
            contrast=_int(f, "contrast"),
            hue=_int(f, "hue"),
            saturation=_int(f, "saturation"),
            sharpness=_int(f, "sharpness"),
            denoise=_int(f, "denoiseLevel"),
        )

    def set_image_setting(self, name: str, value: int) -> None:
        if name not in IMAGE_COMMANDS:
            raise ValueError(f"unknown image setting {name!r}")
        if not 0 <= int(value) <= 100:
            raise ValueError(f"{name} must be 0-100")
        cmd, param = IMAGE_COMMANDS[name]
        self.call(cmd, **{param: int(value)})

    # -- mirror / flip, infrared, OSD ------------------------------------------------

    def video_settings(self) -> VideoSettings:
        mirror = self.call("getMirrorAndFlipSetting")
        infrared = self.call("getInfraLedConfig")
        osd = self.call("getOSDSetting")
        return VideoSettings(
            mirror=_int(mirror, "isMirror") == 1,
            flip=_int(mirror, "isFlip") == 1,
            infrared_mode=_int(infrared, "mode"),
            osd_timestamp=_int(osd, "isEnableTimeStamp") == 1,
            osd_name=_int(osd, "isEnableDevName") == 1,
            osd_position=_int(osd, "dispPos"),
            osd_temp_humid=_int(osd, "isEnableTempAndHumid") == 1,
            osd_mask=_int(osd, "isEnableOSDMask") == 1,
        )

    def set_mirror(self, on: bool) -> None:
        self.call("mirrorVideo", isMirror=1 if on else 0)

    def set_flip(self, on: bool) -> None:
        self.call("flipVideo", isFlip=1 if on else 0)

    def set_infrared_mode(self, mode: int) -> None:
        if int(mode) not in (0, 1):
            raise ValueError("infrared mode must be 0 (auto) or 1 (manual)")
        self.call("setInfraLedConfig", mode=int(mode))

    def set_infrared(self, on: bool) -> None:
        self.call("openInfraLed" if on else "closeInfraLed")

    def set_osd(
        self,
        *,
        timestamp: bool,
        name: bool,
        position: int,
        temp_humid: bool = False,
        mask: bool = False,
    ) -> None:
        if not 0 <= int(position) <= 3:
            raise ValueError("OSD position must be 0-3")
        self.call(
            "setOSDSetting",
            isEnableTimeStamp=1 if timestamp else 0,
            isEnableTempAndHumid=1 if temp_humid else 0,
            isEnableDevName=1 if name else 0,
            dispPos=int(position),
            isEnableOSDMask=1 if mask else 0,
        )

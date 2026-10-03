"""Foscam CGI client (RW-4): the documented HTTP API on port 88, no plugin involved.

Offline: every request goes through an httpx.MockTransport that plays the camera.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from rtsp_warden.vendors.foscam import (
    RESOLUTION_LABELS,
    FoscamClient,
    FoscamError,
    StreamProfile,
    resolution_label,
    result_text,
)

PASSWORD = "s3cret/pw"


def _xml(body: str, result: int = 0) -> str:
    return f"<CGI_Result><result>{result}</result>{body}</CGI_Result>"


DEV_INFO = _xml(
    "<productName>C1+V3</productName><serialNo>X</serialNo><devName>My%5FCam</devName>"
    "<firmwareVer>2.82.2.35</firmwareVer><hardwareVer>1.12.5.4</hardwareVer>"
)
STREAM_PARAMS = _xml(
    "".join(
        f"<resolution{i}>{r}</resolution{i}><bitRate{i}>{b}</bitRate{i}>"
        f"<frameRate{i}>{f}</frameRate{i}><GOP{i}>{g}</GOP{i}><isVBR{i}>{v}</isVBR{i}>"
        for i, (r, b, f, g, v) in enumerate(
            [
                (0, 2097152, 25, 50, 1),
                (0, 1048576, 15, 30, 0),
                (3, 524288, 15, 60, 1),
                (0, 2097152, 30, 60, 1),
            ]
        )
    )
)
IMAGE = _xml(
    "<brightness>60</brightness><contrast>48</contrast><hue>50</hue>"
    "<saturation>56</saturation><sharpness>48</sharpness><denoiseLevel>50</denoiseLevel>"
)
MIRROR = _xml("<isMirror>0</isMirror><isFlip>1</isFlip>")
IR = _xml("<mode>1</mode>")
OSD = _xml(
    "<isEnableTimeStamp>1</isEnableTimeStamp><isEnableTempAndHumid>0</isEnableTempAndHumid>"
    "<isEnableDevName>1</isEnableDevName><dispPos>0</dispPos><isEnableOSDMask>0</isEnableOSDMask>"
)
JPEG = b"\xff\xd8\xff\xe0fake\xff\xd9"


class Camera:
    """Plays a Foscam: answers by ``cmd`` and records every request's query."""

    def __init__(self) -> None:
        self.queries: list[dict[str, str]] = []
        self.answers: dict[str, str | bytes | int] = {
            "getDevInfo": DEV_INFO,
            "getMainVideoStreamType": _xml("<streamType>1</streamType>"),
            "getSubVideoStreamType": _xml("<streamType>0</streamType>"),
            "getVideoStreamParam": STREAM_PARAMS,
            "getSubVideoStreamParam": STREAM_PARAMS,
            "getImageSetting": IMAGE,
            "getMirrorAndFlipSetting": MIRROR,
            "getInfraLedConfig": IR,
            "getOSDSetting": OSD,
            "snapPicture2": JPEG,
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        query = {k: v[0] for k, v in parse_qs(urlsplit(str(request.url)).query).items()}
        self.queries.append(query)
        answer = self.answers.get(query.get("cmd", ""), _xml(""))
        if isinstance(answer, bytes):
            return httpx.Response(200, content=answer, headers={"content-type": "image/jpeg"})
        if isinstance(answer, int):
            return httpx.Response(answer, text="nope")
        return httpx.Response(200, text=answer, headers={"content-type": "text/xml"})

    @property
    def cmds(self) -> list[str]:
        return [q["cmd"] for q in self.queries]


@pytest.fixture
def cam() -> Camera:
    return Camera()


@pytest.fixture
def client(cam: Camera) -> FoscamClient:
    return FoscamClient(
        "192.0.2.72",
        port=88,
        username="jc",
        password=PASSWORD,
        transport=httpx.MockTransport(cam.handler),
    )


def test_every_call_goes_to_the_cgi_with_the_credentials_as_query_fields(
    client: FoscamClient, cam: Camera
) -> None:
    info = client.device_info()

    assert cam.queries[0]["cmd"] == "getDevInfo"
    assert (cam.queries[0]["usr"], cam.queries[0]["pwd"]) == ("jc", PASSWORD)
    assert client.base_url == "http://192.0.2.72:88/cgi-bin/CGIProxy.fcgi"
    assert (info.product, info.firmware, info.hardware, info.name) == (
        "C1+V3",
        "2.82.2.35",
        "1.12.5.4",
        "My_Cam",  # percent-decoded
    )


def test_stream_type_and_profiles_are_read_per_stream(client: FoscamClient, cam: Camera) -> None:
    assert client.stream_type("main") == 1
    assert client.stream_type("sub") == 0
    profiles = client.stream_profiles("main")
    assert cam.cmds == ["getMainVideoStreamType", "getSubVideoStreamType", "getVideoStreamParam"]
    assert len(profiles) == 4
    assert profiles[0] == StreamProfile(
        index=0, resolution=0, bit_rate=2097152, frame_rate=25, gop=50, vbr=True
    )
    assert profiles[1].vbr is False
    assert profiles[2].resolution == 3
    client.stream_profiles("sub")
    assert cam.cmds[-1] == "getSubVideoStreamParam"


def test_set_stream_type_and_profile_send_the_documented_fields(
    client: FoscamClient, cam: Camera
) -> None:
    client.set_stream_type("main", 0)
    client.set_stream_type("sub", 2)
    client.set_stream_profile(
        "main",
        StreamProfile(index=1, resolution=0, bit_rate=1048576, frame_rate=20, gop=40, vbr=True),
    )

    assert cam.queries[0]["cmd"] == "setMainVideoStreamType"
    assert cam.queries[0]["streamType"] == "0"
    assert cam.queries[1]["cmd"] == "setSubVideoStreamType"
    assert cam.queries[1]["streamType"] == "2"
    q = cam.queries[2]
    assert q["cmd"] == "setVideoStreamParam"
    assert (
        q["streamType"],
        q["resolution"],
        q["bitRate"],
        q["frameRate"],
        q["GOP"],
        q["isVBR"],
    ) == (
        "1",
        "0",
        "1048576",
        "20",
        "40",
        "1",
    )


def test_image_settings_round_trip(client: FoscamClient, cam: Camera) -> None:
    image = client.image_settings()
    assert (image.brightness, image.contrast, image.hue) == (60, 48, 50)
    assert (image.saturation, image.sharpness, image.denoise) == (56, 48, 50)

    client.set_image_setting("brightness", 70)
    client.set_image_setting("denoise", 30)
    assert cam.queries[1]["cmd"] == "setBrightness" and cam.queries[1]["brightness"] == "70"
    assert cam.queries[2]["cmd"] == "setDenoiseLevel" and cam.queries[2]["level"] == "30"
    with pytest.raises(ValueError):
        client.set_image_setting("gamma", 1)
    with pytest.raises(ValueError):
        client.set_image_setting("brightness", 101)


def test_video_settings_round_trip(client: FoscamClient, cam: Camera) -> None:
    video = client.video_settings()
    assert cam.cmds == ["getMirrorAndFlipSetting", "getInfraLedConfig", "getOSDSetting"]
    assert (video.mirror, video.flip, video.infrared_mode) == (False, True, 1)
    assert (video.osd_timestamp, video.osd_name, video.osd_position) == (True, True, 0)

    client.set_mirror(True)
    client.set_flip(False)
    client.set_infrared_mode(0)
    client.set_infrared(True)
    client.set_infrared(False)
    client.set_osd(timestamp=False, name=True, position=2)
    sent = cam.queries[3:]
    assert [q["cmd"] for q in sent] == [
        "mirrorVideo",
        "flipVideo",
        "setInfraLedConfig",
        "openInfraLed",
        "closeInfraLed",
        "setOSDSetting",
    ]
    assert sent[0]["isMirror"] == "1" and sent[1]["isFlip"] == "0" and sent[2]["mode"] == "0"
    osd = sent[5]
    assert (osd["isEnableTimeStamp"], osd["isEnableDevName"], osd["dispPos"]) == ("0", "1", "2")
    assert (osd["isEnableTempAndHumid"], osd["isEnableOSDMask"]) == ("0", "0")


def test_snapshot_returns_the_jpeg_and_reboot_sends_the_command(
    client: FoscamClient, cam: Camera
) -> None:
    assert client.snapshot() == JPEG
    client.reboot()
    assert cam.cmds == ["snapPicture2", "rebootSystem"]


def test_non_zero_result_raises_with_the_documented_meaning(
    client: FoscamClient, cam: Camera
) -> None:
    cam.answers["getDevInfo"] = _xml("", result=-2)
    with pytest.raises(FoscamError) as info:
        client.device_info()
    assert info.value.code == -2
    assert "user name or password" in str(info.value)
    assert result_text(-3) == "the camera denied access to this account"
    assert result_text(-99).startswith("the camera answered result -99")


def test_errors_never_carry_the_password(cam: Camera) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"boom {request.url}", request=request)

    client = FoscamClient(
        "192.0.2.72", username="jc", password=PASSWORD, transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(FoscamError) as info:
        client.device_info()
    assert PASSWORD not in str(info.value)
    assert "192.0.2.72:88" in str(info.value)

    cam.answers["getDevInfo"] = 500
    client2 = FoscamClient(
        "192.0.2.72", username="jc", password=PASSWORD, transport=httpx.MockTransport(cam.handler)
    )
    with pytest.raises(FoscamError) as info2:
        client2.device_info()
    assert PASSWORD not in str(info2.value)
    assert "HTTP 500" in str(info2.value)


def test_snapshot_that_is_not_an_image_is_an_error(client: FoscamClient, cam: Camera) -> None:
    cam.answers["snapPicture2"] = _xml("", result=-2)
    with pytest.raises(FoscamError):
        client.snapshot()


def test_resolution_labels_name_only_verified_codes() -> None:
    assert RESOLUTION_LABELS[0] == "1280x720"
    assert RESOLUTION_LABELS[3] == "640x360"
    assert resolution_label(0) == "1280x720 (code 0)"
    assert resolution_label(7) == "code 7"


def test_stream_name_is_validated(client: FoscamClient) -> None:
    with pytest.raises(ValueError):
        client.stream_type("third")

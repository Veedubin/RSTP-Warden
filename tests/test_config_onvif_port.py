"""CameraConfig.onvif_port: optional ONVIF device-service port (None means 80)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rtsp_warden.config import CameraConfig


class TestOnvifPortField:
    def test_defaults_to_none(self) -> None:
        cam = CameraConfig(name="c", main_url="rtsp://h/m")
        assert cam.onvif_port is None

    def test_accepts_a_port(self) -> None:
        cam = CameraConfig(name="c", main_url="rtsp://h/m", onvif_port=888)
        assert cam.onvif_port == 888

    @pytest.mark.parametrize("port", [0, 65536, -1])
    def test_rejects_out_of_range(self, port: int) -> None:
        with pytest.raises(ValidationError, match="onvif_port"):
            CameraConfig(name="c", main_url="rtsp://h/m", onvif_port=port)

    def test_survives_a_dump_round_trip(self) -> None:
        cam = CameraConfig(name="c", main_url="rtsp://h/m", onvif_port=8080)
        again = CameraConfig.model_validate(cam.model_dump(mode="json"))
        assert again.onvif_port == 8080

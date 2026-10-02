from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.web.routes._common import find_camera, get_cfg, get_config_path


def _req(**state):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


def test_get_cfg_raises_503_when_missing():
    with pytest.raises(HTTPException) as ei:
        get_cfg(_req(cfg=None))
    assert ei.value.status_code == 503


def test_get_config_path_none_or_path():
    assert get_config_path(_req(config_path=None)) is None
    assert get_config_path(_req(config_path="/tmp/x.yaml")) == Path("/tmp/x.yaml")


def test_find_camera():
    cfg = AppConfig(cameras=[CameraConfig(name="a", main_url="rtsp://h/a")])
    assert find_camera(cfg, "a").name == "a"
    assert find_camera(cfg, "zzz") is None

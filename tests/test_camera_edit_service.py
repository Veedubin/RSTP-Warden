"""Unit tests for web/services/camera_edit.py, the camera edit plan (RW-2 Task 8)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml

from rtsp_warden.config import CameraConfig, expand_env
from rtsp_warden.web.services.camera_config import validate_entry
from rtsp_warden.web.services.camera_edit import (
    CameraEditError,
    CameraEditPlan,
    EditCameraInput,
    apply_plan,
    apply_to_running_config,
    edit_form_values,
    env_userinfo,
    join_userinfo,
    plan_camera_edit,
    read_camera_entry,
    split_userinfo,
)

RAW = {
    "name": "front",
    "main_url": "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoMain",
    "sub_url": "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoSub",
    "record": {"enabled": True, "output_dir": "/srv/recordings"},
    "sensitivity": 65,
}
ENV = {"CAM_FRONT_USER": "admin", "CAM_FRONT_PASS": "old%2Fpass"}


def _cam(raw: dict = RAW, env: dict = ENV) -> CameraConfig:
    return CameraConfig.model_validate(expand_env(raw, env))


def _inp(**overrides: object) -> EditCameraInput:
    values: dict[str, object] = {
        "main_url": "rtsp://192.0.2.10:554/videoMain",
        "sub_url": "rtsp://192.0.2.10:554/videoSub",
        "username": "admin",
        "password": "",
        "onvif_port": "",
        "record_enabled": True,
    }
    values.update(overrides)
    return EditCameraInput(**values)  # type: ignore[arg-type]


class TestUserinfo:
    def test_split_and_join_round_trip_env_references(self) -> None:
        userinfo, bare = split_userinfo(RAW["main_url"])
        assert userinfo == "${CAM_FRONT_USER}:${CAM_FRONT_PASS}"
        assert bare == "rtsp://192.0.2.10:554/videoMain"
        assert join_userinfo(userinfo, bare) == RAW["main_url"]

    def test_at_sign_in_the_query_is_not_userinfo(self) -> None:
        assert split_userinfo("rtsp://h/p?a=b@c") == ("", "rtsp://h/p?a=b@c")

    def test_ipv6_host_survives(self) -> None:
        assert split_userinfo("rtsp://u:p@[2001:db8::1]:554/x") == (
            "u:p",
            "rtsp://[2001:db8::1]:554/x",
        )

    def test_join_without_userinfo_changes_nothing(self) -> None:
        assert join_userinfo("", "rtsp://h/x") == "rtsp://h/x"

    def test_env_userinfo_uses_the_camera_slug(self) -> None:
        assert env_userinfo("front-door") == "${CAM_FRONT_DOOR_USER}:${CAM_FRONT_DOOR_PASS}"


class TestFormValues:
    def test_values_come_from_raw_yaml_and_hold_no_secret(self) -> None:
        values = edit_form_values(RAW, _cam())
        assert values["main_url"] == "rtsp://192.0.2.10:554/videoMain"
        assert values["sub_url"] == "rtsp://192.0.2.10:554/videoSub"
        assert values["username"] == "admin"
        assert values["password_entered"] is False
        assert values["proxy_port"] == 9001
        assert "old" not in str(values)
        assert "CAM_FRONT" not in str(values)

    def test_read_camera_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"cameras": [RAW]}), encoding="utf-8")
        assert read_camera_entry(path, "front") == RAW
        with pytest.raises(KeyError):
            read_camera_entry(path, "nope")


class TestPlan:
    def test_unchanged_form_is_empty(self) -> None:
        plan = plan_camera_edit(RAW, _cam(), _inp())
        assert plan == CameraEditPlan()
        assert plan.is_empty

    def test_main_url_change_keeps_the_env_references(self) -> None:
        plan = plan_camera_edit(RAW, _cam(), _inp(main_url="rtsp://192.0.2.10:554/videoMain2"))
        assert plan.patch == {
            "main_url": "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoMain2"
        }
        assert plan.env_values == {}
        assert plan.remove_keys == []
        assert plan.restart is True

    def test_new_password_goes_to_env_values_percent_encoded(self) -> None:
        """(review focus) '/', '#', '?', '@' and '%' survive the round trip."""
        secret = "n3w/p#ss?@%"
        plan = plan_camera_edit(RAW, _cam(), _inp(password=secret))
        assert plan.env_values == {
            "CAM_FRONT_USER": "admin",
            "CAM_FRONT_PASS": "n3w%2Fp%23ss%3F%40%25",
        }
        assert plan.patch == {}
        assert plan.restart is True
        updated = validate_entry(apply_plan(RAW, plan), plan.env_values)
        parts = urlsplit(updated.main_url)
        assert (parts.hostname, parts.port, unquote(parts.password or "")) == (
            "192.0.2.10",
            554,
            secret,
        )

    def test_same_password_typed_again_changes_nothing(self) -> None:
        assert plan_camera_edit(RAW, _cam(), _inp(password="old/pass")).is_empty

    def test_literal_credentials_move_to_env_references_when_changed(self) -> None:
        raw = {"name": "back", "main_url": "rtsp://u:p@192.0.2.11/m"}
        cam = CameraConfig.model_validate(raw)
        unchanged = _inp(main_url="rtsp://192.0.2.11/m", sub_url="", username="u")
        assert plan_camera_edit(raw, cam, unchanged).is_empty
        changed = _inp(main_url="rtsp://192.0.2.11/m", sub_url="", username="u", password="x")
        plan = plan_camera_edit(raw, cam, changed)
        assert plan.patch == {"main_url": "rtsp://${CAM_BACK_USER}:${CAM_BACK_PASS}@192.0.2.11/m"}
        assert plan.env_values == {"CAM_BACK_USER": "u", "CAM_BACK_PASS": "x"}

    def test_user_name_change_alone_keeps_the_current_password(self) -> None:
        raw = {"name": "back", "main_url": "rtsp://u:p%40ss@192.0.2.11/m"}
        cam = CameraConfig.model_validate(raw)
        renamed = _inp(main_url="rtsp://192.0.2.11/m", sub_url="", username="v")
        plan = plan_camera_edit(raw, cam, renamed)
        assert plan.env_values == {"CAM_BACK_USER": "v", "CAM_BACK_PASS": "p%40ss"}

    def test_new_sub_url_borrows_the_main_url_userinfo(self) -> None:
        raw = {"name": "back", "main_url": "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.11/m"}
        cam = CameraConfig.model_validate({"name": "back", "main_url": "rtsp://u:p@192.0.2.11/m"})
        added = _inp(main_url="rtsp://192.0.2.11/m", sub_url="rtsp://192.0.2.11/s", username="u")
        plan = plan_camera_edit(raw, cam, added)
        assert plan.patch == {"sub_url": "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.11/s"}

    def test_blank_sub_url_and_unchecked_recording(self) -> None:
        plan = plan_camera_edit(RAW, _cam(), _inp(sub_url="", record_enabled=False))
        assert plan.remove_keys == ["sub_url"]
        assert plan.patch == {"record": {"enabled": False}}
        assert plan.restart is True
        entry = apply_plan(RAW, plan)
        assert "sub_url" not in entry
        assert entry["record"] == {"enabled": False, "output_dir": "/srv/recordings"}
        assert RAW["record"]["enabled"] is True  # the input mapping is not mutated

    def test_onvif_port_set_and_cleared_without_restart(self) -> None:
        plan = plan_camera_edit(RAW, _cam(), _inp(onvif_port="888"))
        assert plan.patch == {"onvif_port": 888}
        assert plan.restart is False
        with_port = {**RAW, "onvif_port": 888}
        cleared = plan_camera_edit(with_port, _cam(with_port), _inp(onvif_port=" "))
        assert cleared.remove_keys == ["onvif_port"]
        assert cleared.restart is False

    @pytest.mark.parametrize("port", ["0", "70000", "80a", "-1"])
    def test_bad_onvif_port(self, port: str) -> None:
        with pytest.raises(CameraEditError, match="ONVIF port") as exc_info:
            plan_camera_edit(RAW, _cam(), _inp(onvif_port=port))
        assert exc_info.value.field_name == "onvif_port"

    def test_password_typed_into_the_url_is_rejected(self) -> None:
        with pytest.raises(CameraEditError, match="own fields") as exc_info:
            plan_camera_edit(RAW, _cam(), _inp(main_url="rtsp://admin:hunter2@192.0.2.10/x"))
        assert exc_info.value.field_name == "main_url"
        assert "hunter2" not in str(exc_info.value)

    def test_unchanged_url_is_not_revalidated(self) -> None:
        raw = {"name": "back", "main_url": "${BACK_URL}"}
        cam = CameraConfig.model_validate({"name": "back", "main_url": "rtsp://u:p@192.0.2.11/m"})
        assert edit_form_values(raw, cam)["main_url"] == "${BACK_URL}"
        plan = plan_camera_edit(
            raw, cam, _inp(main_url="${BACK_URL}", sub_url="", username="u", record_enabled=False)
        )
        assert plan.patch == {"record": {"enabled": False}}

    def test_env_reference_typed_into_the_url_is_kept(self) -> None:
        typed = "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.10:554/videoMain"
        plan = plan_camera_edit(RAW, _cam(), _inp(main_url=typed))
        assert plan.patch == {"main_url": typed}

    @pytest.mark.parametrize(
        ("field_name", "value", "message"),
        [
            ("main_url", "  ", "required"),
            ("main_url", "http://192.0.2.10/x", "rtsp://"),
            ("main_url", "-i rtsp://x", "rtsp://"),
            ("sub_url", "file:///etc/passwd", "rtsp://"),
        ],
    )
    def test_bad_url(self, field_name: str, value: str, message: str) -> None:
        with pytest.raises(CameraEditError, match=message) as exc_info:
            plan_camera_edit(RAW, _cam(), _inp(**{field_name: value}))
        assert exc_info.value.field_name == field_name


def test_apply_to_running_config_assigns_in_place() -> None:
    cam = _cam()
    plan = plan_camera_edit(RAW, cam, _inp(sub_url="", onvif_port="888", record_enabled=False))
    updated = validate_entry(apply_plan(RAW, plan), ENV)
    record_obj = cam.record
    apply_to_running_config(cam, updated)
    assert cam.sub_url is None
    assert cam.proxy.stream == "main"
    assert cam.onvif_port == 888
    assert cam.record is record_obj
    assert cam.record.enabled is False

"""Tests for web/services/camera_config.py (names, ports, URLs, raw YAML edits)."""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml

from rtsp_warden import ports
from rtsp_warden.config import AppConfig, load_config
from rtsp_warden.ffmpeg import redact_url
from rtsp_warden.status_model import redact_rtsp_url
from rtsp_warden.web.services.camera_config import (
    NAME_RE,
    NewCameraInput,
    allocate_proxy_port,
    append_camera,
    build_rtsp_url,
    credential_env_values,
    env_slug,
    env_var_names,
    patch_camera,
    raw_camera_entry,
    remove_camera,
    update_raw_config,
    url_with_env_credentials,
    validate_entry,
    validate_name,
)


def _app_config(*ports_in_use: int) -> AppConfig:
    cams = [
        {"name": f"cam{i}", "main_url": "rtsp://h/m", "proxy": {"port": p}}
        for i, p in enumerate(ports_in_use)
    ]
    return AppConfig.model_validate({"cameras": cams})


class TestNames:
    @pytest.mark.parametrize("name", ["front", "Front-Door", "cam_1", "9lives", "a" * 32])
    def test_valid_names(self, name: str) -> None:
        assert NAME_RE.match(name)
        assert validate_name(name, []) == name

    def test_name_is_stripped(self) -> None:
        assert validate_name("  front  ", []) == "front"

    @pytest.mark.parametrize(
        "name", ["", "   ", "-front", "_x", "front door", "a/b", "a.b", "%Y", "café", "a" * 33]
    )
    def test_invalid_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            validate_name(name, [])

    @pytest.mark.parametrize("name", ["new", "New", "NEW"])
    def test_new_is_reserved(self, name: str) -> None:
        with pytest.raises(ValueError, match="reserved"):
            validate_name(name, [])

    def test_duplicate_in_another_case_is_rejected(self) -> None:
        """(review focus) 'front' next to 'Front' would be one directory on macOS/Windows."""
        with pytest.raises(ValueError, match="already exists"):
            validate_name("front", ["back", "Front"])

    def test_names_sharing_a_slug_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="CAM_FRONT_DOOR_USER"):
            validate_name("front-door", ["front_door"])

    def test_unrelated_existing_names_are_fine(self) -> None:
        assert validate_name("garage", ["front", "back"]) == "garage"


class TestSlugAndVariables:
    def test_env_slug(self) -> None:
        assert env_slug("front-door") == "FRONT_DOOR"
        assert env_slug("Cam_1") == "CAM_1"

    def test_env_var_names(self) -> None:
        assert env_var_names("FRONT") == ("CAM_FRONT_USER", "CAM_FRONT_PASS")

    def test_credential_values_are_percent_encoded(self) -> None:
        values = credential_env_values("FRONT", "ad min", "p/w#?@%")
        assert values == {"CAM_FRONT_USER": "ad%20min", "CAM_FRONT_PASS": "p%2Fw%23%3F%40%25"}


class TestAllocateProxyPort:
    def test_skips_ports_used_by_cameras_without_probing_them(self, monkeypatch) -> None:
        probed: list[int] = []

        def fake_free(host: str, port: int) -> bool:
            probed.append(port)
            return True

        monkeypatch.setattr(ports, "port_is_free", fake_free)
        assert allocate_proxy_port(_app_config(9001, 9002)) == 9003
        assert probed == [9003]

    def test_skips_ports_that_are_busy(self, monkeypatch) -> None:
        busy = {9002, 9003}
        monkeypatch.setattr(ports, "port_is_free", lambda host, port: port not in busy)
        assert allocate_proxy_port(_app_config(9001)) == 9004

    def test_passes_host_and_honours_start(self, monkeypatch) -> None:
        seen: list[tuple[str, int]] = []

        def fake_free(host: str, port: int) -> bool:
            seen.append((host, port))
            return True

        monkeypatch.setattr(ports, "port_is_free", fake_free)
        assert allocate_proxy_port(_app_config(), host="127.0.0.1", start=9100) == 9100
        assert seen == [("127.0.0.1", 9100)]

    def test_no_free_port_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(ports, "port_is_free", lambda host, port: False)
        with pytest.raises(ValueError, match="No free proxy port"):
            allocate_proxy_port(_app_config(), start=65530)


class TestUrlBuilding:
    def test_build_rtsp_url_uses_env_references(self) -> None:
        url = build_rtsp_url("192.0.2.10", 554, "videoMain", "FRONT")
        assert url == "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoMain"

    def test_build_rtsp_url_normalises_the_leading_slash(self) -> None:
        a = build_rtsp_url("cam.local", 554, "/live/ch0", "X")
        b = build_rtsp_url("cam.local", 554, "live/ch0", "X")
        assert a == b == "rtsp://${CAM_X_USER}:${CAM_X_PASS}@cam.local:554/live/ch0"
        assert build_rtsp_url("cam.local", 554, "", "X").endswith("@cam.local:554/")

    def test_build_rtsp_url_brackets_ipv6(self) -> None:
        url = build_rtsp_url("fe80::1", 554, "/m", "X")
        assert url == "rtsp://${CAM_X_USER}:${CAM_X_PASS}@[fe80::1]:554/m"

    @pytest.mark.parametrize("host", ["", "rtsp://h", "u@h", "h/path"])
    def test_build_rtsp_url_rejects_bad_hosts(self, host: str) -> None:
        with pytest.raises(ValueError):
            build_rtsp_url(host, 554, "/m", "X")

    def test_env_credentials_added_to_a_bare_onvif_uri(self) -> None:
        url = url_with_env_credentials("rtsp://192.0.2.10:554/videoMain?a=1", "FRONT")
        assert url == "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoMain?a=1"

    def test_env_credentials_replace_literal_userinfo(self) -> None:
        url = url_with_env_credentials("rtsp://admin:pw@192.0.2.10/m", "FRONT")
        assert url == "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10/m"

    def test_existing_env_references_are_kept(self) -> None:
        url = "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.10:554/m"
        assert url_with_env_credentials(url, "FRONT") == url

    @pytest.mark.parametrize("url", ["http://h/m", "file:///etc/passwd", "-i"])
    def test_non_rtsp_urls_are_rejected(self, url: str) -> None:
        with pytest.raises(ValueError, match="rtsp://"):
            url_with_env_credentials(url, "X")

    def test_unparseable_url_error_does_not_echo_the_password(self) -> None:
        with pytest.raises(ValueError) as ei:
            url_with_env_credentials("rtsp://admin:pa/ss@192.0.2.10/m", "X")
        assert "pa/ss" not in str(ei.value)


class TestRawCameraEntry:
    def _inp(self, **kw: object) -> NewCameraInput:
        base: dict[str, object] = {
            "name": "front",
            "host": "192.0.2.10",
            "username": "admin",
            "password": "s3cret/#?@%",
        }
        base.update(kw)
        return NewCameraInput(**base)  # type: ignore[arg-type]

    def test_default_main_url_and_env_values(self) -> None:
        entry, env = raw_camera_entry(self._inp(), 9003)
        assert entry == {
            "name": "front",
            "main_url": "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/",
            "record": {"enabled": True},
            "proxy": {"enabled": True, "mode": "mjpeg", "stream": "main", "port": 9003},
        }
        assert env == {"CAM_FRONT_USER": "admin", "CAM_FRONT_PASS": "s3cret%2F%23%3F%40%25"}

    def test_entry_never_contains_the_password(self) -> None:
        entry, _env = raw_camera_entry(
            self._inp(main_url="rtsp://192.0.2.10:554/videoMain", sub_url="rtsp://192.0.2.10/s"),
            9001,
        )
        dumped = yaml.safe_dump(entry)
        assert "s3cret" not in dumped
        assert "${CAM_FRONT_USER}" in dumped

    def test_given_urls_get_env_references_and_sub_selects_the_proxy_stream(self) -> None:
        entry, _env = raw_camera_entry(
            self._inp(
                main_url="rtsp://192.0.2.10:554/videoMain",
                sub_url="rtsp://192.0.2.10:554/videoSub",
                onvif_port=888,
                record_enabled=False,
            ),
            9002,
        )
        assert entry["main_url"] == (
            "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoMain"
        )
        assert entry["sub_url"] == (
            "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoSub"
        )
        assert entry["onvif_port"] == 888
        assert entry["record"] == {"enabled": False}
        assert entry["proxy"]["stream"] == "sub"
        assert list(entry) == ["name", "main_url", "sub_url", "onvif_port", "record", "proxy"]

    def test_blank_sub_url_is_omitted(self) -> None:
        entry, _env = raw_camera_entry(self._inp(sub_url="   "), 9001)
        assert "sub_url" not in entry
        assert entry["proxy"]["stream"] == "main"

    def test_credentials_typed_into_the_url_move_to_env(self) -> None:
        entry, env = raw_camera_entry(
            self._inp(username="", password="", main_url="rtsp://admin:p%40ss@192.0.2.10/m"),
            9001,
        )
        assert entry["main_url"] == "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10/m"
        assert env == {"CAM_FRONT_USER": "admin", "CAM_FRONT_PASS": "p%40ss"}

    def test_no_credentials_means_no_userinfo_and_no_env(self) -> None:
        entry, env = raw_camera_entry(self._inp(username="", password=""), 9001)
        assert entry["main_url"] == "rtsp://192.0.2.10:554/"
        assert env == {}

    def test_user_written_env_references_are_kept_verbatim(self) -> None:
        url = "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.10:554/m"
        entry, env = raw_camera_entry(self._inp(username="", password="", main_url=url), 9001)
        assert entry["main_url"] == url
        assert env == {}

    def test_slug_follows_the_name(self) -> None:
        entry, env = raw_camera_entry(self._inp(name="front-door"), 9001)
        assert "${CAM_FRONT_DOOR_USER}" in entry["main_url"]
        assert set(env) == {"CAM_FRONT_DOOR_USER", "CAM_FRONT_DOOR_PASS"}

    def test_missing_host_and_url_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            raw_camera_entry(self._inp(host="", username=""), 9001)


# --- raw config.yaml edits ------------------------------------------------------------

_BASE_CONFIG = {
    "runtime": {"ffmpeg_path": "ffmpeg", "workspace_dir": "./workspace"},
    "onvif": {"username": "admin", "password": "${ONVIF_PASS}"},
    "cameras": [
        {
            "name": "front",
            "main_url": "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.10:554/videoMain",
            "record": {"enabled": True, "output_dir": "/srv/rec"},
            "proxy": {"enabled": True, "port": 9001, "fps": 5},
            "zones": [{"name": "road", "frame_width": 640, "frame_height": 360}],
            "detectors": [{"type": "motion", "min_area": 500}],
        },
        {
            "name": "back",
            "main_url": "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.11:554/m",
            "sub_url": "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.11:554/s",
            "proxy": {"port": 9002},
        },
    ],
}

_NEW_ENTRY = {
    "name": "garage",
    "main_url": "rtsp://${CAM_GARAGE_USER}:${CAM_GARAGE_PASS}@192.0.2.12:554/",
    "record": {"enabled": True},
    "proxy": {"enabled": True, "mode": "mjpeg", "stream": "main", "port": 9003},
}


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_BASE_CONFIG, sort_keys=False), encoding="utf-8")
    return path


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestAppendCamera:
    def test_appends_and_keeps_everything_else(self, config_path: Path) -> None:
        append_camera(config_path, dict(_NEW_ENTRY))

        data = _load(config_path)
        assert [c["name"] for c in data["cameras"]] == ["front", "back", "garage"]
        assert data["cameras"][2] == _NEW_ENTRY
        assert data["cameras"][0] == _BASE_CONFIG["cameras"][0]
        assert data["runtime"] == _BASE_CONFIG["runtime"]
        text = config_path.read_text(encoding="utf-8")
        assert "${ONVIF_PASS}" in text
        assert "${CAM_USER}:${CAM_PASS}" in text
        assert "${CAM_GARAGE_USER}:${CAM_GARAGE_PASS}" in text

    def test_rejects_a_name_already_in_the_file_in_any_case(self, config_path: Path) -> None:
        before = config_path.read_text(encoding="utf-8")
        with pytest.raises(ValueError, match="already in config.yaml"):
            append_camera(config_path, {**_NEW_ENTRY, "name": "Front"})
        assert config_path.read_text(encoding="utf-8") == before

    def test_creates_the_cameras_list_when_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("runtime:\n  ffmpeg_path: ffmpeg\n", encoding="utf-8")
        append_camera(path, dict(_NEW_ENTRY))
        assert _load(path)["cameras"] == [_NEW_ENTRY]

    def test_concurrent_appends_do_not_lose_cameras(self, config_path: Path) -> None:
        names = [f"cam{i}" for i in range(8)]
        threads = [
            threading.Thread(target=append_camera, args=(config_path, {**_NEW_ENTRY, "name": n}))
            for n in names
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        loaded = [c["name"] for c in _load(config_path)["cameras"]]
        assert sorted(loaded) == sorted(["front", "back", *names])


class TestPatchCamera:
    def test_patches_only_the_named_camera_and_given_keys(self, config_path: Path) -> None:
        patch_camera(
            config_path,
            "front",
            {"main_url": "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.20:554/x", "onvif_port": 888},
        )

        data = _load(config_path)
        front, back = data["cameras"]
        assert front["main_url"] == "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.20:554/x"
        assert front["onvif_port"] == 888
        assert front["zones"] == _BASE_CONFIG["cameras"][0]["zones"]
        assert front["detectors"] == _BASE_CONFIG["cameras"][0]["detectors"]
        assert back == _BASE_CONFIG["cameras"][1]

    def test_record_and_proxy_merge_one_level(self, config_path: Path) -> None:
        patch_camera(config_path, "front", {"record": {"enabled": False}, "proxy": {"port": 9010}})

        front = _load(config_path)["cameras"][0]
        assert front["record"] == {"enabled": False, "output_dir": "/srv/rec"}
        assert front["proxy"] == {"enabled": True, "port": 9010, "fps": 5}

    def test_record_is_created_when_the_entry_has_none(self, config_path: Path) -> None:
        patch_camera(config_path, "back", {"record": {"enabled": False}})
        assert _load(config_path)["cameras"][1]["record"] == {"enabled": False}

    def test_remove_keys(self, config_path: Path) -> None:
        patch_camera(config_path, "back", {}, remove_keys=("sub_url", "not_there"))
        back = _load(config_path)["cameras"][1]
        assert "sub_url" not in back
        assert back["main_url"] == _BASE_CONFIG["cameras"][1]["main_url"]

    def test_unknown_camera_raises_key_error(self, config_path: Path) -> None:
        with pytest.raises(KeyError):
            patch_camera(config_path, "nope", {"onvif_port": 80})

    def test_rename_is_refused(self, config_path: Path) -> None:
        with pytest.raises(ValueError, match="renamed"):
            patch_camera(config_path, "front", {"name": "porch"})
        with pytest.raises(ValueError, match="renamed"):
            patch_camera(config_path, "front", {}, remove_keys=("name",))

    def test_entry_name_with_surrounding_spaces_still_matches(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump({"cameras": [{"name": " front ", "main_url": "rtsp://h/m"}]}),
            encoding="utf-8",
        )
        patch_camera(path, "front", {"onvif_port": 8080})
        assert _load(path)["cameras"][0]["onvif_port"] == 8080


class TestRemoveCamera:
    def test_removes_only_that_camera(self, config_path: Path) -> None:
        remove_camera(config_path, "front")
        data = _load(config_path)
        assert [c["name"] for c in data["cameras"]] == ["back"]
        assert data["onvif"] == _BASE_CONFIG["onvif"]

    def test_removing_the_last_camera_leaves_an_empty_list(self, config_path: Path) -> None:
        remove_camera(config_path, "front")
        remove_camera(config_path, "back")
        assert _load(config_path)["cameras"] == []
        assert "cameras: []" in config_path.read_text(encoding="utf-8")

    def test_unknown_camera_raises_key_error(self, config_path: Path) -> None:
        with pytest.raises(KeyError):
            remove_camera(config_path, "nope")


class TestValidateEntry:
    def test_expands_and_validates(self) -> None:
        cam = validate_entry(_NEW_ENTRY, {"CAM_GARAGE_USER": "admin", "CAM_GARAGE_PASS": "pw"})
        assert cam.name == "garage"
        assert cam.main_url == "rtsp://admin:pw@192.0.2.12:554/"
        assert cam.proxy.port == 9003

    def test_missing_variable_becomes_value_error(self) -> None:
        with pytest.raises(ValueError, match="CAM_GARAGE_USER"):
            validate_entry(_NEW_ENTRY, {})

    def test_validation_error_becomes_value_error_without_values(self) -> None:
        bad = {**_NEW_ENTRY, "proxy": {"port": 70000}}
        with pytest.raises(ValueError) as ei:
            validate_entry(bad, {"CAM_GARAGE_USER": "admin", "CAM_GARAGE_PASS": "hunter2"})
        message = str(ei.value)
        assert "proxy.port" in message
        assert "hunter2" not in message
        assert ei.value.__cause__ is None
        assert ei.value.__suppress_context__ is True


class TestCredentialRoundTrip:
    """(review focus) A password with / # ? @ % must survive and never be printed."""

    PASSWORD = "s3cr/t#?@%$x"

    def test_password_round_trips_and_is_redacted(self, tmp_path: Path) -> None:
        inp = NewCameraInput(
            name="front",
            host="192.0.2.10",
            username="ad@min",
            password=self.PASSWORD,
            main_url="rtsp://192.0.2.10:554/videoMain",
        )
        entry, env = raw_camera_entry(inp, 9001)

        path = tmp_path / "config.yaml"
        path.write_text("cameras: []\n", encoding="utf-8")
        append_camera(path, entry)
        text = path.read_text(encoding="utf-8")
        assert self.PASSWORD not in text
        assert "s3cr" not in text

        cam = validate_entry(entry, env)
        parts = urlsplit(cam.main_url)
        assert parts.hostname == "192.0.2.10"
        assert parts.port == 554
        assert parts.path == "/videoMain"
        assert unquote(parts.username or "") == "ad@min"
        assert unquote(parts.password or "") == self.PASSWORD

        for redacted in (redact_rtsp_url(cam.main_url), redact_url(cam.main_url)):
            assert "s3cr" not in redacted
            assert env["CAM_FRONT_PASS"] not in redacted
            assert "192.0.2.10:554/videoMain" in redacted

    def test_written_config_loads_with_the_env_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inp = NewCameraInput(
            name="front", host="192.0.2.10", username="admin", password=self.PASSWORD
        )
        entry, env = raw_camera_entry(inp, 9001)
        path = tmp_path / "config.yaml"
        path.write_text("cameras: []\n", encoding="utf-8")
        append_camera(path, entry)
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        cfg = load_config(path)

        assert unquote(urlsplit(cfg.cameras[0].main_url).password or "") == self.PASSWORD


class TestUpdateRawConfig:
    """Every config.yaml read-modify-write goes through one lock (zones, presets, cameras)."""

    def test_mutate_changes_the_raw_file(self, config_path: Path) -> None:
        def set_port(data: dict) -> None:
            data["cameras"][1]["onvif_port"] = 888

        update_raw_config(config_path, set_port)

        data = _load(config_path)
        assert data["cameras"][1]["onvif_port"] == 888
        assert "${CAM_USER}:${CAM_PASS}" in config_path.read_text(encoding="utf-8")

    def test_an_exception_in_mutate_leaves_the_file_untouched(self, config_path: Path) -> None:
        before = config_path.read_text(encoding="utf-8")

        def broken(data: dict) -> None:
            data["cameras"].clear()
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            update_raw_config(config_path, broken)
        assert config_path.read_text(encoding="utf-8") == before

    def test_a_slow_writer_holds_off_append_camera(self, config_path: Path) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow(data: dict) -> None:
            started.set()
            release.wait(5)
            data["cameras"][0]["onvif_port"] = 888

        writer = threading.Thread(target=update_raw_config, args=(config_path, slow))
        writer.start()
        assert started.wait(5)
        adder = threading.Thread(target=append_camera, args=(config_path, dict(_NEW_ENTRY)))
        adder.start()
        adder.join(0.2)
        assert adder.is_alive()  # waits for the slow read-modify-write to finish
        release.set()
        writer.join(5)
        adder.join(5)

        data = _load(config_path)
        assert data["cameras"][0]["onvif_port"] == 888  # the slow change was not lost
        assert [c["name"] for c in data["cameras"]] == ["front", "back", "garage"]

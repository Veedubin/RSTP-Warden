"""Tests for Sprint 2 Batch 5: HLS Playback (HTL endpoint and segment serving).

Validates:
- m3u8 parsing
- HTL playlist generation
- Segment scanning
- HTL endpoint authentication and content-type
- Segment serving with path traversal protection
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from rtsp_warden.auth import hash_password
from rtsp_warden.config import load_config
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import create_admin_user, ensure_schema
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.services.htl import (
    build_htl_playlist,
    get_recordings_dir,
    parse_m3u8,
    scan_segments,
)

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Create a sample config.yaml with 1 camera and a recordings dir."""
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "cameras": [
            {
                "name": "front",
                "main_url": "rtsp://user:pass@192.168.1.50:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.50:554/sub",
                "record": {
                    "enabled": True,
                    "output_dir": str(rec_dir),
                    "main": {"enabled": True, "container": "ts", "chunk_seconds": 300},
                    "sub": {"enabled": False, "container": "ts", "chunk_seconds": 300},
                },
                "proxy": {
                    "enabled": True,
                    "mode": "mjpeg",
                    "stream": "sub",
                    "bind_host": "127.0.0.1",
                    "port": 9001,
                },
            },
        ],
        "runtime": {
            "ffmpeg_path": "ffmpeg",
            "workspace_dir": str(tmp_path / "workspace"),
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


@pytest.fixture
def admin_client(sample_config: Path) -> TestClient:
    """Authenticated TestClient as admin user with config."""
    db_url = f"sqlite:///{sample_config.parent / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg = load_config(sample_config)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    client = TestClient(app)

    # Login as admin
    r = client.get("/login")
    csrf = r.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return client


@pytest.fixture
def unauthenticated_client() -> TestClient:
    """Unauthenticated TestClient (no login)."""
    settings = WebSettings()
    app = create_app(settings)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests: parse_m3u8
# ---------------------------------------------------------------------------


class TestParseM3u8:
    """Tests for the parse_m3u8 function."""

    def test_parse_m3u8_basic(self) -> None:
        m3u8 = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:5.005,
seg-00001.ts
#EXTINF:5.005,
seg-00002.ts
#EXT-X-ENDLIST"""
        segments = parse_m3u8(m3u8)
        assert len(segments) == 2
        assert segments[0]["duration"] == 5.005
        assert segments[0]["uri"] == "seg-00001.ts"
        assert segments[1]["duration"] == 5.005
        assert segments[1]["uri"] == "seg-00002.ts"

    def test_parse_m3u8_empty(self) -> None:
        m3u8 = "#EXTM3U\n#EXT-X-ENDLIST"
        segments = parse_m3u8(m3u8)
        assert len(segments) == 0

    def test_parse_m3u8_with_titles(self) -> None:
        m3u8 = """#EXTM3U
#EXTINF:5.005,Segment 1
seg-00001.ts
#EXTINF:3.003,
seg-00002.ts"""
        segments = parse_m3u8(m3u8)
        assert len(segments) == 2
        assert segments[0]["duration"] == 5.005
        assert segments[1]["duration"] == 3.003

    def test_parse_m3u8_single_segment(self) -> None:
        m3u8 = "#EXTM3U\n#EXTINF:6.0,\nseg.ts\n#EXT-X-ENDLIST"
        segments = parse_m3u8(m3u8)
        assert len(segments) == 1
        assert segments[0]["duration"] == 6.0
        assert segments[0]["uri"] == "seg.ts"


# ---------------------------------------------------------------------------
# Unit tests: build_htl_playlist
# ---------------------------------------------------------------------------


class TestBuildHtlPlaylist:
    """Tests for the build_htl_playlist function."""

    def test_build_htl_playlist_empty_window(self) -> None:
        playlist = build_htl_playlist(
            camera_name="front",
            stream_name="sub",
            start_ts=1000,
            end_ts=2000,
            segments_dir=Path("/tmp/nope"),
            segments_index=[],
        )
        assert "#EXTM3U" in playlist
        assert "#EXT-X-ENDLIST" in playlist
        assert "#EXT-X-VERSION:3" in playlist

    def test_build_htl_playlist_with_segments(self) -> None:
        segments_index = [
            {"path": "seg-00001.ts", "start_time": 1000.0, "duration": 5.0},
            {"path": "seg-00002.ts", "start_time": 1005.0, "duration": 5.0},
            {"path": "seg-00003.ts", "start_time": 1010.0, "duration": 5.0},
        ]
        playlist = build_htl_playlist(
            camera_name="front",
            stream_name="sub",
            start_ts=1002,
            end_ts=1015,
            segments_dir=Path("/recordings/front/sub"),
            segments_index=segments_index,
        )
        assert "#EXTM3U" in playlist
        # All three segments overlap the window [1002, 1015]:
        # seg1: [1000, 1005] overlaps [1002, 1015] -> yes
        # seg2: [1005, 1010] overlaps [1002, 1015] -> yes
        # seg3: [1010, 1015] overlaps [1002, 1015] -> yes
        assert "seg-00001.ts" in playlist
        assert "seg-00002.ts" in playlist
        assert "seg-00003.ts" in playlist
        assert "/segments/front/sub/seg-00001.ts" in playlist

    def test_build_htl_playlist_partial_overlap(self) -> None:
        segments_index = [
            {"path": "seg-00001.ts", "start_time": 1000.0, "duration": 5.0},
            {"path": "seg-00002.ts", "start_time": 1005.0, "duration": 5.0},
            {"path": "seg-00003.ts", "start_time": 1010.0, "duration": 5.0},
        ]
        # Only seg-00002 should overlap: [1005, 1010] overlaps [1006, 1009]
        playlist = build_htl_playlist(
            camera_name="front",
            stream_name="sub",
            start_ts=1006,
            end_ts=1009,
            segments_dir=Path("/recordings/front/sub"),
            segments_index=segments_index,
        )
        assert "seg-00001.ts" not in playlist
        assert "seg-00002.ts" in playlist
        assert "seg-00003.ts" not in playlist

    def test_build_htl_playlist_target_duration(self) -> None:
        segments_index = [
            {"path": "seg-00001.ts", "start_time": 1000.0, "duration": 5.5},
        ]
        playlist = build_htl_playlist(
            camera_name="front",
            stream_name="sub",
            start_ts=999,
            end_ts=1006,
            segments_dir=Path("/recordings/front/sub"),
            segments_index=segments_index,
        )
        assert "#EXT-X-TARGETDURATION:6" in playlist

    def test_build_htl_playlist_exact_boundary(self) -> None:
        """Segment that starts exactly at end_ts should not be included."""
        segments_index = [
            {"path": "seg-00001.ts", "start_time": 1000.0, "duration": 5.0},
            {"path": "seg-00002.ts", "start_time": 1010.0, "duration": 5.0},
        ]
        # Window [1000, 1005]: seg1 [1000, 1005] overlaps, seg2 [1010, 1015] does not
        playlist = build_htl_playlist(
            camera_name="front",
            stream_name="sub",
            start_ts=1000,
            end_ts=1005,
            segments_dir=Path("/recordings/front/sub"),
            segments_index=segments_index,
        )
        # seg-00001 ends at 1005.0, window ends at 1005.0 -> overlaps
        # seg-00002 starts at 1010.0 > 1005.0 -> does not overlap
        assert "seg-00001.ts" in playlist
        assert "seg-00002.ts" not in playlist


# ---------------------------------------------------------------------------
# Unit tests: scan_segments
# ---------------------------------------------------------------------------


class TestScanSegments:
    """Tests for the scan_segments function."""

    def test_scan_segments_basic(self, tmp_path: Path) -> None:
        seg_dir = tmp_path / "front" / "sub"
        seg_dir.mkdir(parents=True)
        (seg_dir / "seg-00001.ts").write_bytes(b"fake_ts_data_1")
        (seg_dir / "seg-00002.ts").write_bytes(b"fake_ts_data_2")
        (seg_dir / "not-a-segment.txt").write_bytes(b"x")

        segments = scan_segments(seg_dir)
        assert len(segments) == 2
        assert all(s["path"].endswith(".ts") for s in segments)
        assert all("start_time" in s for s in segments)
        assert all("duration" in s for s in segments)

    def test_scan_segments_empty_dir(self, tmp_path: Path) -> None:
        seg_dir = tmp_path / "empty"
        seg_dir.mkdir()
        segments = scan_segments(seg_dir)
        assert len(segments) == 0

    def test_scan_segments_nonexistent_dir(self) -> None:
        segments = scan_segments(Path("/nonexistent/path"))
        assert len(segments) == 0

    def test_scan_segments_single_file(self, tmp_path: Path) -> None:
        seg_dir = tmp_path / "single"
        seg_dir.mkdir()
        (seg_dir / "seg-00001.ts").write_bytes(b"fake")

        segments = scan_segments(seg_dir)
        assert len(segments) == 1
        assert segments[0]["path"] == "seg-00001.ts"
        # Last segment gets default duration
        assert segments[0]["duration"] == 5.0

    def test_scan_segments_mtime_ordering(self, tmp_path: Path) -> None:
        seg_dir = tmp_path / "ordered"
        seg_dir.mkdir()
        (seg_dir / "seg-00001.ts").write_bytes(b"a")
        (seg_dir / "seg-00002.ts").write_bytes(b"bb")

        segments = scan_segments(seg_dir)
        assert len(segments) == 2
        # Segments should be sorted by mtime
        assert segments[0]["path"] == "seg-00001.ts"
        assert segments[1]["path"] == "seg-00002.ts"


# ---------------------------------------------------------------------------
# Unit tests: get_recordings_dir
# ---------------------------------------------------------------------------


class TestGetRecordingsDir:
    """Tests for the get_recordings_dir function."""

    def test_get_recordings_dir_found(self, sample_config: Path) -> None:
        cfg = load_config(sample_config)
        rec_dir = get_recordings_dir(cfg, "front", "main")
        assert rec_dir is not None
        assert "front" in str(rec_dir)
        assert "main" in str(rec_dir)

    def test_get_recordings_dir_unknown_camera(self, sample_config: Path) -> None:
        cfg = load_config(sample_config)
        rec_dir = get_recordings_dir(cfg, "nonexistent", "main")
        assert rec_dir is None


# ---------------------------------------------------------------------------
# Integration tests: HTL endpoint
# ---------------------------------------------------------------------------


class TestHtlEndpoint:
    """Tests for the /htl/ route."""

    def test_htl_endpoint_requires_auth(self, unauthenticated_client: TestClient) -> None:
        r = unauthenticated_client.get("/htl/front/sub/1000/2000.m3u8", follow_redirects=False)
        assert r.status_code in (401, 303, 307, 302)

    def test_htl_endpoint_returns_m3u8(self, admin_client: TestClient, sample_config: Path) -> None:
        # Set up recording directory with segments
        cfg = load_config(sample_config)
        rec_dir = cfg.cameras[0].record.output_dir / "front" / "sub"
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / "seg-00001.ts").write_bytes(b"x" * 1000)

        end_ts = int(time.time()) + 3600
        r = admin_client.get(f"/htl/front/sub/0/{end_ts}.m3u8")
        assert r.status_code == 200
        assert "vnd.apple.mpegurl" in r.headers["content-type"]
        assert "#EXTM3U" in r.text
        assert "#EXT-X-ENDLIST" in r.text

    def test_htl_endpoint_unknown_camera(self, admin_client: TestClient) -> None:
        r = admin_client.get("/htl/nonexistent/main/0/9999999.m3u8")
        assert r.status_code == 404

    def test_htl_endpoint_invalid_timestamps(self, admin_client: TestClient) -> None:
        r = admin_client.get("/htl/front/main/2000/1000.m3u8")
        assert r.status_code == 400

    def test_htl_endpoint_empty_window(self, admin_client: TestClient, sample_config: Path) -> None:
        cfg = load_config(sample_config)
        rec_dir = cfg.cameras[0].record.output_dir / "front" / "sub"
        rec_dir.mkdir(parents=True, exist_ok=True)
        # No segments -- should return empty playlist
        r = admin_client.get("/htl/front/sub/0/1000.m3u8")
        assert r.status_code == 200
        assert "#EXTM3U" in r.text
        assert "#EXT-X-ENDLIST" in r.text


# ---------------------------------------------------------------------------
# Integration tests: Segment serving endpoint
# ---------------------------------------------------------------------------


class TestSegmentsEndpoint:
    """Tests for the /segments/ route."""

    def test_segments_endpoint_serves_file(
        self, admin_client: TestClient, sample_config: Path
    ) -> None:
        cfg = load_config(sample_config)
        rec_dir = cfg.cameras[0].record.output_dir / "front" / "sub"
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / "seg-00001.ts").write_bytes(b"\x47" * 188)  # TS sync byte

        r = admin_client.get("/segments/front/sub/seg-00001.ts")
        assert r.status_code == 200
        assert len(r.content) == 188

    def test_segments_endpoint_404_missing(
        self, admin_client: TestClient, sample_config: Path
    ) -> None:
        cfg = load_config(sample_config)
        rec_dir = cfg.cameras[0].record.output_dir / "front" / "sub"
        rec_dir.mkdir(parents=True, exist_ok=True)

        r = admin_client.get("/segments/front/sub/nonexistent.ts")
        assert r.status_code == 404

    def test_segments_endpoint_path_traversal_blocked(self, admin_client: TestClient) -> None:
        r = admin_client.get("/segments/front/sub/../../../etc/passwd")
        assert r.status_code in (400, 403, 404)

    def test_segments_endpoint_path_traversal_double_dot(self, admin_client: TestClient) -> None:
        r = admin_client.get("/segments/front/sub/../main/seg.ts")
        assert r.status_code in (400, 403, 404)

    def test_segments_endpoint_absolute_path_blocked(self, admin_client: TestClient) -> None:
        r = admin_client.get("/segments/front/sub//etc/passwd")
        assert r.status_code in (400, 403, 404)

    def test_segments_endpoint_requires_auth(self) -> None:
        settings = WebSettings()
        app = create_app(settings)
        client = TestClient(app)
        r = client.get("/segments/front/sub/seg.ts", follow_redirects=False)
        assert r.status_code in (401, 303, 307, 302)

    def test_segments_endpoint_unknown_camera(self, admin_client: TestClient) -> None:
        r = admin_client.get("/segments/nonexistent/main/seg.ts")
        assert r.status_code == 404

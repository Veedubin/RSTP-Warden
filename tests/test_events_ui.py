"""Events UI (RW-3 Task 13): event cards, filters, pagination, thumbnails, clips, dashboard strip.

Events are seeded with the Task 7 helpers (``insert_event``, ``update_event``,
``insert_action_run``); thumbnails and clips are small files written under a
temp ``record.output_dir``. Times are aware UTC; expected display strings are
computed with ``astimezone()`` so the tests pass in any server time zone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rtsp_warden import __version__
from rtsp_warden.config import AppConfig, CameraConfig, RecordConfig
from rtsp_warden.db.schema import insert_action_run, insert_event, update_event
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.paths import STATIC_DIR, TEMPLATES_DIR
from rtsp_warden.web.services import events as svc

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"
T0 = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post(
        "/login", data={"username": "admin", "password": "testpass123", "csrf_token": token}
    )


@pytest.fixture
def rec_dir(tmp_path: Path) -> Path:
    path = tmp_path / "recordings"
    path.mkdir()
    return path


@pytest.fixture
def cfg(rec_dir: Path) -> AppConfig:
    return AppConfig(
        cameras=[
            CameraConfig(
                name="front", main_url="rtsp://u:p@h/m", record=RecordConfig(output_dir=rec_dir)
            ),
            CameraConfig(
                name="back", main_url="rtsp://u:p@h/m2", record=RecordConfig(output_dir=rec_dir)
            ),
        ]
    )


@pytest.fixture
def client(db_with_user: str, cfg: AppConfig) -> TestClient:
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: None)
    c = TestClient(app)
    _login(c)
    return c


def _seed(
    rec_dir: Path,
    *,
    camera: str = "front",
    label: str | None = "person",
    event_type: str = "person",
    confidence: float | None = 0.87,
    zone: str = "",
    created_at: datetime = T0,
    thumbnail: bool = True,
) -> int:
    """Insert one event; write its thumbnail under rec_dir when asked."""
    event_id = insert_event(
        camera_name=camera,
        event_type=event_type,
        label=label,
        confidence=confidence,
        zone=zone,
        track_id=7,
        message=f"{label or event_type} on {camera}",
        created_at=created_at,
    )
    if thumbnail:
        rel = f"{camera}/thumbnails/{event_id}.jpg"
        (rec_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (rec_dir / rel).write_bytes(JPEG)
        update_event(event_id, thumbnail_path=rel)
    return event_id


def _clip(rec_dir: Path, event_id: int, suffix: str, data: bytes = b"clipdata") -> Path:
    rel = f"front/clips/{event_id}{suffix}"
    (rec_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (rec_dir / rel).write_bytes(data)
    update_event(event_id, clip_path=rel, ended_at=T0 + timedelta(seconds=20))
    return rec_dir / rel


def _local(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- service


def test_to_db_utc_converts_aware_and_reads_naive_as_local() -> None:
    assert svc.to_db_utc(None) is None
    minus5 = timezone(timedelta(hours=-5))
    assert svc.to_db_utc(datetime(2026, 10, 2, 7, 0, tzinfo=minus5)) == datetime(2026, 10, 2, 12)
    naive_local = datetime(2026, 10, 2, 7, 0)
    expected = naive_local.astimezone().astimezone(timezone.utc).replace(tzinfo=None)
    assert svc.to_db_utc(naive_local) == expected


def test_parse_date_range_covers_whole_days_in_the_given_zone() -> None:
    minus5 = timezone(timedelta(hours=-5))
    since, until = svc.parse_date_range("2026-10-02", "2026-10-02", tz=minus5)
    assert since == datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)
    assert until == datetime(2026, 10, 3, 5, 0, tzinfo=timezone.utc)
    assert svc.parse_date_range(None, None) == (None, None)
    assert svc.parse_date_range("not-a-date", "2026-13-40") == (None, None)


def test_format_local_uses_the_given_zone() -> None:
    minus5 = timezone(timedelta(hours=-5))
    assert svc.format_local(datetime(2026, 10, 2, 12, 0), tz=minus5) == "2026-10-02 07:00:00"
    assert svc.format_local(T0, tz=timezone.utc) == "2026-10-02 12:00:00"
    assert svc.format_local(None) == ""


def test_resolve_media_path_stays_inside_output_dirs(tmp_path: Path, cfg: AppConfig) -> None:
    rec_dir = tmp_path / "recordings"
    thumb = rec_dir / "front" / "thumbnails" / "1.jpg"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(JPEG)
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(JPEG)
    (rec_dir / "front" / "thumbnails" / "2.jpg").symlink_to(secret)

    assert svc.resolve_media_path(cfg, "front", "front/thumbnails/1.jpg") == thumb.resolve()
    assert svc.resolve_media_path(cfg, "front", str(thumb)) == thumb.resolve()
    # A camera deleted from config: its files are still found under the shared output dir.
    assert svc.resolve_media_path(cfg, "garage", "front/thumbnails/1.jpg") == thumb.resolve()
    assert svc.resolve_media_path(cfg, "front", "front/thumbnails/9.jpg") is None
    assert svc.resolve_media_path(cfg, "front", "../secret.jpg") is None
    assert svc.resolve_media_path(cfg, "front", str(secret)) is None
    assert svc.resolve_media_path(cfg, "front", "front/thumbnails/2.jpg") is None
    assert svc.resolve_media_path(cfg, "front", None) is None
    assert svc.resolve_media_path(None, "front", "front/thumbnails/1.jpg") is None


def test_list_events_builds_card_dicts(clean_db: None, tmp_path: Path, cfg: AppConfig) -> None:
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir(exist_ok=True)
    event_id = _seed(rec_dir, zone="driveway")

    events, total = svc.list_events(cfg=cfg)

    assert total == 1
    evt = events[0]
    assert evt["id"] == event_id
    assert evt["camera_name"] == "front"
    assert evt["label"] == "person"
    assert evt["confidence_pct"] == 87
    assert evt["zone"] == "driveway"
    assert evt["created_at"] == T0
    assert evt["started_display"] == _local(T0)
    assert evt["thumbnail_url"] == f"/events/{event_id}/thumbnail.jpg"
    assert evt["thumbnail_expired"] is False
    assert evt["clip_url"] is None
    assert evt["is_test"] is False
    assert evt["detail_url"] == f"/events/{event_id}"


def test_list_events_marks_missing_thumbnail_expired_only_with_a_config(
    clean_db: None, tmp_path: Path, cfg: AppConfig
) -> None:
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir(exist_ok=True)
    event_id = _seed(rec_dir)
    (rec_dir / "front" / "thumbnails" / f"{event_id}.jpg").unlink()

    with_cfg, _ = svc.list_events(cfg=cfg)
    without_cfg, _ = svc.list_events()

    assert with_cfg[0]["thumbnail_url"] is None
    assert with_cfg[0]["thumbnail_expired"] is True
    # The dashboard calls list_events without a config: the URL stays and the
    # card's onerror fallback shows "expired".
    assert without_cfg[0]["thumbnail_url"] == f"/events/{event_id}/thumbnail.jpg"
    assert without_cfg[0]["thumbnail_expired"] is False


def test_list_events_filters_by_camera_name_label_and_window(
    clean_db: None, tmp_path: Path
) -> None:
    rec_dir = tmp_path / "recordings"
    front = _seed(rec_dir, camera="front", label="person", created_at=T0)
    back = _seed(rec_dir, camera="back", label="car", event_type="car", created_at=T0)
    old = _seed(
        rec_dir, camera="front", label="dog", event_type="dog", created_at=T0 - timedelta(days=2)
    )

    assert [e["id"] for e in svc.list_events(camera_name="back")[0]] == [back]
    assert [e["id"] for e in svc.list_events(label="person")[0]] == [front]
    window = svc.list_events(since=T0 - timedelta(hours=1), until=T0 + timedelta(hours=1))
    assert sorted(e["id"] for e in window[0]) == sorted([front, back])
    assert window[1] == 2
    # until is exclusive
    assert svc.list_events(until=T0)[0][0]["id"] == old


def test_list_events_label_falls_back_to_event_type_for_old_rows(
    clean_db: None, tmp_path: Path
) -> None:
    event_id = _seed(
        tmp_path / "recordings", label=None, event_type="motion", confidence=None, thumbnail=False
    )

    events, _ = svc.list_events(label="motion")

    assert [e["id"] for e in events] == [event_id]
    assert events[0]["label"] == "motion"
    assert events[0]["confidence_pct"] is None
    assert events[0]["thumbnail_url"] is None
    assert svc.event_filter_options(None)["labels"] == ["motion"]


def test_list_events_newest_first_with_id_tiebreak(clean_db: None, tmp_path: Path) -> None:
    rec_dir = tmp_path / "recordings"
    first = _seed(rec_dir, created_at=T0, thumbnail=False)
    second = _seed(rec_dir, created_at=T0, thumbnail=False)
    newest = _seed(rec_dir, created_at=T0 + timedelta(seconds=1), thumbnail=False)

    events, total = svc.list_events()

    assert [e["id"] for e in events] == [newest, second, first]
    assert total == 3
    assert svc.get_recent_events(limit=2)[0]["id"] == newest


def test_count_events_by_type_reads_naive_since_as_local_time(
    clean_db: None, tmp_path: Path
) -> None:
    _seed(tmp_path / "recordings", created_at=T0, thumbnail=False)
    before = (T0 - timedelta(hours=1)).astimezone().replace(tzinfo=None)
    after = (T0 + timedelta(hours=1)).astimezone().replace(tzinfo=None)

    assert svc.count_events_by_type(since=before) == {"person": 1}
    assert svc.count_events_by_type(since=after) == {}
    assert svc.count_events_by_type() == {"person": 1}


def test_event_filter_options_merge_config_and_database(
    clean_db: None, tmp_path: Path, cfg: AppConfig
) -> None:
    rec_dir = tmp_path / "recordings"
    _seed(rec_dir, camera="garage", label="cat", event_type="cat", thumbnail=False)
    _seed(rec_dir, camera="front", label="person", thumbnail=False)

    options = svc.event_filter_options(cfg)

    assert options["cameras"] == ["back", "front", "garage"]
    assert options["labels"] == ["cat", "person"]


def test_estimate_clip_seconds_and_playlist() -> None:
    seconds = svc.estimate_clip_seconds(
        T0, T0 + timedelta(seconds=20), pre_seconds=10, post_seconds=10, max_duration=120
    )
    assert seconds == 40.0
    capped = svc.estimate_clip_seconds(
        T0, T0 + timedelta(minutes=10), pre_seconds=10, post_seconds=10, max_duration=120
    )
    assert capped == 120.0
    assert (
        svc.estimate_clip_seconds(T0, None, pre_seconds=5, post_seconds=5, max_duration=120) == 10.0
    )

    text = svc.clip_playlist(12, 40.0)

    assert text.splitlines() == [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "#EXT-X-TARGETDURATION:40",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXTINF:40.000,",
        "/events/12/clip",
        "#EXT-X-ENDLIST",
    ]


# --------------------------------------------------------------------------- pages


def test_events_page_renders_cards(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir, zone="driveway")

    r = client.get("/events")

    assert r.status_code == 200
    assert f'id="event-card-{event_id}"' in r.text
    assert f'<img class="event-thumb" src="/events/{event_id}/thumbnail.jpg"' in r.text
    assert f'<a class="event-label" href="/events/{event_id}">person</a>' in r.text
    assert '<span class="event-confidence">87%</span>' in r.text
    assert "front &middot; driveway" in r.text
    assert _local(T0) in r.text
    assert "1 event" in r.text
    assert f"rtsp-warden v{__version__}" in r.text
    assert "<table" not in r.text


def test_events_page_filter_form_lists_cameras_and_labels(
    client: TestClient, rec_dir: Path
) -> None:
    _seed(rec_dir, camera="garage", label="cat", event_type="cat", thumbnail=False)

    r = client.get("/events?camera=garage")

    assert '<option value="back">back</option>' in r.text
    assert '<option value="garage" selected>garage</option>' in r.text
    assert '<option value="cat">cat</option>' in r.text
    assert '<input type="date" name="from" value="">' in r.text


def test_events_page_filters_by_camera_and_label(client: TestClient, rec_dir: Path) -> None:
    front = _seed(rec_dir, camera="front", label="person")
    back = _seed(rec_dir, camera="back", label="car", event_type="car")

    by_camera = client.get("/events?camera=back").text
    by_label = client.get("/events?label=person").text

    assert f'id="event-card-{back}"' in by_camera
    assert f'id="event-card-{front}"' not in by_camera
    assert f'id="event-card-{front}"' in by_label
    assert f'id="event-card-{back}"' not in by_label
    assert "matching the filters" in by_label


def test_events_page_ignores_empty_filter_values(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)

    r = client.get("/events?camera=&label=&from=&to=")

    assert f'id="event-card-{event_id}"' in r.text
    assert "matching the filters" not in r.text


def test_events_page_date_filter_uses_local_days(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)
    day = T0.astimezone().date()
    next_day = (day + timedelta(days=1)).isoformat()

    same_day = client.get(f"/events?from={day.isoformat()}&to={day.isoformat()}").text
    later = client.get(f"/events?from={next_day}").text

    assert f'id="event-card-{event_id}"' in same_day
    assert f'id="event-card-{event_id}"' not in later
    assert "No events found." in later


def test_events_partial_returns_only_the_filtered_grid(client: TestClient, rec_dir: Path) -> None:
    front = _seed(rec_dir, camera="front")
    back = _seed(rec_dir, camera="back")

    r = client.get("/events/partial?camera=back", headers={"HX-Request": "true"})

    assert r.status_code == 200
    assert '<div class="event-grid">' in r.text
    assert f'id="event-card-{back}"' in r.text
    assert f'id="event-card-{front}"' not in r.text
    assert "<html" not in r.text
    assert "<form" not in r.text


def test_events_page_refreshes_page_one_with_its_filters(client: TestClient, rec_dir: Path) -> None:
    _seed(rec_dir, camera="back")

    r = client.get("/events?camera=back&label=person")

    assert 'hx-get="/events/partial?camera=back&amp;label=person" hx-trigger="every 10s"' in r.text


def test_events_second_page_works(client: TestClient, rec_dir: Path) -> None:
    ids = [
        _seed(rec_dir, created_at=T0 + timedelta(seconds=i), thumbnail=False)
        for i in range(svc.PAGE_SIZE + 2)
    ]

    first = client.get("/events")
    second = client.get("/events?page=2")

    assert first.status_code == 200
    assert 'href="/events?page=2"' in first.text
    assert second.status_code == 200
    assert second.text.count('class="event-card"') == 2
    assert f'id="event-card-{ids[0]}"' in second.text
    assert f'id="event-card-{ids[1]}"' in second.text
    assert 'href="/events" role="button" class="outline secondary">Previous</a>' in second.text
    assert "Page 2 of 2" in second.text
    assert "hx-trigger" not in second.text.split('id="event-grid"')[1].split(">")[0]


def test_events_page_keeps_filters_in_pagination_links(client: TestClient, rec_dir: Path) -> None:
    for i in range(svc.PAGE_SIZE + 1):
        _seed(rec_dir, camera="back", created_at=T0 + timedelta(seconds=i), thumbnail=False)

    r = client.get("/events?camera=back")

    assert 'href="/events?camera=back&amp;page=2"' in r.text


def test_thumbnail_route_serves_the_file(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)

    r = client.get(f"/events/{event_id}/thumbnail.jpg")

    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == JPEG


def test_expired_thumbnail_404s_and_card_says_expired(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)
    (rec_dir / "front" / "thumbnails" / f"{event_id}.jpg").unlink()

    assert client.get(f"/events/{event_id}/thumbnail.jpg").status_code == 404
    page = client.get("/events").text
    assert '<span class="event-thumb-placeholder">expired</span>' in page
    assert f"/events/{event_id}/thumbnail.jpg" not in page


def test_thumbnail_route_404s_for_unknown_event_and_motion_event(
    client: TestClient, rec_dir: Path
) -> None:
    motion = _seed(rec_dir, label="motion", event_type="motion", thumbnail=False)

    assert client.get("/events/9999/thumbnail.jpg").status_code == 404
    assert client.get(f"/events/{motion}/thumbnail.jpg").status_code == 404
    assert '<span class="event-thumb-placeholder">no image</span>' in client.get("/events").text


def test_thumbnail_route_refuses_paths_outside_the_output_dir(
    client: TestClient, rec_dir: Path, tmp_path: Path
) -> None:
    (tmp_path / "secret.jpg").write_bytes(JPEG)
    event_id = _seed(rec_dir, thumbnail=False)
    update_event(event_id, thumbnail_path="../secret.jpg")

    assert client.get(f"/events/{event_id}/thumbnail.jpg").status_code == 404


def test_media_routes_require_login(db_with_user: str, cfg: AppConfig, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)
    anonymous = TestClient(create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: None))

    assert anonymous.get(f"/events/{event_id}/thumbnail.jpg").status_code == 401
    assert anonymous.get(f"/events/{event_id}/clip").status_code == 401


def test_event_detail_shows_thumbnail_details_and_action_runs(
    client: TestClient, rec_dir: Path
) -> None:
    event_id = _seed(rec_dir, zone="driveway")
    update_event(event_id, ended_at=T0 + timedelta(seconds=75))
    insert_action_run(event_id=event_id, action_name="phone", status="ok", error=None)
    insert_action_run(
        event_id=event_id, action_name="hook", status="failed", error="HTTP 500: upstream down"
    )

    r = client.get(f"/events/{event_id}")

    assert r.status_code == 200
    assert f'<img class="event-detail-thumb" src="/events/{event_id}/thumbnail.jpg"' in r.text
    assert "<td>driveway</td>" in r.text
    assert f"<td>{_local(T0 + timedelta(seconds=75))}</td>" in r.text
    assert "<td>1 min 15 s</td>" in r.text
    assert "<td>phone</td>" in r.text
    assert '<span class="action-ok">ok</span>' in r.text
    assert "<td>hook</td>" in r.text
    assert '<span class="action-failed">failed</span>' in r.text
    assert "HTTP 500: upstream down" in r.text
    assert r.text.index("<td>phone</td>") < r.text.index("<td>hook</td>")


def test_event_detail_without_runs_or_clip(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)

    r = client.get(f"/events/{event_id}")

    assert "No actions ran for this event." in r.text
    assert "No clip for this event." in r.text
    assert client.get(f"/events/{event_id}/clip").status_code == 404
    assert client.get(f"/events/{event_id}/clip.m3u8").status_code == 404


def test_mp4_clip_is_served_and_played_inline(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)
    _clip(rec_dir, event_id, ".mp4", b"mp4bytes")

    clip = client.get(f"/events/{event_id}/clip")
    page = client.get(f"/events/{event_id}").text

    assert clip.status_code == 200
    assert clip.headers["content-type"] == "video/mp4"
    assert clip.content == b"mp4bytes"
    assert (
        f'<video class="event-clip" controls preload="metadata" src="/events/{event_id}/clip">'
        in page
    )
    assert client.get(f"/events/{event_id}/clip.m3u8").status_code == 404


def test_ts_clip_gets_a_playlist_and_the_hls_player(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)
    _clip(rec_dir, event_id, ".ts", b"tsbytes")

    clip = client.get(f"/events/{event_id}/clip")
    playlist = client.get(f"/events/{event_id}/clip.m3u8")
    page = client.get(f"/events/{event_id}").text

    assert clip.headers["content-type"] == "video/mp2t"
    assert clip.content == b"tsbytes"
    assert playlist.status_code == 200
    assert playlist.headers["content-type"].startswith("application/vnd.apple.mpegurl")
    lines = playlist.text.splitlines()
    assert lines[0] == "#EXTM3U"
    assert "#EXTINF:40.000," in lines
    assert f"/events/{event_id}/clip" in lines
    assert lines[-1] == "#EXT-X-ENDLIST"
    # partials/hls_player.html (Task 7) renders the player for htl_src.
    assert f"/events/{event_id}/clip.m3u8" in page
    assert "/static/js/hls.min.js" in page
    assert 'id="player"' in page


def test_expired_clip_says_expired(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir)
    _clip(rec_dir, event_id, ".mp4").unlink()

    assert "Clip expired." in client.get(f"/events/{event_id}").text
    assert client.get(f"/events/{event_id}/clip").status_code == 404


def test_test_events_carry_the_badge(client: TestClient, rec_dir: Path) -> None:
    event_id = _seed(rec_dir, event_type="test")

    card = client.get("/events").text
    detail = client.get(f"/events/{event_id}").text

    assert '<span class="event-badge">test</span>' in card
    assert '<span class="event-badge">test</span>' in detail


def test_dashboard_strip_uses_event_cards(client: TestClient, rec_dir: Path) -> None:
    ids = [_seed(rec_dir, created_at=T0 + timedelta(seconds=i), thumbnail=False) for i in range(8)]

    r = client.get("/")

    assert r.status_code == 200
    assert r.text.count('class="event-card"') == 6
    assert f'id="event-card-{ids[-1]}"' in r.text
    assert f'id="event-card-{ids[0]}"' not in r.text
    assert '<a href="/events">All events &rarr;</a>' in r.text
    assert "Recent recordings" not in r.text


def test_dashboard_without_events_says_so(client: TestClient) -> None:
    r = client.get("/")

    assert "No events yet." in r.text
    assert 'class="event-card"' not in r.text


def test_event_card_css_is_shipped_and_old_row_partial_is_gone() -> None:
    css = (STATIC_DIR / "css" / "warden.css").read_text(encoding="utf-8")

    assert "/* --- detection (RW-3) --- */" in css
    assert ".event-grid {" in css
    assert ".event-thumb-placeholder[hidden]" in css
    assert not (TEMPLATES_DIR / "partials" / "event_row.html").exists()


# --------------------------------------------------------------------------- RW-5: night badge


def _seed_with_metadata(label: str, metadata: dict | None) -> int:
    return insert_event(
        camera_name="front",
        event_type="object",
        label=label,
        confidence=0.8,
        zone="",
        track_id=3,
        message=f"{label} on front",
        created_at=T0,
        metadata=metadata,
    )


def test_old_rows_without_night_render_without_badge(client: TestClient, rec_dir: Path) -> None:
    """(review focus) Rows from before 1.4.0 have no night key: no badge, no error."""
    _seed(rec_dir, label="person")
    _seed_with_metadata("cat", {"bbox": [1, 2, 3, 4]})
    _seed_with_metadata("dog", None)
    r = client.get("/events")
    assert r.status_code == 200
    assert ">night<" not in r.text


def test_night_events_show_a_badge_on_the_grid_and_the_detail(
    client: TestClient, rec_dir: Path
) -> None:
    night_id = _seed_with_metadata("fox", {"night": True})
    _seed_with_metadata("cat", {"night": False})
    r = client.get("/events")
    assert r.status_code == 200
    assert r.text.count('<span class="event-badge">night</span>') == 1
    r = client.get(f"/events/{night_id}")
    assert r.status_code == 200
    assert '<span class="event-badge">night</span>' in r.text
    assert '<th scope="row">Night</th><td>yes</td>' in r.text


def test_event_dict_reads_the_night_flag(clean_db: None, cfg: AppConfig) -> None:
    night_id = _seed_with_metadata("raccoon", {"night": True})
    day_id = _seed_with_metadata("cat", {"night": False})
    odd_id = _seed_with_metadata("dog", {"night": "yes"})
    assert svc.get_event_by_id(night_id, cfg)["night"] is True
    assert svc.get_event_by_id(day_id, cfg)["night"] is False
    assert svc.get_event_by_id(odd_id, cfg)["night"] is None
    rows, _total = svc.list_events(cfg=cfg)
    assert {r["id"]: r["night"] for r in rows} == {night_id: True, day_id: False, odd_id: None}

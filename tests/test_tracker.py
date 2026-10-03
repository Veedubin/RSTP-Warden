"""Tests for detectors/tracking.py: the per-camera IoU tracker (spec 7.1, 7.2, 12)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rtsp_warden.detectors.base import Detection
from rtsp_warden.detectors.tracking import Track, Tracker, TrackerUpdate, iou

FRAME_SHAPE = (90, 160, 3)


def _frame(value: int = 0) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def _det(label: str, box: tuple[int, int, int, int], conf: float = 0.8) -> Detection:
    return Detection(kind=label, confidence=conf, bbox=box)


# ---------------------------------------------------------------------------
# iou() and the data types
# ---------------------------------------------------------------------------


def test_iou_identical_boxes_is_one() -> None:
    assert iou((10, 20, 30, 40), (10, 20, 30, 40)) == 1.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((0, 0, 10, 10), (20, 20, 10, 10)),  # far apart
        ((0, 0, 10, 10), (10, 0, 10, 10)),  # edges touch, no shared area
    ],
)
def test_iou_without_shared_area_is_zero(a, b) -> None:
    assert iou(a, b) == 0.0


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 0, 100, 100), (50, 0, 100, 100), 1 / 3),  # half overlap: 5000 / 15000
        ((0, 0, 100, 100), (25, 25, 50, 50), 0.25),  # contained: 2500 / 10000
        ((0, 0, 100, 200), (20, 0, 100, 200), 2 / 3),  # moved 20 px: 16000 / 24000
    ],
)
def test_iou_partial_overlap(a, b, expected) -> None:
    assert iou(a, b) == pytest.approx(expected)
    assert iou(b, a) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [(0, 0, 0, 10), (0, 0, 10, 0), (0, 0, -5, 10)])
def test_iou_with_a_box_without_area_is_zero(bad) -> None:
    assert iou(bad, (0, 0, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), bad) == 0.0


def test_iou_returns_plain_float_for_numpy_inputs() -> None:
    a = tuple(np.int64(v) for v in (0, 0, 100, 100))
    b = tuple(np.int32(v) for v in (50, 0, 100, 100))
    value = iou(a, b)
    assert type(value) is float
    assert value == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iou_threshold": 0.0}, "iou_threshold"),
        ({"iou_threshold": 1.5}, "iou_threshold"),
        ({"grace_seconds": 0.0}, "grace_seconds"),
        ({"min_frames": 0}, "min_frames"),
    ],
)
def test_tracker_rejects_bad_parameters(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        Tracker(**kwargs)


def test_tracker_defaults_match_the_spec() -> None:
    tracker = Tracker()
    assert tracker.iou_threshold == 0.3
    assert tracker.grace_seconds == 3.0
    assert tracker.min_frames == 2


def test_tracker_update_lists_default_to_empty() -> None:
    update = TrackerUpdate()
    assert (update.opened, update.improved, update.closed, update.active) == ([], [], [], [])


def test_track_equality_is_identity_and_repr_hides_the_frame() -> None:
    def make() -> Track:
        return Track(
            id=1,
            label="person",
            first_seen=1.0,
            last_seen=1.0,
            frames=1,
            bbox=(0, 0, 10, 10),
            best_confidence=0.5,
            best_bbox=(0, 0, 10, 10),
            best_frame=np.zeros((4, 4, 3), dtype=np.uint8),
            best_ts=1.0,
        )

    a, b = make(), make()
    # A generated __eq__ would compare the ndarrays and raise "truth value ... is ambiguous".
    assert a != b
    assert a in [a]
    assert b not in [a]
    assert "best_frame" not in repr(a)
    assert a.open is True and a.event_id is None and a.zone == ""


# ---------------------------------------------------------------------------
# update(): matching, opening, grace, best frame
# ---------------------------------------------------------------------------


def test_first_detection_creates_a_tentative_track() -> None:
    tracker = Tracker()
    update = tracker.update([_det("person", (10, 10, 40, 80), 0.7)], _frame(), 100.0)
    assert update.opened == [] and update.improved == [] and update.closed == []
    assert len(update.active) == 1
    track = update.active[0]
    assert (track.id, track.label, track.frames) == (1, "person", 1)
    assert (track.first_seen, track.last_seen, track.best_ts) == (100.0, 100.0, 100.0)
    assert track.bbox == track.best_bbox == (10, 10, 40, 80)
    assert track.best_confidence == 0.7
    assert track.open is True


def test_track_opens_after_min_frames_consecutive_matches() -> None:
    tracker = Tracker(min_frames=3)
    box = (10, 10, 40, 80)
    assert tracker.update([_det("person", box)], _frame(), 100.0).opened == []
    assert tracker.update([_det("person", box)], _frame(), 100.5).opened == []
    third = tracker.update([_det("person", box)], _frame(), 101.0)
    assert [(t.id, t.frames) for t in third.opened] == [(1, 3)]
    fourth = tracker.update([_det("person", box)], _frame(), 101.5)
    assert fourth.opened == []  # opened once only
    assert fourth.active[0].frames == 4


def test_tentative_track_that_misses_once_is_discarded_silently() -> None:
    tracker = Tracker(min_frames=2)
    box = (10, 10, 40, 80)
    tentative = tracker.update([_det("person", box)], _frame(), 100.0).active[0]
    miss = tracker.update([], _frame(), 100.5)
    assert miss.closed == [] and miss.active == []
    assert tentative.open is False
    again = tracker.update([_det("person", box)], _frame(), 101.0)
    assert again.opened == []
    assert [t.id for t in again.active] == [2]  # a new tentative track, not the old one
    opened = tracker.update([_det("person", box)], _frame(), 101.5)
    assert [t.id for t in opened.opened] == [2]


def test_min_frames_one_opens_on_the_first_detection() -> None:
    tracker = Tracker(min_frames=1)
    update = tracker.update([_det("car", (0, 0, 50, 30))], _frame(), 5.0)
    assert [t.id for t in update.opened] == [1]
    assert update.improved == []


def test_moving_box_keeps_its_track() -> None:
    tracker = Tracker(min_frames=2)
    ids: set[int] = set()
    for i in range(10):
        box = (20 * i, 0, 100, 200)  # 20 px per update: IoU 2/3 with the previous box
        update = tracker.update([_det("person", box)], _frame(), 100.0 + 0.5 * i)
        ids.update(t.id for t in update.active)
    assert ids == {1}
    assert update.active[0].bbox == (180, 0, 100, 200)
    assert update.active[0].frames == 10


def test_box_jump_below_the_threshold_starts_a_new_track() -> None:
    tracker = Tracker(min_frames=1)
    tracker.update([_det("person", (0, 0, 100, 100))], _frame(), 1.0)
    update = tracker.update([_det("person", (60, 0, 100, 100))], _frame(), 1.5)  # IoU 0.25
    assert [t.id for t in update.opened] == [2]
    assert [t.id for t in update.active] == [1, 2]  # track 1 now waits out its grace


def test_labels_are_tracked_separately() -> None:
    tracker = Tracker(min_frames=1)
    box = (10, 10, 40, 80)
    first = tracker.update([_det("person", box)], _frame(), 1.0)
    second = tracker.update([_det("car", box)], _frame(), 1.5)
    assert [t.id for t in first.opened] == [1]
    assert [(t.id, t.label) for t in second.opened] == [(2, "car")]
    person = first.opened[0]
    assert person.frames == 1 and person.last_seen == 1.0  # the car box did not match it


def test_detections_without_a_usable_bbox_are_ignored() -> None:
    tracker = Tracker(min_frames=1)
    detections = [
        Detection(kind="person", confidence=0.9, bbox=None),
        _det("person", (0, 0, 0, 50)),
        _det("person", (0, 0, 50, -1)),
    ]
    update = tracker.update(detections, _frame(), 1.0)
    assert update.opened == [] and update.active == []


def test_opened_track_survives_a_miss_inside_its_grace() -> None:
    tracker = Tracker(min_frames=1, grace_seconds=3.0)
    tracker.update([_det("person", (10, 10, 40, 80))], _frame(), 100.0)
    assert tracker.update([], _frame(), 101.0).closed == []
    back = tracker.update([_det("person", (12, 10, 40, 80))], _frame(), 102.5)
    assert back.opened == [] and back.closed == []
    assert [(t.id, t.frames, t.last_seen) for t in back.active] == [(1, 2, 102.5)]


def test_opened_track_closes_when_its_grace_runs_out() -> None:
    tracker = Tracker(min_frames=1, grace_seconds=3.0)
    tracker.update([_det("person", (10, 10, 40, 80))], _frame(), 100.0)
    for ts in (101.0, 102.0, 102.5):
        update = tracker.update([], _frame(), ts)
        assert update.closed == []
        assert [t.id for t in update.active] == [1]
    update = tracker.update([], _frame(), 103.0)  # missed for exactly grace_seconds
    assert [t.id for t in update.closed] == [1]
    assert update.closed[0].open is False
    assert update.closed[0].last_seen == 100.0
    assert update.active == []
    assert tracker.update([], _frame(), 104.0).closed == []  # never closed twice


def test_detection_after_the_grace_starts_a_new_track() -> None:
    tracker = Tracker(min_frames=1, grace_seconds=3.0)
    box = (10, 10, 40, 80)
    tracker.update([_det("person", box)], _frame(), 100.0)
    update = tracker.update([_det("person", box)], _frame(), 103.0)
    assert [t.id for t in update.closed] == [1]
    assert [t.id for t in update.opened] == [2]


def test_best_frame_is_a_private_copy_of_the_best_frame() -> None:
    tracker = Tracker(min_frames=1)
    frames = {10: _frame(10), 20: _frame(20), 30: _frame(30)}
    tracker.update([_det("person", (10, 10, 40, 80), 0.5)], frames[10], 1.0)
    tracker.update([_det("person", (12, 10, 40, 80), 0.9)], frames[20], 1.5)
    update = tracker.update([_det("person", (14, 10, 40, 80), 0.7)], frames[30], 2.0)
    track = update.active[0]
    assert track.best_confidence == 0.9
    assert track.best_bbox == (12, 10, 40, 80)
    assert track.best_ts == 1.5
    assert track.bbox == (14, 10, 40, 80)  # the current box follows every match
    assert track.best_frame is not None
    assert track.best_frame is not frames[20]
    assert int(track.best_frame[0, 0, 0]) == 20
    frames[20][:] = 99  # the caller reuses its buffer
    assert int(track.best_frame[0, 0, 0]) == 20


def test_improved_lists_only_rises_after_the_track_opened() -> None:
    tracker = Tracker(min_frames=2)
    box = (10, 10, 40, 80)
    tracker.update([_det("person", box, 0.5)], _frame(), 100.0)
    opened = tracker.update([_det("person", box, 0.6)], _frame(), 100.5)
    assert [t.best_confidence for t in opened.opened] == [0.6]  # best of the tentative frames
    assert opened.improved == []  # a rise on the opening update is reported as opened only
    assert len(tracker.update([_det("person", box, 0.9)], _frame(), 101.0).improved) == 1
    assert tracker.update([_det("person", box, 0.8)], _frame(), 101.5).improved == []
    assert tracker.update([_det("person", box, 0.9)], _frame(), 102.0).improved == []  # equal
    assert len(tracker.update([_det("person", box, 0.95)], _frame(), 102.5).improved) == 1


def test_update_without_a_frame_keeps_best_frame_none() -> None:
    tracker = Tracker(min_frames=1)
    update = tracker.update([_det("person", (10, 10, 40, 80), 0.5)], None, 1.0)
    assert update.opened[0].best_frame is None
    later = tracker.update([_det("person", (10, 10, 40, 80), 0.9)], None, 1.5)
    assert later.improved[0].best_frame is None
    assert later.improved[0].best_confidence == 0.9


def test_ids_count_up_from_id_start_and_are_never_reused() -> None:
    tracker = Tracker(min_frames=1, grace_seconds=1.0, id_start=10)
    first = tracker.update(
        [_det("person", (0, 0, 40, 80)), _det("car", (100, 0, 50, 30))], _frame(), 1.0
    )
    assert [(t.id, t.label) for t in first.opened] == [(10, "person"), (11, "car")]
    later = tracker.update([_det("person", (0, 0, 40, 80))], _frame(), 3.0)  # both expired
    assert [t.id for t in later.closed] == [10, 11]
    assert [t.id for t in later.opened] == [12]


def test_greedy_matching_takes_the_highest_iou_first() -> None:
    # Track 1 overlaps D1 (IoU 0.379) more than D2 (0.333), but track 2 overlaps D1 far more
    # (0.905). Matching in track order would hand D1 to track 1 and leave track 2 unmatched.
    tracker = Tracker(min_frames=1)
    tracker.update(
        [_det("person", (100, 0, 100, 100)), _det("person", (150, 0, 100, 100))], _frame(), 1.0
    )
    d1, d2 = (145, 0, 100, 100), (50, 0, 100, 100)
    update = tracker.update([_det("person", d1), _det("person", d2)], _frame(), 1.5)
    assert update.opened == []
    assert [(t.id, t.bbox) for t in update.active] == [(1, d2), (2, d1)]


def test_update_active_lists_live_tracks_by_id_including_grace_and_tentative() -> None:
    tracker = Tracker(min_frames=2)
    a, b = (0, 0, 40, 80), (200, 0, 40, 80)
    tracker.update([_det("person", a), _det("person", b)], _frame(), 1.0)
    tracker.update([_det("person", a), _det("person", b)], _frame(), 1.5)  # both opened
    update = tracker.update([_det("person", a), _det("car", (100, 100, 30, 30))], _frame(), 2.0)
    assert [(t.id, t.label, t.frames) for t in update.active] == [
        (1, "person", 3),  # matched
        (2, "person", 2),  # missed, inside its grace
        (3, "car", 1),  # new, tentative
    ]


def test_wall_clock_stepping_back_is_clamped() -> None:
    tracker = Tracker(min_frames=1)
    box = (10, 10, 40, 80)
    tracker.update([_det("person", box)], _frame(), 100.0)
    track = tracker.update([_det("person", box)], _frame(), 99.0).active[0]
    assert track.id == 1
    assert track.first_seen == 100.0 and track.last_seen == 100.0


# ---------------------------------------------------------------------------
# Review focus: one event per object visit (grace and IoU)
# ---------------------------------------------------------------------------


def test_two_overlapping_people_stay_two_tracks() -> None:
    # Two people walk side by side; their boxes overlap with IoU 0.43, above the 0.3 threshold,
    # so anything that merged overlapping boxes would turn them into one event.
    assert iou((100, 100, 100, 200), (140, 100, 100, 200)) > 0.3
    tracker = Tracker(min_frames=2)
    opened: list[int] = []
    for i in range(20):
        a = (100 + 10 * i, 100, 100, 200)
        b = (140 + 10 * i, 100, 100, 200)
        detections = [_det("person", b, 0.8), _det("person", a, 0.9)]  # B listed first
        update = tracker.update(detections, _frame(), 100.0 + 0.5 * i)
        opened.extend(t.id for t in update.opened)
        assert len(update.active) == 2
    assert sorted(opened) == [1, 2]
    by_id = {t.id: t for t in update.active}
    # Track 1 was created from B, track 2 from A; each kept following its own person.
    assert by_id[1].bbox == (140 + 190, 100, 100, 200)
    assert by_id[2].bbox == (100 + 190, 100, 100, 200)
    assert by_id[1].frames == by_id[2].frames == 20


def test_person_standing_still_for_ten_minutes_is_one_track() -> None:
    # Two detections per second for 600 s. The box jitters by a few pixels, and every 25 s the
    # detector misses the person on four updates in a row (2.5 s between matches; grace is 3 s).
    tracker = Tracker(min_frames=2, grace_seconds=3.0)
    frame = _frame()
    opened: list[int] = []
    closed: list[int] = []
    ts = 0.0
    for i in range(1200):
        ts = 1000.0 + 0.5 * i
        if i % 50 in (10, 11, 12, 13):
            detections = []
        else:
            box = (300 + i % 7 - 3, 200 + i % 5 - 2, 80 + i % 3, 200)
            detections = [_det("person", box, 0.6 + 0.03 * (i % 11))]
        update = tracker.update(detections, frame, ts)
        opened.extend(t.id for t in update.opened)
        closed.extend(t.id for t in update.closed)
    assert opened == [1]
    assert closed == []
    track = update.active[0]
    assert track.frames == 1200 - 24 * 4  # matched on every update that had a detection
    # The person walks away: the track closes once, grace_seconds after it was last seen.
    for k in range(1, 7):
        update = tracker.update([], frame, ts + 0.5 * k)
        closed.extend(t.id for t in update.closed)
    assert closed == [1]
    assert track.open is False
    assert track.last_seen == ts
    assert update.active == []


# ---------------------------------------------------------------------------
# active_boxes(), close_all(), reset()
# ---------------------------------------------------------------------------


def test_active_boxes_lists_only_tracks_seen_on_the_latest_update() -> None:
    tracker = Tracker(min_frames=2)
    a, b = (0, 0, 40, 80), (200, 0, 40, 80)
    assert tracker.active_boxes() == []
    tracker.update([_det("person", a, 0.9), _det("person", b, 0.8)], _frame(), 1.0)
    assert tracker.active_boxes() == [("person", a, 0.9), ("person", b, 0.8)]  # tentative too
    tracker.update([_det("person", a, 0.9), _det("person", b, 0.8)], _frame(), 1.5)
    moved, dog = (4, 0, 40, 80), (100, 50, 20, 20)
    tracker.update([_det("person", moved, 0.6), _det("dog", dog, 0.4)], _frame(), 2.0)
    # Track 2 is inside its grace period: no box. Track 1 shows this update's confidence.
    assert tracker.active_boxes() == [("person", moved, 0.6), ("dog", dog, 0.4)]
    tracker.update([], _frame(), 2.5)
    assert tracker.active_boxes() == []


def test_active_boxes_and_tracks_hold_plain_python_numbers() -> None:
    tracker = Tracker(min_frames=1)
    box = tuple(np.int64(v) for v in (10, 10, 40, 80))
    det = Detection(kind="person", confidence=np.float32(0.75), bbox=box)
    track = tracker.update([det], _frame(), np.float64(5.0)).opened[0]
    assert all(type(v) is int for v in track.bbox + track.best_bbox)
    assert type(track.best_confidence) is float
    assert type(track.first_seen) is float and type(track.last_seen) is float
    _label, bbox, confidence = tracker.active_boxes()[0]
    assert all(type(v) is int for v in bbox)
    assert type(confidence) is float
    json.dumps(tracker.active_boxes())  # JSON-safe for status() and the live preview


def test_close_all_returns_opened_tracks_once_and_drops_tentative_ones() -> None:
    tracker = Tracker(min_frames=2)
    tracker.update([_det("person", (0, 0, 40, 80))], _frame(), 1.0)
    update = tracker.update(
        [_det("person", (0, 0, 40, 80)), _det("car", (200, 0, 40, 80))], _frame(), 1.5
    )
    opened, tentative = update.active  # track 1 opened, track 2 tentative
    assert [t.id for t in tracker.close_all(2.0)] == [1]
    assert opened.open is False and tentative.open is False
    assert opened.last_seen == 1.5  # close_all keeps last_seen as the end time
    assert tracker.active_boxes() == []
    assert tracker.close_all(3.0) == []
    after = tracker.update([], _frame(), 4.0)
    assert after.closed == [] and after.active == []


def test_reset_discards_tracks_without_reporting_and_ids_keep_counting() -> None:
    tracker = Tracker(min_frames=1)
    track = tracker.update([_det("person", (0, 0, 40, 80))], _frame(), 100.0).opened[0]
    tracker.reset()
    assert track.open is False
    assert tracker.active_boxes() == []
    update = tracker.update([_det("person", (0, 0, 40, 80))], _frame(), 50.0)
    assert update.closed == []
    # Ids keep counting; the clock base is reset too, so 50.0 is not clamped to 100.0.
    assert [(t.id, t.first_seen) for t in update.opened] == [(2, 50.0)]


def test_every_opened_track_is_closed_exactly_once() -> None:
    rng = np.random.default_rng(1234)
    tracker = Tracker(min_frames=2, grace_seconds=1.5)
    opened: list[int] = []
    closed: list[int] = []
    for i in range(400):
        detections = [
            _det(
                str(rng.choice(["person", "car"])),
                (int(rng.integers(0, 6)) * 40, 0, 60, 60),
                float(rng.random()),
            )
            for _ in range(int(rng.integers(0, 4)))
        ]
        update = tracker.update(detections, _frame(), 10.0 + 0.5 * i)
        opened.extend(t.id for t in update.opened)
        closed.extend(t.id for t in update.closed)
    closed.extend(t.id for t in tracker.close_all(10.0 + 0.5 * 400))
    assert len(opened) == len(set(opened))
    assert len(opened) > 10
    assert sorted(closed) == sorted(opened)

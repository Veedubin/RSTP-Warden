"""RuleEngine: predicates, cooldown keying, between windows (spec 8.2)."""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from rtsp_warden.actions.rules import RuleDecision, RuleEngine, RuleMatch, in_window
from rtsp_warden.config import RuleConfig
from rtsp_warden.detectors.event_builder import EventInfo

NOON_UTC = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc).timestamp()


class Clock:
    """Injected ``now``: Unix seconds, moved by hand."""

    def __init__(self, t: float = NOON_UTC) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _event(
    label: str = "person", zone: str = "", confidence: float = 0.9, camera: str = "yard"
) -> EventInfo:
    return EventInfo(
        id=1,
        camera=camera,
        label=label,
        confidence=confidence,
        zone=zone,
        started_at=datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc),
        ended_at=None,
        thumbnail_path=None,
        clip_path=None,
        track_id=7,
        event_type="detection",
    )


def _rule(name: str = "r", **fields: Any) -> RuleConfig:
    fields.setdefault("actions", ["phone"])
    return RuleConfig(name=name, **fields)


def _engine(*rules: RuleConfig, clock: Clock | None = None, tz: Any = timezone.utc) -> RuleEngine:
    return RuleEngine("yard", rules, now=clock or Clock(), local_tz=tz)


# --- predicates --------------------------------------------------------------


def test_no_rules_gives_an_empty_decision() -> None:
    assert _engine().evaluate(_event()) == RuleDecision(matched=[], suppressed=[], reasons={})


def test_match_carries_actions_and_clip_flag() -> None:
    rule = _rule(labels=["person"], actions=["phone", "ha"], clip=True)
    decision = _engine(rule).evaluate(_event())
    assert decision.matched == [RuleMatch(rule=rule, actions=["phone", "ha"], clip=True)]
    assert decision.suppressed == []
    assert decision.reasons == {}


def test_empty_labels_match_any_label() -> None:
    decision = _engine(_rule(labels=[])).evaluate(_event(label="motion"))
    assert [m.rule.name for m in decision.matched] == ["r"]


def test_label_mismatch_is_explained() -> None:
    decision = _engine(_rule(labels=["car", "truck"])).evaluate(_event(label="person"))
    assert decision.matched == []
    assert decision.reasons == {"r": "label 'person' is not one of: car, truck"}


def test_zone_predicate() -> None:
    engine = _engine(_rule(zones=["driveway"]))
    assert engine.evaluate(_event(zone="driveway")).matched
    other = engine.evaluate(_event(zone="porch"))
    assert other.reasons["r"] == "zone 'porch' is not one of: driveway"
    none = engine.evaluate(_event(zone=""))
    assert none.reasons["r"] == "event is in no area zone; rule needs one of: driveway"


def test_empty_zones_match_an_event_outside_every_zone() -> None:
    assert _engine(_rule(zones=[])).evaluate(_event(zone="")).matched


def test_min_confidence_is_inclusive() -> None:
    engine = _engine(_rule(min_confidence=0.6, cooldown_seconds=0))
    assert engine.evaluate(_event(confidence=0.6)).matched
    low = engine.evaluate(_event(confidence=0.59))
    assert low.matched == []
    assert low.reasons["r"] == "confidence 0.59 is below 0.60"


def test_rules_are_evaluated_independently_in_config_order() -> None:
    engine = _engine(
        _rule("people", labels=["person"]),
        _rule("cars", labels=["car"]),
        _rule("anything"),
    )
    decision = engine.evaluate(_event(label="person"))
    assert [m.rule.name for m in decision.matched] == ["people", "anything"]
    assert list(decision.reasons) == ["cars"]


# --- cooldown ----------------------------------------------------------------


def test_cooldown_suppresses_then_expires() -> None:
    clock = Clock()
    engine = _engine(_rule(cooldown_seconds=60), clock=clock)
    assert engine.evaluate(_event()).matched

    clock.t += 59
    second = engine.evaluate(_event())
    assert second.matched == []
    assert second.suppressed == ["r"]
    assert second.reasons["r"] == "cooldown: fired 59s ago, cooldown is 60s"

    clock.t += 1
    assert engine.evaluate(_event()).matched


def test_suppressed_event_does_not_extend_the_cooldown() -> None:
    clock = Clock()
    engine = _engine(_rule(cooldown_seconds=60), clock=clock)
    engine.evaluate(_event())
    clock.t += 30
    assert engine.evaluate(_event()).suppressed == ["r"]
    clock.t += 30  # 60 s after the first fire, 30 s after the suppressed one
    assert engine.evaluate(_event()).matched


def test_cooldown_is_keyed_by_rule_and_label() -> None:
    clock = Clock()
    engine = _engine(_rule("a"), _rule("b"), clock=clock)
    first = engine.evaluate(_event(label="person"))
    assert [m.rule.name for m in first.matched] == ["a", "b"]

    clock.t += 5
    assert engine.evaluate(_event(label="person")).suppressed == ["a", "b"]
    dog = engine.evaluate(_event(label="dog"))
    assert [m.rule.name for m in dog.matched] == ["a", "b"]


def test_cooldown_is_keyed_by_camera() -> None:
    clock = Clock()
    rules = [_rule()]
    yard = RuleEngine("yard", rules, now=clock, local_tz=timezone.utc)
    porch = RuleEngine("porch", rules, now=clock, local_tz=timezone.utc)
    assert yard.evaluate(_event(camera="yard")).matched
    assert porch.evaluate(_event(camera="porch")).matched


def test_zero_cooldown_never_suppresses() -> None:
    engine = _engine(_rule(cooldown_seconds=0))
    assert engine.evaluate(_event()).matched
    assert engine.evaluate(_event()).matched


def test_clock_going_backwards_does_not_suppress() -> None:
    clock = Clock()
    engine = _engine(_rule(cooldown_seconds=60), clock=clock)
    engine.evaluate(_event())
    clock.t -= 3600  # NTP stepped the wall clock back an hour
    assert engine.evaluate(_event()).matched


def test_bypass_cooldown_ignores_and_does_not_record_stamps() -> None:
    clock = Clock()
    engine = _engine(_rule(cooldown_seconds=60), clock=clock)
    assert engine.evaluate(_event(), bypass_cooldown=True).matched
    assert engine.evaluate(_event(), bypass_cooldown=True).matched
    # No stamp was recorded, so a real event right after still fires ...
    assert engine.evaluate(_event()).matched
    # ... and a test event inside that real cooldown still fires too.
    clock.t += 1
    assert engine.evaluate(_event(), bypass_cooldown=True).matched
    assert engine.evaluate(_event()).suppressed == ["r"]


def test_reset_cooldowns() -> None:
    engine = _engine(_rule())
    engine.evaluate(_event())
    engine.reset_cooldowns()
    assert engine.evaluate(_event()).matched


# --- between -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "local", "inside"),
    [
        ("22:00-06:00", time(23, 30), True),
        ("22:00-06:00", time(5, 30), True),
        ("22:00-06:00", time(12, 0), False),
        ("22:00-06:00", time(22, 0), True),  # start is inclusive
        ("22:00-06:00", time(6, 0), False),  # end is exclusive
        ("22:00-06:00", time(5, 59, 59), True),
        ("08:00-17:30", time(8, 0), True),
        ("08:00-17:30", time(17, 29), True),
        ("08:00-17:30", time(17, 30), False),
        ("08:00-17:30", time(3, 0), False),
        ("7:00-9:00", time(8, 0), True),  # one-digit hour accepted by parse_between
    ],
)
def test_in_window(spec: str, local: time, inside: bool) -> None:
    assert in_window(spec, local) is inside


@pytest.mark.parametrize("spec", ["", "22:00", "22-06", "25:00-06:00", "07:00-07:00"])
def test_in_window_rejects_what_parse_between_rejects(spec: str) -> None:
    with pytest.raises(ValueError, match="between"):
        in_window(spec, time(12, 0))


def test_between_uses_now_in_the_local_zone() -> None:
    clock = Clock(datetime(2026, 10, 2, 23, 30, tzinfo=timezone.utc).timestamp())
    engine = _engine(_rule(between="22:00-06:00", cooldown_seconds=0), clock=clock)
    assert engine.evaluate(_event()).matched

    clock.t = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc).timestamp()
    noon = engine.evaluate(_event())
    assert noon.matched == []
    assert noon.reasons["r"] == "local time 12:00 is outside 22:00-06:00"


def _new_york() -> ZoneInfo:
    try:
        return ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        pytest.skip("no IANA time zone database on this machine")


def test_between_night_window_in_a_real_zone() -> None:
    """(review focus) 23:30 and 05:30 local match "22:00-06:00"; 12:00 local does not."""
    ny = _new_york()
    clock = Clock()
    engine = _engine(_rule(between="22:00-06:00", cooldown_seconds=0), clock=clock, tz=ny)
    for local, expected in [
        (datetime(2026, 10, 1, 23, 30, tzinfo=ny), True),
        (datetime(2026, 10, 2, 5, 30, tzinfo=ny), True),
        (datetime(2026, 10, 2, 12, 0, tzinfo=ny), False),
    ]:
        clock.t = local.timestamp()
        assert bool(engine.evaluate(_event()).matched) is expected, local


def test_between_across_dst_transitions_never_raises() -> None:
    """(review focus) Every minute of both 2026 US DST change days evaluates cleanly."""
    ny = _new_york()
    clock = Clock()
    engine = _engine(
        _rule("gap", between="02:00-03:00", cooldown_seconds=0),
        _rule("fold", between="01:00-02:00", cooldown_seconds=0),
        clock=clock,
        tz=ny,
    )
    change_days = [
        datetime(2026, 3, 8, tzinfo=timezone.utc),
        datetime(2026, 11, 1, tzinfo=timezone.utc),
    ]
    for day in change_days:
        start = day.timestamp()
        for minute in range(0, 24 * 60):
            clock.t = start + minute * 60
            engine.evaluate(_event())

    # 2026-03-08 02:00-03:00 local does not exist: 06:59:59Z is 01:59:59 EST, 07:00Z is 03:00 EDT.
    clock.t = datetime(2026, 3, 8, 6, 59, 59, tzinfo=timezone.utc).timestamp()
    assert [m.rule.name for m in engine.evaluate(_event()).matched] == ["fold"]
    clock.t = datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc).timestamp()
    assert engine.evaluate(_event()).matched == []
    # 2026-11-01 01:30 local happens twice (EDT, then EST); both are inside "01:00-02:00".
    for utc_hour in (5, 6):
        clock.t = datetime(2026, 11, 1, utc_hour, 30, tzinfo=timezone.utc).timestamp()
        assert [m.rule.name for m in engine.evaluate(_event()).matched] == ["fold"]

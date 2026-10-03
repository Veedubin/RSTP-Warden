"""Rule engine (spec 8.2): which rules an event matches, and which actions fire.

Pure apart from the cooldown map, which is keyed by (camera, rule name, label)
and guarded by a lock because the detector worker and the web test route both
call ``evaluate``.
"""

from __future__ import annotations

import threading
import time as _clock
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, tzinfo
from typing import TYPE_CHECKING

from ..config import parse_between

if TYPE_CHECKING:
    from ..config import RuleConfig
    from ..detectors.event_builder import EventInfo


@dataclass(slots=True)
class RuleMatch:
    """One rule that fired: the actions to run and whether to cut a clip on close."""

    rule: RuleConfig
    actions: list[str]
    clip: bool


@dataclass(slots=True)
class RuleDecision:
    """Result of one evaluation.

    ``suppressed`` names rules that matched but were inside their cooldown.
    ``reasons`` maps every rule that did not fire to a short explanation.
    """

    matched: list[RuleMatch] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


def _inside(start: time, end: time, local_time: time) -> bool:
    t = local_time.replace(tzinfo=None, fold=0)
    if start < end:
        return start <= t < end
    return t >= start or t < end  # wraps midnight (parse_between rejects start == end)


def in_window(spec: str, local_time: time) -> bool:
    """True when ``local_time`` is in ``spec``; start inclusive, end exclusive, may wrap.

    ``spec`` is parsed by ``config.parse_between`` (the same parser that validates
    ``rules[].between``), which raises ``ValueError`` for a malformed window.
    """
    start, end = parse_between(spec)
    return _inside(start, end, local_time)


class RuleEngine:
    """Evaluate one camera's rules against an event.

    ``now`` returns Unix seconds (injected in tests). ``between`` windows are
    checked against ``now`` converted to ``local_tz`` (``None`` = the system
    local zone).
    """

    def __init__(
        self,
        camera: str,
        rules: Sequence[RuleConfig],
        *,
        now: Callable[[], float] = _clock.time,
        local_tz: tzinfo | None = None,
    ) -> None:
        self.camera = camera
        self._rules = list(rules)
        self._now = now
        self._local_tz = local_tz
        self._windows: dict[int, tuple[time, time]] = {
            index: parse_between(rule.between)
            for index, rule in enumerate(self._rules)
            if rule.between
        }
        self._last_fired: dict[tuple[str, str, str], float] = {}
        self._lock = threading.Lock()

    @property
    def rules(self) -> list[RuleConfig]:
        return list(self._rules)

    def evaluate(self, event: EventInfo, *, bypass_cooldown: bool = False) -> RuleDecision:
        """Match ``event`` against every rule in config order.

        A rule that matches records its cooldown stamp unless ``bypass_cooldown``
        is set (the UI test event), which also ignores existing stamps.
        """
        now = self._now()
        local_time = datetime.fromtimestamp(now, tz=self._local_tz).time()
        decision = RuleDecision()
        with self._lock:
            for index, rule in enumerate(self._rules):
                reason = self._mismatch(index, rule, event, local_time)
                if reason is not None:
                    decision.reasons[rule.name] = reason
                    continue
                if not bypass_cooldown:
                    key = (self.camera, rule.name, event.label)
                    last = self._last_fired.get(key)
                    if last is not None and 0.0 <= now - last < rule.cooldown_seconds:
                        decision.suppressed.append(rule.name)
                        decision.reasons[rule.name] = (
                            f"cooldown: fired {now - last:.0f}s ago, "
                            f"cooldown is {rule.cooldown_seconds:g}s"
                        )
                        continue
                    self._last_fired[key] = now
                decision.matched.append(
                    RuleMatch(rule=rule, actions=list(rule.actions), clip=rule.clip)
                )
        return decision

    def reset_cooldowns(self) -> None:
        """Forget every cooldown stamp."""
        with self._lock:
            self._last_fired.clear()

    def _mismatch(
        self, index: int, rule: RuleConfig, event: EventInfo, local_time: time
    ) -> str | None:
        if rule.labels and event.label not in rule.labels:
            return f"label {event.label!r} is not one of: {', '.join(rule.labels)}"
        if rule.zones and event.zone not in rule.zones:
            if not event.zone:
                return f"event is in no area zone; rule needs one of: {', '.join(rule.zones)}"
            return f"zone {event.zone!r} is not one of: {', '.join(rule.zones)}"
        if event.confidence < rule.min_confidence:
            return f"confidence {event.confidence:.2f} is below {rule.min_confidence:.2f}"
        window = self._windows.get(index)
        if window is not None and not _inside(window[0], window[1], local_time):
            return f"local time {local_time:%H:%M} is outside {rule.between}"
        return None

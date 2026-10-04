"""Day / night (IR) detection from the decoded tap frame.

The Foscam, like most IR cameras, flips its IR-cut filter at night and the frame becomes
grayscale: red, green and blue are equal within JPEG chroma noise. ``channel_spread``
measures that (mean of ``max(B,G,R) - min(B,G,R)`` over a subsampled frame; about 0-2 for
an IR frame, tens for daylight). ``DayNight`` turns the measure into a state with
hysteresis so dusk does not flap it. A very dark colour frame also measures low and counts
as night, which is the intended meaning: "IR, or too dark for colour".

Pure NumPy; no detector or runtime imports, so the training tool can copy the formula
(``tools/wildlife/wildlife_data.py`` carries the same function, pinned by a test).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Mean channel spread below which a frame counts as grayscale (night).
NIGHT_SPREAD = 4.0
#: Consecutive frames on the other side before the state flips.
SWITCH_FRAMES = 3
_SUBSAMPLE = 4


def channel_spread(frame_bgr: np.ndarray) -> float:
    """Mean of ``max(B,G,R) - min(B,G,R)`` over every 4th pixel; 0.0 for unusable input."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3 or frame_bgr.size == 0:
        return 0.0
    small = frame_bgr[::_SUBSAMPLE, ::_SUBSAMPLE, :3]
    if small.size == 0:
        return 0.0
    spread = small.max(axis=2).astype(np.int16) - small.min(axis=2).astype(np.int16)
    return float(spread.mean())


@dataclass
class DayNight:
    """Night state of one camera, updated once per decoded frame.

    The first frame sets ``night`` at once. Afterwards the state flips only after
    ``switch_frames`` consecutive frames measured on the other side of ``threshold``;
    a frame back on the current side resets that count. ``since_ts`` is the frame time
    of the last change, ``switches`` counts changes (not the first frame).
    """

    threshold: float = NIGHT_SPREAD
    switch_frames: int = SWITCH_FRAMES
    night: bool | None = None
    since_ts: float | None = None
    switches: int = 0
    last_spread: float = 0.0
    _pending: int = field(default=0, init=False, repr=False)

    def update(self, frame_bgr: np.ndarray, ts_unix: float) -> bool:
        spread = channel_spread(frame_bgr)
        self.last_spread = spread
        observed = spread < self.threshold
        if self.night is None:
            self.night = observed
            self.since_ts = float(ts_unix)
            self._pending = 0
            return observed
        if observed == self.night:
            self._pending = 0
            return self.night
        self._pending += 1
        if self._pending >= max(1, int(self.switch_frames)):
            self.night = observed
            self.since_ts = float(ts_unix)
            self.switches += 1
            self._pending = 0
        return self.night


def allows(when: str, night: bool | None) -> bool:
    """Whether a detector with ``when`` runs in the given state (``None`` counts as day)."""
    if when == "night":
        return night is True
    if when == "day":
        return not night
    return True


__all__ = ["NIGHT_SPREAD", "SWITCH_FRAMES", "DayNight", "allows", "channel_spread"]

"""The slow layer: colour over a scene rather than over a beat.

Kept separate from the effects that consume it because the *source* of a mood is
meant to change. Right now there is one, derived from audio. A picture feed --
Hyperion-style, from what is actually on screen -- is the obvious second, and
for film it is the better signal. Philips' own patent on this blends video and
audio contributions by weight rather than choosing between them, which is why
:class:`Mood` carries a weight and :func:`blend` exists before there is anything
to blend with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ambviz.dsp import AdaptiveRange, ExpFilter, RateLimiter
from ambviz.features import Features
from ambviz.settings import Settings


@dataclass
class Mood:
    """Where the light should sit, independent of how an effect draws it."""

    hue: float = 0.5
    """0-1 around the colour circle."""

    saturation: float = 0.9
    level: float = 0.0
    """Brightness before any floor is applied, 0-1."""

    detail: float = 0.0
    """How far to cross-fade from a plain wash toward a spectral display, 0-1.

    Not "how much sparkle to add": at 1 the effect is essentially a smoothed
    ``bars``. Driven by scene energy and damped by dialogue, so a fight scene
    gets a spectrum and a quiet conversation gets a wash."""

    weight: float = 1.0
    """Confidence in this mood, for blending several sources."""


def blend(moods: list[Mood]) -> Mood:
    """Weighted average of several moods.

    Hue is circular, so it is averaged as unit vectors -- the mean of 0.95 and
    0.05 is 0.0, not 0.5, and a numeric average would send the colour to the
    opposite side of the wheel.
    """
    live = [m for m in moods if m.weight > 0]
    if not live:
        return Mood()
    if len(live) == 1:
        return live[0]

    total = sum(m.weight for m in live)
    angles = np.array([m.hue * 2 * np.pi for m in live])
    weights = np.array([m.weight for m in live])
    hue = float(np.arctan2(np.sum(weights * np.sin(angles)),
                           np.sum(weights * np.cos(angles))) / (2 * np.pi)) % 1.0
    return Mood(
        hue=hue,
        saturation=sum(m.saturation * m.weight for m in live) / total,
        level=sum(m.level * m.weight for m in live) / total,
        detail=sum(m.detail * m.weight for m in live) / total,
        weight=max(m.weight for m in live),
    )


class AudioMood:
    """Derives a mood from the analysis, and moves it slowly.

    The centroid is **smoothed first**, then range-adapted, then rate-limited.
    Both of the first two steps were originally in the other order, and both were
    wrong for the same reason.

    Adapting before smoothing learns how far speech jitters frame to frame --
    449 to 1635 Hz on test material -- when what matters is how far the mood moves
    across a scene, a much smaller range sitting inside it. Rescale against the
    jitter and the slow trend becomes a sliver in the middle, so it averages to
    one colour: precisely the symptom this was built to fix.

    Rate-limiting before smoothing fails differently. A limiter fed a signal that
    reverses several times a second is whipsawed -- it chases up, the target
    reverses, it chases down, net movement nothing. It caps speed; it cannot
    extract a trend.

    Measured on film-like audio, both wrong orders gave less hue movement than
    doing nothing at all.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        cfg = settings.mood
        # Smoothed in centroid space, which is linear. Smoothing hue directly
        # would break at the wrap, where 0.99 and 0.01 average to 0.5.
        alpha = float(np.clip(1.0 / max(cfg.response_seconds * settings.audio.fps, 1.0),
                              1e-4, 0.5))
        self._smooth = ExpFilter(0.0, alpha_decay=alpha, alpha_rise=alpha)
        self._primed = False
        self._last_target = 0.5
        self._range = AdaptiveRange(seconds=cfg.range_seconds, fps=float(settings.audio.fps))
        self._hue = RateLimiter(cfg.hue_rate, value=0.5, deadband=cfg.deadband, wrap=True)
        self._level = RateLimiter(1.0 / max(cfg.response_seconds, 1e-3), value=0.0)
        self._last_t: float | None = None

    def update(self, features: Features) -> Mood:
        cfg = self.settings.mood
        dt = 0.0 if self._last_t is None else max(0.0, features.t - self._last_t)
        self._last_t = features.t

        # Smooth in Hz, learn the range of the smoothed signal, then cap the
        # speed. Each step depends on the one before it.
        # Start the smoother at the first real value. Starting from zero makes
        # it ramp up over tens of seconds, and the range then learns that ramp
        # instead of the content.
        if not self._primed and features.centroid_hz > 0.0:
            self._smooth.value = features.centroid_hz
            self._primed = True
        smoothed = float(self._smooth.update(features.centroid_hz))
        target = self._range.update(smoothed)
        self._last_target = target
        hue = self._hue.update(target, dt)
        level = self._level.update(float(np.clip(features.slow, 0.0, 1.0)), dt)

        # Scene energy decides how energetic to be; dialogue pulls it back. The
        # point is not to be subtle always -- a fight scene should look close to
        # bars, just smoother -- but to be subtle when the content is.
        detail = cfg.detail * features.energy * (1.0 - cfg.dialogue_damping * features.dialogue)

        return Mood(
            hue=hue,
            saturation=0.85,
            level=max(level, 0.0),
            detail=float(np.clip(detail, 0.0, 1.0)),
            weight=cfg.audio_weight,
        )

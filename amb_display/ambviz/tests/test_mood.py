"""The slow layer, and the ordering it depends on."""

import numpy as np
import pytest

from ambviz.dsp import AdaptiveRange, RateLimiter
from ambviz.mood import AudioMood, Mood, blend
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings

SR = 44100


# ── primitives ───────────────────────────────────────────────────────────────
def test_adaptive_range_stretches_a_narrow_input():
    r = AdaptiveRange(seconds=10, fps=60)
    rng = np.random.default_rng(0)
    out = [r.update(rng.uniform(0.40, 0.45)) for _ in range(1200)]
    tail = out[-200:]
    assert min(tail) < 0.2 and max(tail) > 0.8, f"only spanned {min(tail):.2f}-{max(tail):.2f}"


def test_adaptive_range_ignores_a_lone_outlier():
    """Percentiles, not min and max: one transient must not flatten the mapping."""
    r = AdaptiveRange(seconds=10, fps=60)
    for _ in range(600):
        r.update(0.5)
    r.update(1000.0)
    for _ in range(600):
        r.update(0.5)
    assert r.high < 10.0, f"one outlier set the ceiling to {r.high}"


def test_adaptive_range_holds_still_when_nothing_varies():
    """A steady input must read as steady, not have its noise amplified."""
    r = AdaptiveRange(seconds=10, fps=60)
    rng = np.random.default_rng(1)
    out = [r.update(1200.0 + rng.normal(0, 0.2)) for _ in range(1200)]
    assert all(abs(v - 0.5) < 1e-9 for v in out[-100:])


def test_rate_limiter_respects_its_cap():
    lim = RateLimiter(0.1, value=0.0)
    lim.update(1.0, 1.0)
    assert lim.value == pytest.approx(0.1)


def test_rate_limiter_deadband_holds_still():
    lim = RateLimiter(1.0, value=0.5, deadband=0.05)
    lim.update(0.52, 1.0)
    assert lim.value == 0.5


def test_rate_limiter_takes_the_short_way_round():
    """Hue is circular: 0.95 to 0.05 is forward, not most of the way back."""
    lim = RateLimiter(1.0, value=0.95, deadband=0.0, wrap=True)
    lim.update(0.05, 1 / 60)
    assert lim.value > 0.95 or lim.value < 0.05


def test_default_hue_rate_traverses_in_the_intended_band():
    """The subtlety target: a full circle in roughly 5-10 s."""
    rate = Settings.load().mood.hue_rate
    lim = RateLimiter(rate, value=0.0, deadband=0.01)
    steps = 0
    while abs(0.99 - lim.value) > lim.deadband and steps < 200_000:
        lim.update(0.99, 1 / 60)
        steps += 1
    assert 5.0 <= steps / 60 <= 10.0, f"full traverse took {steps / 60:.1f}s"


# ── blending ─────────────────────────────────────────────────────────────────
def test_blend_averages_hue_around_the_circle():
    """0.95 and 0.05 average to 0.0, not to 0.5 on the far side."""
    out = blend([Mood(hue=0.95, weight=1.0), Mood(hue=0.05, weight=1.0)])
    assert min(out.hue, 1 - out.hue) < 0.02


def test_blend_respects_weight():
    out = blend([Mood(level=0.0, weight=3.0), Mood(level=1.0, weight=1.0)])
    assert out.level == pytest.approx(0.25)


def test_blend_ignores_zero_weight_sources():
    out = blend([Mood(hue=0.1, weight=0.0), Mood(hue=0.8, weight=1.0)])
    assert out.hue == pytest.approx(0.8)


# ── the claim that motivated all of this ─────────────────────────────────────
def film_like(v, seconds, scene_period=20.0):
    """Centred speech-band content whose spectrum drifts across a scene.

    Deliberately narrow-range: that is the condition under which a fixed
    mapping collapses to one colour.
    """
    n = v.samples_per_frame
    hues = []
    for i in range(int(seconds * v.settings.audio.fps)):
        t = (np.arange(n) + i * n) / SR
        f0 = 700 + 250 * np.sin(((i * n / SR) / scene_period) * 2 * np.pi)
        speech = np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) * 0.7
        v.process(np.stack([speech, speech * 0.98], axis=1) * 7000)
        hues.append(v.effect.mood.hue)
    return np.array(hues[int(len(hues) * 0.3):])


def circular_sd(hues):
    ang = hues * 2 * np.pi
    r = np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))
    return float(np.sqrt(-2 * np.log(max(r, 1e-12))) / (2 * np.pi))


def cinema():
    return Visualizer(Settings.load(overrides={"effect": {"name": "cinema"}}))


def test_adaptive_range_is_what_stops_colour_sticking():
    """The whole point, measured.

    With a fixed mapping the hue barely moves on this material -- that is the
    reported symptom. Adaptive rescaling should make it use most of the circle.
    """
    fixed = cinema()
    fixed.effect.mood_source._range.update = lambda x: float(np.clip(x / (SR / 2), 0, 1))
    stuck = circular_sd(film_like(fixed, 90))

    moving = circular_sd(film_like(cinema(), 90))

    assert stuck < 0.02, f"the fixed mapping was expected to stick, got sd {stuck:.4f}"
    assert moving > stuck * 10, f"adaptive sd {moving:.4f} vs fixed {stuck:.4f}"


def test_hue_is_smoothed_before_being_rate_limited():
    """Ordering regression.

    Feeding a rate limiter the unsmoothed centroid whipsaws it -- it chases a
    target that reverses several times a second and nets no movement. That
    produced *less* colour movement than no adaptation at all.
    """
    v = cinema()
    assert hasattr(v.effect.mood_source, "_smooth")
    hues = film_like(v, 60)
    assert circular_sd(hues) > 0.05


def test_time_advances_with_the_audio_not_the_clock():
    """Offline processing runs far faster than real time; a wall-clock t would
    make the mood race and results irreproducible."""
    v = cinema()
    n = v.samples_per_frame
    for _ in range(v.settings.audio.fps):
        v.process(np.zeros((n, 2)))
    assert v.features.t == pytest.approx(1.0, abs=0.05)


def test_dialogue_damps_the_fast_layer():
    s = Settings.load()
    mood = AudioMood(s)
    from ambviz.features import Features

    talking = mood.update(Features(mel=np.zeros(24), volume=0.2, dialogue=1.0, t=1.0))
    music = mood.update(Features(mel=np.zeros(24), volume=0.2, dialogue=0.0, t=2.0))
    assert talking.detail < music.detail


def test_brightness_never_reaches_zero():
    """A strip snapping fully dark mid-scene is worse than one that drifts."""
    v = cinema()
    out = v.process(np.zeros((v.samples_per_frame, 2)))
    assert out.max() == 0.0            # true silence still blanks
    for _ in range(30):
        out = v.process(np.full((v.samples_per_frame, 2), 30.0))
    assert out.max() > 0.0, "quiet audio should still show the floor"

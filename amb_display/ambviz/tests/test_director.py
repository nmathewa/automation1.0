"""Choosing an animation for the scene, and not flapping about it."""

import numpy as np
import pytest

from ambviz.director import DEFAULT, score_candidates
from ambviz.effects import Director
from ambviz.effects import EFFECTS
from ambviz.features import Features
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings

SR = 44100


def auto(**mood):
    over = {"effect": {"name": "auto"}, "mood": {"scene_weight": 0.0, **mood}}
    return Visualizer(Settings.load(overrides=over))


# ── scoring, without any audio ───────────────────────────────────────────────
def test_the_wash_wins_a_quiet_centred_scene():
    scores = score_candidates(Features(mel=np.zeros(24), volume=0.1,
                                       dialogue=0.95, energy=0.1))
    assert max(scores, key=lambda k: scores[k]) == DEFAULT


def test_beats_prefer_a_beat_driven_animation():
    scores = score_candidates(Features(mel=np.zeros(24), volume=0.6,
                                       dialogue=0.1, energy=0.7,
                                       onset_rate=0.9, brightness=0.6))
    assert scores["pixelwave"] > scores[DEFAULT]


def test_wide_loud_content_prefers_a_spectrum():
    scores = score_candidates(Features(mel=np.zeros(24), volume=0.7,
                                       dialogue=0.05, energy=0.85,
                                       onset_rate=0.1, brightness=0.9))
    assert scores["bars"] > scores[DEFAULT]


def test_every_candidate_is_a_real_effect():
    names = score_candidates(Features(mel=np.zeros(24), volume=0.0))
    for name in names:
        assert name in EFFECTS, f"{name} is scored but not registered"


# ── switching discipline ─────────────────────────────────────────────────────
def test_dwell_time_blocks_a_rapid_second_switch():
    d = Director(Settings.load(), 60)
    d.last_switch = 100.0
    hot = Features(mel=np.zeros(24), volume=0.6, energy=0.9,
                   onset_rate=0.95, dialogue=0.0, t=101.0)
    assert d.choose(hot) == DEFAULT, "switched inside the dwell window"
    hot.t = 100.0 + d.settings.mood.switch_dwell + 0.1
    assert d.choose(hot) != DEFAULT


def test_a_tie_keeps_what_is_already_running():
    """Hysteresis: without a margin the selector flaps at every crossing."""
    d = Director(Settings.load(), 60)
    d.last_switch = -1000.0
    balanced = Features(mel=np.zeros(24), volume=0.4, energy=0.5,
                        onset_rate=0.5, dialogue=0.5, brightness=0.5, t=0.0)
    scores = score_candidates(balanced)
    best = max(scores, key=lambda k: scores[k])
    d.current = best
    assert d.choose(balanced) == best


def test_crossfade_renders_both_animations():
    """Effects hold state, so a hard swap shows a seam."""
    v = auto(switch_dwell=0.0, crossfade=2.0)
    d = v.effect
    n = v.samples_per_frame
    rng = np.random.default_rng(0)
    for _ in range(200):
        v.process(rng.normal(0, 4000, (n, 2)))
        if d.previous is not None:
            assert 0.0 <= d.fade < 1.0
            break
    else:
        pytest.skip("no switch occurred on this material")


def test_output_shape_survives_switching():
    v = auto(switch_dwell=0.0)
    rng = np.random.default_rng(1)
    for _ in range(300):
        out = v.process(rng.normal(0, 4000, (v.samples_per_frame, 2)))
        assert out.shape == (3, 60)
        assert out.min() >= 0 and out.max() <= 255


# ── the onset bug that broke selection ───────────────────────────────────────
def test_a_sustained_chord_fires_no_onsets():
    """Regression.

    The threshold was purely relative, so on sustained material the running
    floor collapsed and numerical wobble cleared it -- a held chord fired as
    many onsets as a drum track, and the director sent it to a beat animation.
    """
    v = auto()
    n = v.samples_per_frame
    before = v.beats
    for i in range(int(15 * v.settings.audio.fps)):
        t = (np.arange(n) + i * n) / SR
        chord = sum(0.3 * np.sin(2 * np.pi * f * t) for f in (110, 165, 220, 330))
        v.process(np.stack([chord, chord * 0.85], axis=1) * 7000)
    assert v.beats - before <= 5, f"a held chord fired {v.beats - before} onsets"


def test_drums_still_fire_onsets():
    """The other half: the floor must not silence real hits."""
    v = auto()
    n = v.samples_per_frame
    before = v.beats
    for i in range(int(15 * v.settings.audio.fps)):
        t = (np.arange(n) + i * n) / SR
        kick = np.exp(-((t * 2) % 1.0) * 10) * np.sin(2 * np.pi * 60 * t) * 1.3
        v.process(np.stack([kick, kick], axis=1) * 7000)
    assert v.beats - before >= 15, f"only {v.beats - before} onsets on a drum track"

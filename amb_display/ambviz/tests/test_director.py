"""Choosing an animation for the scene, and not flapping about it."""

import numpy as np
import pytest

from ambviz.director import score_candidates
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
def test_a_quiet_centred_scene_picks_the_calmest_option():
    """``spectrum`` again. It briefly became ``pacifica``, but pacifica drifts
    once every 18-42 s under a 2.8 s level filter, so on screen it neither
    moved nor tracked the music and it left the shortlist. Spectrum is the
    calmest option *available*, which is not the same as a calm one -- the
    shortlist has no ambient member until something is tuned to be one."""
    quiet = Features(mel=np.zeros(24), volume=0.1, dialogue=0.95, energy=0.1)
    scores = score_candidates(quiet, Settings.load().mood.animations)
    assert max(scores, key=lambda k: scores[k]) == "spectrum"


def test_beats_prefer_a_pulse_driven_animation():
    beat = Features(mel=np.zeros(24), volume=0.6, dialogue=0.1, energy=0.7,
                    onset_rate=0.9, brightness=0.6)
    scores = score_candidates(beat, Settings.load().mood.animations)
    # puddles is the pulse-driven member now that scroll has left the list.
    assert scores["puddles"] > scores["spectrum"]


def test_wide_loud_content_prefers_a_band_display():
    wide = Features(mel=np.zeros(24), volume=0.7, dialogue=0.05, energy=0.85,
                    onset_rate=0.1, brightness=0.9)
    scores = score_candidates(wide, Settings.load().mood.animations)
    assert scores["bars"] > scores["spectrum"]


def test_every_candidate_is_a_real_effect():
    for name in score_candidates(Features(mel=np.zeros(24), volume=0.0)):
        assert name in EFFECTS, f"{name} is scored but not registered"


def test_the_shortlist_is_honoured():
    """The candidate set is configuration, not a code edit."""
    f = Features(mel=np.zeros(24), volume=0.6, energy=0.8, onset_rate=0.9)
    scores = score_candidates(f, ("bars", "waterfall"))
    assert set(scores) == {"bars", "waterfall"}

    v = auto()
    assert v.effect.allowed == tuple(Settings.load().mood.animations)
    assert "pixelwave" not in v.effect.allowed


def test_an_unknown_animation_is_rejected_at_load():
    with pytest.raises(ValueError, match="unknown effect"):
        Settings.load(overrides={"mood": {"animations": ["bars", "nope"]}})


def test_auto_cannot_contain_itself():
    with pytest.raises(ValueError, match="must not contain"):
        Settings.load(overrides={"mood": {"animations": ["auto"]}})


# ── switching discipline ─────────────────────────────────────────────────────
def test_dwell_time_blocks_a_rapid_second_switch():
    d = Director(Settings.load(), 60)
    d.last_switch = 100.0
    d.current = "spectrum"
    hot = Features(mel=np.zeros(24), volume=0.6, energy=0.9,
                   onset_rate=0.95, dialogue=0.0, brightness=0.9, t=101.0)
    # Five seconds of identical content. Scores are smoothed now, so a
    # favourite has to establish itself over time rather than in one frame --
    # the dwell must hold throughout regardless.
    for i in range(300):
        hot.t = 100.0 + i / 60.0
        assert d.choose(hot) == "spectrum", "switched inside the dwell window"
    hot.t = 100.0 + d.settings.mood.switch_dwell + 0.1
    assert d.choose(hot) != "spectrum"


def test_a_tie_keeps_what_is_already_running():
    """Hysteresis: without a margin the selector flaps at every crossing."""
    d = Director(Settings.load(), 60)
    d.last_switch = -1000.0
    balanced = Features(mel=np.zeros(24), volume=0.4, energy=0.5,
                        onset_rate=0.5, dialogue=0.5, brightness=0.5, t=0.0)
    scores = score_candidates(balanced, d.allowed)
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


def _f(**kw):
    kw.setdefault("mel", np.zeros(24))
    kw.setdefault("volume", 0.2)
    return Features(**kw)


#: A shortlist with one member of each family, as the new effects are meant to
#: be used. Scored against the whole library instead, ``puddles`` and
#: ``pixelwave`` are near-ties -- they are both "light on an onset", and
#: ``puddles`` exists as the replacement for ``pixelwave`` rather than
#: alongside it. Nothing distinguishes them until there is a feature for how
#: *regular* a pulse is.
WIDE = ("bars", "energy", "scroll", "spectrum", "waterfall",
        "pacifica", "puddles", "freqwave", "fire")


def test_each_new_effect_wins_the_scene_it_was_built_for():
    """A candidate nobody ever picks is dead weight in the shortlist."""
    cases = {
        # calm, no pulse -> the ambient swell
        "pacifica": _f(energy=0.15, dialogue=0.9, onset_rate=0.0, brightness=0.1),
        # a strong rhythm -> the sparse hits
        "puddles": _f(energy=0.6, onset_rate=1.0, dialogue=0.0, brightness=0.3),
        # loud and dark -> fire
        "fire": _f(energy=1.0, onset_rate=0.2, dialogue=0.0, brightness=0.0),
    }
    for want, features in cases.items():
        scores = score_candidates(features, WIDE)
        assert max(scores, key=scores.get) == want, (want, sorted(
            scores.items(), key=lambda kv: -kv[1])[:3])


def test_puddles_and_pixelwave_overlap():
    """Documents the tie above, so a future feature that separates them fails
    this test loudly rather than silently changing behaviour."""
    f = _f(energy=0.6, onset_rate=1.0)
    scores = score_candidates(f)
    assert abs(scores["puddles"] - scores["pixelwave"]) < 0.1


def test_new_effects_are_scored_at_all():
    scores = score_candidates(_f(energy=0.5, onset_rate=0.5))
    for name in ("pacifica", "puddles", "freqwave", "fire"):
        assert name in scores, name

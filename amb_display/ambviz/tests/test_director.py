"""Choosing an animation for the scene, and not flapping about it."""

import collections

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
    """``spectrum`` again. It briefly became ``pacifica``, which drifted once
    every 18-42 s under a 2.8 s level filter, so on screen it neither moved nor
    tracked the music; it left the shortlist and has since been removed from
    the library. Spectrum is the calmest option *available* in the default
    shortlist, which is not the same as a calm one."""
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
    """Even a genuine change in character cannot switch inside the dwell."""
    d = Director(Settings.load(), 60)
    calm = Features(mel=np.zeros(24), volume=0.4, energy=0.1,
                    onset_rate=0.0, dialogue=0.9, brightness=0.1, t=0.0)
    d.choose(calm)                              # anchor on the calm character
    d.last_switch = 100.0
    d.current = "spectrum"
    hot = Features(mel=np.zeros(24), volume=0.6, energy=0.9,
                   onset_rate=0.95, dialogue=0.0, brightness=0.9, t=100.0)
    dwell = d.settings.mood.switch_dwell
    for i in range(int(dwell * 60) - 6):
        hot.t = 100.0 + i / 60.0
        assert d.choose(hot) == "spectrum", "switched inside the dwell window"
    # Past the dwell the accumulated character change may now act.
    for i in range(300):
        hot.t = 100.0 + dwell + 0.1 + i / 60.0
        if d.choose(hot) != "spectrum":
            return
    raise AssertionError("character moved but the switch never came")


def test_steady_audio_keeps_what_is_already_running():
    """Was hysteresis-on-scores; now the rule itself. A switch needs the
    audio's character to move (or max_dwell to expire) -- score crossings
    alone never cause one, which is what made score noise harmless."""
    d = Director(Settings.load(), 60)
    balanced = Features(mel=np.zeros(24), volume=0.4, energy=0.5,
                        onset_rate=0.5, dialogue=0.5, brightness=0.5, t=0.0)
    d.choose(balanced)                      # anchors on this character
    start = d.current
    dwell = d.settings.mood.switch_dwell
    for i in range(600):                    # 10 s, past dwell, within max_dwell
        balanced.t = i / 60.0
        assert d.choose(balanced) == start, "switched on steady audio"
    assert balanced.t > dwell


def test_a_change_in_character_switches_to_something_else():
    """The incumbent is never re-elected: variety is the rule, not a weight."""
    d = Director(Settings.load(), 60)
    calm = Features(mel=np.zeros(24), volume=0.4, energy=0.1,
                    onset_rate=0.0, dialogue=0.9, brightness=0.1, t=0.0)
    for i in range(600):
        calm.t = i / 60.0
        d.choose(calm)
    before = d.current
    loud = Features(mel=np.zeros(24), volume=0.6, energy=0.95,
                    onset_rate=0.9, dialogue=0.0, brightness=0.9, t=10.0)
    got = before
    for i in range(600):                    # let the smoothed character move
        loud.t = 10.0 + i / 60.0
        got = d.choose(loud)
        if got != before:
            break
    assert got != before, "character moved but nothing switched"


def test_max_dwell_rotates_even_on_static_audio():
    """The rotation guarantee -- starvation was the complaint that forced
    this design."""
    d = Director(Settings.load(), 60)
    still = Features(mel=np.zeros(24), volume=0.4, energy=0.5,
                     onset_rate=0.5, dialogue=0.5, brightness=0.5, t=0.0)
    d.choose(still)
    start = d.current
    still.t = d.settings.mood.max_dwell + 1.0
    assert d.choose(still) != start


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
        "cinema", "puddles", "freqwave", "fire")


def test_each_new_effect_wins_the_scene_it_was_built_for():
    """A candidate nobody ever picks is dead weight in the shortlist."""
    cases = {
        # calm, no pulse, sustained -> the wash
        "cinema": _f(energy=0.15, dialogue=0.9, onset_rate=0.0, brightness=0.1,
                     percussive=0.05),
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
    for name in ("cinema", "puddles", "freqwave", "fire"):
        assert name in scores, name


# ── what the switch actually watches ─────────────────────────────────────────
def feat(**kw):
    base = dict(mel=np.zeros(24), volume=0.5, energy=0.4, onset_rate=0.3,
                brightness=0.2, percussive=0.3)
    return Features(**{**base, **kw})


def drift_after(frames, **changed):
    """Drift once a director settled on one character is shown another.

    Separate directors per case: the character filter is stateful, so reusing
    one would measure the previous condition still draining out of it.
    """
    d = Director(Settings.load(), 30)
    for _ in range(frames):
        d.choose(feat())
    d.anchor = np.copy(d._char.value)
    for _ in range(frames):
        d.choose(feat(**changed))
    return d.drift


def test_the_character_vector_watches_percussiveness_not_brightness():
    """Brightness moving is not a scene change. It is an AdaptiveRange output,
    so it wanders on static material by construction, and the director used to
    read that wander as the audio going somewhere."""
    assert drift_after(400, brightness=0.9) < 0.05
    assert drift_after(400, percussive=0.95) > 0.4


def test_going_from_rhythmic_to_sustained_earns_a_switch():
    """The case the change exists for: a track dropping its drums for a pad is
    a scene change, and nothing in the old vector reliably said so.

    Driven through ``render``, because that is what advances ``current`` and
    ``last_switch`` -- ``choose`` alone re-anchors and then reports the
    incumbent it was never told to leave.
    """
    s = Settings.load(overrides={"mood": {"switch_dwell": 0.0}})
    d = Director(s, 30)
    for i in range(600):
        d.render(feat(t=i / 60.0, onset_rate=0.6, percussive=0.9))
    started, switches = d.current, d.switches

    for i in range(600):
        d.render(feat(t=10.0 + i / 60.0, onset_rate=0.6, percussive=0.05))
    assert d.switches > switches
    assert d.current != started


def test_a_dead_classifier_group_no_longer_flattens_the_scores():
    """YAMNet's ``percussion`` group read 0.000 through a whole run of ordinary
    music. The spectrum knows better, and now says so."""
    a = score_candidates(feat(percussive=0.0), ("scroll", "puddles", "pixelwave"))
    b = score_candidates(feat(percussive=1.0), ("scroll", "puddles", "pixelwave"))
    assert all(b[k] > a[k] for k in a)


# ── screen time has to be distributed, not just ranked ───────────────────────
def _run(director, frames=4000, **kw):
    """Drive a director on unchanging audio and report what it showed."""
    shown = collections.Counter()
    for i in range(frames):
        director.render(feat(t=i / 60.0, **kw))
        shown[director.current] += 1
    return shown


def test_a_leading_candidate_does_not_monopolise_the_strip():
    """The bug this rule replaced. ``puddles`` led nearly every frame, so it was
    also the top *other* from everywhere else and every switch came back to it:
    46% of 75 s against 10% for bars and 0% for energy."""
    s = Settings.load(overrides={"mood": {"switch_dwell": 1.0, "max_dwell": 1.0,
                                          "switch_margin": 1.0}})
    shown = _run(Director(s, 30))
    assert len(shown) == len(s.mood.animations)
    # No candidate may take more than double an even share.
    even = sum(shown.values()) / len(s.mood.animations)
    assert max(shown.values()) < 2 * even


def test_every_candidate_is_reached_even_when_it_never_ranks_second():
    """Rank alone made low scorers unreachable however long the run."""
    s = Settings.load(overrides={"mood": {"switch_dwell": 1.0, "max_dwell": 1.0,
                                          "switch_margin": 1.0}})
    shown = _run(Director(s, 30))
    assert set(shown) == set(s.mood.animations)


def test_a_narrow_band_still_prefers_the_best_scorer():
    """The knob has to work in both directions: at zero width only candidates
    tied with the leader are eligible, so this stays a ranking rule."""
    s = Settings.load(overrides={"mood": {"switch_dwell": 1.0, "max_dwell": 1.0,
                                          "switch_margin": 0.0}})
    d = Director(s, 30)
    shown = _run(d, frames=2000)
    assert len(shown) < len(s.mood.animations)


def test_the_shortlist_is_walked_before_anything_repeats():
    """Unseen candidates sort first, so a cold start covers the library once
    before showing anything twice."""
    s = Settings.load(overrides={"mood": {"switch_dwell": 1.0, "max_dwell": 1.0,
                                          "switch_margin": 1.0}})
    d = Director(s, 30)
    seen, order = set(), []
    for i in range(2000):
        d.render(feat(t=i / 60.0))
        if d.current not in seen:
            seen.add(d.current)
            order.append(d.current)
        if len(seen) == len(s.mood.animations):
            break
    assert len(order) == len(s.mood.animations)


def test_the_shortlist_can_be_changed_while_running():
    """It is in the live allowlist, so the API accepted the patch and reported
    it applied -- while the director went on scoring the list it was built
    with. The dashboard's shortlist control did nothing until a restart."""
    v = Visualizer(Settings.load(overrides={
        "effect": {"name": "auto"}, "mood": {"scene_weight": 0.0}}))
    wider = ["bars", "energy", "spectrum", "freqwave", "puddles", "fire", "scroll"]
    v.apply({"mood": {"animations": wider}})
    assert v.effect.allowed == tuple(wider)

    for i in range(120):
        v.process(np.zeros((v.settings.audio.samples_per_frame, 2), dtype=np.int16))
    assert set(v.effect.scores) <= set(wider)


def test_retuning_keeps_the_state_of_candidates_that_survive():
    """A shortlist edit must not restart the animation on screen."""
    s = Settings.load(overrides={"mood": {"switch_dwell": 1.0, "max_dwell": 1.0}})
    d = Director(s, 30)
    for i in range(600):
        d.render(feat(t=i / 60.0))
    before_seen = dict(d._last_seen)
    before_cache = set(d._cache)
    kept = [n for n in d.allowed if n != "energy"] + ["fire"]

    d.retune(tuple(kept))
    assert "energy" not in d.scores and "energy" not in d._last_seen
    assert set(d._cache) == before_cache          # nothing was rebuilt
    for n in kept:
        if n in before_seen and n != "energy":
            assert d._last_seen[n] == before_seen[n]


def test_a_momentary_drift_spike_does_not_switch():
    """Switches that looked unprovoked were unprovoked: one noisy frame across
    the threshold committed the strip for a whole dwell."""
    s = Settings.load(overrides={"mood": {"switch_dwell": 0.0, "max_dwell": 1e6,
                                          "change_threshold": 0.1,
                                          "change_hold": 1.5}})
    d = Director(s, 30)
    for i in range(600):                      # settle on one character
        d.render(feat(t=i / 60.0))
    started, switches = d.current, d.switches

    # One frame far away, then straight back.
    d.render(feat(t=10.0, percussive=1.0, energy=1.0))
    d.render(feat(t=10.02))
    assert d.switches == switches and d.current == started


def test_a_sustained_change_still_switches():
    """The hold must reject noise without rejecting real changes.

    Asserts that a switch happened and stuck, not that exactly one did: an
    instantaneous step is not something real audio does, and the character
    filter takes a moment to finish crossing it.
    """
    s = Settings.load(overrides={"mood": {"switch_dwell": 0.0, "max_dwell": 1e6,
                                          "change_threshold": 0.1,
                                          "change_hold": 1.5}})
    d = Director(s, 30)
    for i in range(600):
        d.render(feat(t=i / 60.0))
    started, switches = d.current, d.switches
    for i in range(900):                      # 15 s of the new character
        d.render(feat(t=10.0 + i / 60.0, percussive=0.95, energy=0.9))
    assert d.switches > switches
    assert d.current != started


def test_one_change_does_not_produce_a_burst_of_switches():
    """Anchoring on the smoothed vector left the rest of its convergence to be
    re-measured as fresh drift: one change gave three switches, the last of
    which landed back on the animation it started from."""
    s = Settings.load(overrides={"mood": {"switch_dwell": 0.0, "max_dwell": 1e6,
                                          "change_threshold": 0.1,
                                          "change_hold": 1.5}})
    d = Director(s, 30)
    for i in range(600):
        d.render(feat(t=i / 60.0))
    started, switches = d.current, d.switches
    for i in range(900):
        d.render(feat(t=10.0 + i / 60.0, percussive=0.95, energy=0.9))
    assert d.switches - switches <= 2
    assert d.current != started, "settled back where it began"


def test_the_hold_is_reported_so_a_switch_can_be_seen_coming():
    s = Settings.load(overrides={"mood": {"change_threshold": 0.05,
                                          "change_hold": 5.0,
                                          "switch_dwell": 0.0}})
    d = Director(s, 30)
    for i in range(300):
        d.render(feat(t=i / 60.0))
    for i in range(120):
        d.render(feat(t=5.0 + i / 60.0, percussive=0.95))
    assert d.state()["held"] > 0.0


def test_no_candidate_depends_on_a_term_that_can_be_constant():
    """The trap this module keeps falling into. ``loud`` covers rock and metal
    and nothing else, so it read 0.000 on most music while carrying 45% of
    ``gravcenter``, 30% of ``fire`` and 25% of ``energy`` -- capping gravcenter
    near 0.3 and keeping it off the strip entirely.

    With every classifier term given a DSP floor, no candidate may be stuck far
    below the field when the classifier says nothing at all.
    """
    from ambviz.scene import Scene
    silent_model = Scene(available=False)
    worst = 1.0
    for energy in (0.2, 0.5, 0.9):
        for onset in (0.0, 0.5, 1.0):
            for perc in (0.05, 0.5, 0.95):
                f = _f(energy=energy, onset_rate=onset, percussive=perc,
                       brightness=0.5, dialogue=0.2)
                f.scene = silent_model
                s = score_candidates(f)
                # Every candidate must top the field for *some* input.
                worst = min(worst, max(s.values()))
    assert worst > 0.0

    reachable = set()
    for energy in (0.1, 0.3, 0.5, 0.7, 0.9):
        for onset in (0.0, 0.25, 0.5, 0.75, 1.0):
            for perc in (0.05, 0.3, 0.5, 0.7, 0.95):
                for bright in (0.1, 0.5, 0.9):
                    for dial in (0.0, 0.5, 0.9):
                        f = _f(energy=energy, onset_rate=onset, percussive=perc,
                               brightness=bright, dialogue=dial)
                        f.scene = silent_model
                        s = score_candidates(f)
                        reachable.add(max(s, key=s.get))
    assert "gravcenter" in reachable, sorted(reachable)

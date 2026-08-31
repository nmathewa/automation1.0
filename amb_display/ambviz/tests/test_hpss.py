"""Harmonic/percussive separation, and the character vector it feeds.

The director's complaint was that it switched on the clock rather than on the
music. Two of the four things it measured could not have told it otherwise:
``energy`` was pinned near a constant by the classifier bias, and ``brightness``
is an AdaptiveRange output that drifts on any material at all. These tests pin
the replacement to the property that makes it usable -- that it is absolute.
"""

import numpy as np
import pytest

from ambviz.dsp import HarmonicPercussive
from ambviz.features import Features
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings

BINS = 1025


def sustained(bins=BINS):
    """A held chord: narrow ridges that persist frame to frame."""
    s = np.zeros(bins)
    s[[120, 240, 360]] = [1.0, 0.7, 0.4]
    return s


def hit(rng, bins=BINS):
    """A drum: broadband, present for one frame."""
    return sustained(bins) + rng.random(bins) * 0.6


# ── the separator itself ─────────────────────────────────────────────────────
def test_a_held_chord_reads_as_harmonic():
    h = HarmonicPercussive(BINS)
    for _ in range(30):
        out = h.update(sustained())
    assert HarmonicPercussive.ratio(*out) < 0.05


def test_a_broadband_hit_reads_as_percussive():
    rng = np.random.default_rng(0)
    h = HarmonicPercussive(BINS)
    for _ in range(30):
        h.update(sustained())
    assert HarmonicPercussive.ratio(*h.update(hit(rng))) > 0.9


def test_the_smoothed_ratio_tracks_hit_density():
    """The point of the feature. One hit in four settles near 0.25, one in two
    near 0.5 -- so the number means "how rhythmic", not merely "loud"."""
    seen = {}
    for period in (2, 4):
        rng = np.random.default_rng(1)
        h = HarmonicPercussive(BINS)
        vals = []
        for i in range(80):
            frame = hit(rng) if i % period == 0 else sustained()
            vals.append(HarmonicPercussive.ratio(*h.update(frame)))
        seen[period] = float(np.mean(vals[40:]))
    assert seen[4] == pytest.approx(0.25, abs=0.1)
    assert seen[2] == pytest.approx(0.5, abs=0.1)
    assert seen[2] > seen[4]


def test_it_is_absolute_rather_than_adaptive():
    """The whole reason it replaced ``brightness``. A tenfold gain change is not
    a change of character, and an AdaptiveRange would have renormalised it away
    to the same mid-range number either way."""
    rng = np.random.default_rng(2)
    out = []
    for gain in (0.1, 1.0, 10.0):
        h = HarmonicPercussive(BINS)
        vals = []
        for i in range(60):
            frame = (hit(rng) if i % 4 == 0 else sustained()) * gain
            vals.append(HarmonicPercussive.ratio(*h.update(frame)))
        out.append(float(np.mean(vals[30:])))
    assert max(out) - min(out) < 0.05


def test_static_material_does_not_drift():
    """What brightness could not promise. Nothing changes, so the feature must
    not move -- drift here is a scene change the director would act on."""
    h = HarmonicPercussive(BINS)
    vals = [HarmonicPercussive.ratio(*h.update(sustained())) for _ in range(120)]
    assert np.ptp(vals[20:]) < 0.02


def test_the_history_is_seeded_rather_than_ramped():
    """Starting from zeros makes the time median near zero, the mask fully
    percussive, and fires a switch at startup on any material."""
    h = HarmonicPercussive(BINS)
    first = HarmonicPercussive.ratio(*h.update(sustained()))
    assert first < 0.05


def test_it_only_ever_looks_backwards():
    """Causal by construction: frames already emitted must never change when
    later audio arrives."""
    rng = np.random.default_rng(3)
    frames = [hit(rng) if i % 3 == 0 else sustained() for i in range(40)]
    a = HarmonicPercussive(BINS)
    short = [HarmonicPercussive.ratio(*a.update(f)) for f in frames[:20]]
    b = HarmonicPercussive(BINS)
    long = [HarmonicPercussive.ratio(*b.update(f)) for f in frames][:20]
    assert short == long


def test_an_even_window_is_rounded_up():
    assert HarmonicPercussive(BINS, frames=8).frames == 9


def test_silence_reports_the_neutral_value():
    h = HarmonicPercussive(BINS)
    assert HarmonicPercussive.ratio(*h.update(np.zeros(BINS))) == 0.5


def test_it_reshapes_rather_than_crashing_on_a_new_bin_count():
    h = HarmonicPercussive(BINS)
    h.update(sustained())
    assert 0.0 <= HarmonicPercussive.ratio(*h.update(sustained(513))) <= 1.0


# ── the pipeline exposes it ──────────────────────────────────────────────────
def tone(seconds, rate, fps, freq=440.0):
    n = int(rate / fps)
    t = np.arange(int(seconds * rate)) / rate
    y = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)
    return [y[i:i + n] for i in range(0, len(y) - n, n)]


def test_a_pure_tone_drives_the_feature_down():
    v = Visualizer(Settings.load(overrides={"mood": {"scene_weight": 0.0}}))
    s = v.settings
    for frame in tone(3.0, s.audio.rate, s.audio.fps):
        v.process(frame)
    assert v.features.percussive < 0.35


def test_the_feature_is_published_in_telemetry():
    v = Visualizer(Settings.load(overrides={"mood": {"scene_weight": 0.0}}))
    s = v.settings
    for frame in tone(1.0, s.audio.rate, s.audio.fps):
        v.process(frame)
    assert "percussive" in v.snapshot()


def test_smoothing_is_tunable_without_losing_the_current_value():
    """Rebuilding the filter at the neutral 0.5 would inject a step into the
    character vector, which the director cannot tell from a scene change."""
    v = Visualizer(Settings.load(overrides={"mood": {"scene_weight": 0.0}}))
    s = v.settings
    for frame in tone(2.0, s.audio.rate, s.audio.fps):
        v.process(frame)
    before = float(v.percussive.value)
    v.apply({"dsp": {"percussive_smoothing": 6.0}})
    assert float(v.percussive.value) == pytest.approx(before)


def test_the_window_can_be_retuned_live():
    v = Visualizer(Settings.load())
    v.apply({"dsp": {"hpss_frames": 15, "hpss_kernel": 9}})
    assert v.hpss.frames == 15 and v.hpss.kernel == 9

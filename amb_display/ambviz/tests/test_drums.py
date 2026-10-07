"""Instrument detectors, the tempo grid, and what drives ``beat``."""

import numpy as np
import pytest

from ambviz.features import BandOnsetDetector, TempoTracker
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings

RATE = 44100


def drum_loop(seconds, bpm=120.0, kick=True, snare=False, tone=0.0, seed=0):
    """A kick on every beat, optionally a snare on the off-beats, over an
    optional sustained tone standing in for a held vocal note."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    out = np.zeros_like(t)
    beat = 60.0 / bpm
    phase = t % beat
    if kick:
        env = np.exp(-phase * 30)
        out += np.sin(2 * np.pi * 60 * phase) * env * 12000
    if snare:
        off = (t + beat / 2) % beat
        out += rng.standard_normal(len(t)) * np.exp(-off * 40) * 6000
    if tone:
        out += np.sin(2 * np.pi * 440 * t) * tone
    return out


def run(v, signal):
    n = v.samples_per_frame
    seen = []
    for i in range(len(signal) // n):
        v.process(signal[i * n:(i + 1) * n])
        seen.append((v.features.t, v.features.beat, v.drums, v.tempo))
    return seen


def spectrum_frames(signal, n_fft=2048, hop=735):
    window = np.hamming(2 * hop)
    for i in range(1, len(signal) // hop):
        chunk = signal[(i - 1) * hop:(i + 1) * hop]
        yield i * hop / RATE, np.abs(np.fft.rfft(np.pad(chunk * window, (0, n_fft - len(chunk)))))


def test_the_kick_detector_finds_kicks_below_the_filterbank():
    """The Mel bank starts at 200 Hz, so a 60 Hz kick was invisible to the
    original detector. The kick band is below it on purpose."""
    d = BandOnsetDetector(40, 150)
    hits = [t for t, spec in spectrum_frames(drum_loop(6.0)) if d.update(spec, RATE, t).hit]
    hits = [h for h in hits if h > 1.0]
    assert len(hits) >= 8
    assert np.median(np.diff(hits)) == pytest.approx(0.5, abs=0.03)


def test_a_held_note_does_not_fire_a_band_detector():
    d = BandOnsetDetector(200, 6000)
    fired = [t for t, spec in spectrum_frames(drum_loop(4.0, kick=False, tone=8000.0))
             if d.update(spec, RATE, t).hit and t > 0.5]
    assert fired == []


def test_detectors_are_independent_of_playback_level():
    """Log compression makes flux a ratio, so the same music at a quarter of
    the volume is the same set of hits."""
    loud, quiet = drum_loop(5.0), drum_loop(5.0) * 0.25
    found = []
    for signal in (loud, quiet):
        d = BandOnsetDetector(40, 150)
        found.append(sum(d.update(s, RATE, t).hit for t, s in spectrum_frames(signal)))
    assert abs(found[0] - found[1]) <= 1


def test_the_tracker_locks_to_a_steady_tempo():
    v = Visualizer(Settings.load())
    seen = run(v, drum_loop(10.0, bpm=120.0, snare=True))
    assert v.tempo.locked
    assert v.tempo.bpm == pytest.approx(120.0, rel=0.03)
    assert v.beat_period == pytest.approx(0.5, rel=0.03)
    pulses = [t for t, _, _, tempo in seen if tempo.pulse and t > 6.0]
    assert np.median(np.diff(pulses)) == pytest.approx(0.5, abs=0.02)


def test_the_tracker_does_not_flip_octaves_once_locked():
    """Double and half time often score within a few percent of the true tempo.
    Flipping between them halves the strip's pace for no audible reason."""
    tracker = TempoTracker(fps=60.0)
    beat_frames = 20                               # 180 BPM at 60 fps
    bpms = []
    for i in range(60 * 12):
        hit = i % beat_frames == 0
        tempo = tracker.update(1.0 if hit else 0.05, i / 60.0, hit=hit)
        if tempo.locked:
            bpms.append(round(tempo.bpm))
    assert bpms, "never locked"
    assert max(bpms) - min(bpms) <= 6, sorted(set(bpms))


def test_a_new_track_forgets_the_old_tempo():
    """Silence longer than output.track_gap is how a song change is seen; the
    grid belonged to the last song and has to be relearned."""
    v = Visualizer(Settings.load())
    n = v.samples_per_frame
    run(v, drum_loop(10.0, bpm=120.0, snare=True))
    assert v.tempo.locked and v.tempo.bpm == pytest.approx(120.0, rel=0.03)
    for _ in range(int(1.0 * 60)):                 # a second of silence
        v.process(np.zeros(n))
    tracks = v.tracks
    run(v, drum_loop(0.5, bpm=150.0, snare=True, seed=1))
    assert v.tracks == tracks + 1
    assert not v.tempo.locked, "still claiming the previous song's grid"
    run(v, drum_loop(10.0, bpm=150.0, snare=True, seed=2))
    assert v.tempo.bpm == pytest.approx(150.0, rel=0.03)


def test_a_moving_bass_line_does_not_take_the_lead():
    """Kick-band energy is often bass notes. In ``auto`` the kick leads only
    while its hits land on the grid; a bass line wandering off it must not."""
    v = Visualizer(Settings.load())
    rng = np.random.default_rng(4)
    t = np.arange(int(12.0 * RATE)) / RATE
    # Drums on a 120 BPM grid, broadband so the full-band detector hears them.
    beat_phase = t % 0.5
    drums = rng.standard_normal(len(t)) * np.exp(-beat_phase * 40) * 7000
    # Bass notes starting at irregular times, off the grid.
    bass = np.zeros_like(t)
    starts = np.sort(rng.uniform(0, 12.0, 30))
    for s0 in starts:
        i = int(s0 * RATE)
        seg = t[i:i + int(0.3 * RATE)] - s0
        bass[i:i + len(seg)] += np.sin(2 * np.pi * rng.uniform(50, 110) * seg) * 9000
    run(v, drums + bass)
    assert v.beat_driver == "mix"


@pytest.mark.parametrize("source", ["auto", "grid", "kick", "snare", "mix", "legacy"])
def test_every_beat_source_runs(source):
    v = Visualizer(Settings.load(overrides={"dsp": {"beat_source": source}}))
    seen = run(v, drum_loop(6.0, bpm=120.0, snare=True))
    assert sum(beat for _, beat, _, _ in seen) >= 4


def test_legacy_is_the_original_detector():
    v = Visualizer(Settings.load(overrides={"dsp": {"beat_source": "legacy"}}))
    signal, n, fired = drum_loop(4.0, snare=True), v.samples_per_frame, 0
    for i in range(len(signal) // n):
        v.process(signal[i * n:(i + 1) * n])
        original = v.onsets._last_beat == v.features.t
        assert v.features.beat == original
        fired += original
    assert fired >= 4


def test_an_unknown_beat_source_is_rejected():
    with pytest.raises(ValueError, match="beat_source"):
        Settings.load(overrides={"dsp": {"beat_source": "vibes"}})


def test_drum_telemetry_is_published():
    v = Visualizer(Settings.load())
    run(v, drum_loop(2.0))
    snap = v.snapshot()
    assert set(snap["drums"]) == {"kick", "snare", "hat"}
    assert {"bpm", "phase", "locked", "confidence"} <= set(snap["tempo"])
    assert snap["beat_driver"] in ("kick", "mix")


def test_the_detectors_can_be_retuned_live():
    v = Visualizer(Settings.load())
    v.apply({"dsp": {"kick_band": [30.0, 120.0], "drum_sensitivity": 3.0}})
    assert (v.kick.low, v.kick.high) == (30.0, 120.0)
    assert v.kick.sensitivity == 3.0

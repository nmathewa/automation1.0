"""Centre-channel suppression: vocals cancel, the kick and panned parts do not."""

import numpy as np
import pytest

from ambviz.pipeline import Visualizer
from ambviz.settings import Settings

RATE = 44100


def mix(seconds=1.0, rate=RATE, kick=0.8):
    """Kick centred at 60 Hz, vocal centred at 900 Hz, guitar panned left."""
    t = np.arange(int(rate * seconds)) / rate
    low = kick * np.sin(2 * np.pi * 60 * t)
    vocal = 0.6 * np.sin(2 * np.pi * 900 * t)
    guitar = 0.5 * np.sin(2 * np.pi * 1500 * t)
    return np.stack([low + vocal + guitar, low + vocal], axis=1)


def band_energy(v, stereo, hz, frames=30):
    """Level in the Mel band nearest ``hz`` after running frames through."""
    total = None
    n = v.samples_per_frame
    for i in range(frames):
        block = stereo[i * n:(i + 1) * n]
        if len(block) < n:
            block = np.resize(block, (n, 2))
        v.process(block * 2 ** 15)
        total = np.copy(v.mel) if total is None else total + v.mel
    idx = int(np.argmin(np.abs(v.mel_bank.center_frequencies - hz)))
    return float(total[idx])


def viz(amount):
    return Visualizer(Settings.load(overrides={"dsp": {
        "vocal_suppression": amount, "fft_bins": 40,
        "min_frequency": 40, "max_frequency": 8000}}))


def test_stereo_is_detected():
    v = viz(0.0)
    v.process(mix(0.1)[:v.samples_per_frame] * 2 ** 15)
    assert v.stereo is True


def test_mono_input_still_works_and_reports_mono():
    v = viz(0.9)
    v.process(np.zeros(v.samples_per_frame))
    assert v.stereo is False


def test_suppression_leaves_rhythm_alone():
    """The reason it applies to colour only.

    L - R removes everything centred, and in a real mix that is the kick and
    snare as much as the voice. Applied to the whole analysis it removed the
    song rather than the singer -- onsets fell from 93 to 21 on this material.
    Level, onsets and energy must come from the full mix.
    """
    rate = RATE

    def clip(i, n):
        t = (np.arange(n) + i * n) / rate
        note = [330, 392, 440, 494][int((i // 25) % 4)]
        voice = np.sin(2 * np.pi * note * t) * (0.55 + 0.45 * np.sin(2 * np.pi * 5 * t)) * 0.8
        kick = np.exp(-((t * 2) % 1.0) * 10) * np.sin(2 * np.pi * 60 * t) * 1.1
        return np.stack([voice + kick + 0.3 * np.sin(2 * np.pi * 147 * t),
                         voice + kick + 0.3 * np.sin(2 * np.pi * 220 * t)], axis=1) * 7000

    counts = []
    for amount in (0.0, 0.9):
        v = Visualizer(Settings.load(overrides={
            "effect": {"name": "cinema"}, "dsp": {"vocal_suppression": amount},
            "mood": {"scene_weight": 0.0}}))
        n = v.samples_per_frame
        before = v.beats
        for i in range(int(15 * v.settings.audio.fps)):
            v.process(clip(i, n))
        counts.append(v.beats - before)

    plain, suppressed = counts
    assert suppressed >= plain * 0.8, \
        f"suppression cost the beat: {plain} onsets became {suppressed}"


def test_suppression_still_removes_centred_content_from_the_colour_path():
    v = viz(0.95)
    bins = 1025
    freqs = np.fft.rfftfreq(2 * (bins - 1), 1 / RATE)
    inside = (freqs >= 180) & (freqs <= 5000)
    out = v._suppress_centre(np.ones(bins), np.zeros(bins), 0.95)
    assert out[inside].max() <= 0.06, "centred content should be nearly gone in band"


def test_the_band_protects_low_frequencies():
    """The kick is centred too, so it would vanish without the band limit."""
    v = viz(1.0)
    bins = 1025                     # what rfft returns for a 2048-sample window
    freqs = np.fft.rfftfreq(2 * (bins - 1), 1 / RATE)
    mid = np.ones(bins)
    side = np.zeros(bins)           # perfectly centred: the side channel is silent
    out = v._suppress_centre(mid, side, 1.0)

    low, high = v.settings.dsp.vocal_band
    assert np.allclose(out[freqs < low], 1.0), "below the band must pass untouched"
    assert np.allclose(out[freqs > high], 1.0), "above the band must pass untouched"
    assert np.allclose(out[(freqs >= low) & (freqs <= high)], 0.0), \
        "centred content inside the band must cancel"


def test_zero_suppression_changes_nothing():
    v = viz(0.0)
    mid = np.linspace(1, 2, 1025)
    assert np.allclose(v._suppress_centre(mid, np.zeros(1025), 0.0), mid)


def test_band_change_rebuilds_the_mask():
    v = viz(0.5)
    v._suppress_centre(np.ones(1025), np.zeros(1025), 0.5)
    assert v._band_mask is not None
    v.apply({"dsp": {"vocal_band": [300.0, 3000.0]}})
    assert v._band_mask is None, "the cached mask must be invalidated"


@pytest.mark.parametrize("bad, message", [
    ({"vocal_suppression": 1.5}, "vocal_suppression"),
    ({"vocal_suppression": -0.1}, "vocal_suppression"),
    ({"vocal_band": [5000.0, 180.0]}, "low below high"),
    ({"vocal_band": [-10.0, 500.0]}, "must not be negative"),
])
def test_validation(bad, message):
    with pytest.raises(ValueError, match=message):
        Settings.load(overrides={"dsp": bad})


def test_suppression_is_live_tunable():
    from ambviz.control import CommandQueue
    q = CommandQueue(Settings.load())
    q.submit({"dsp": {"vocal_suppression": 0.8}})
    assert q.pending.dsp.vocal_suppression == 0.8


def test_telemetry_reports_it():
    v = viz(0.7)
    snap = v.snapshot()
    assert snap["vocal_suppression"] == 0.7
    assert snap["vocal_band"] == [180.0, 5000.0]
    assert "stereo" in snap

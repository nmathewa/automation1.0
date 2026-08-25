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


def test_suppression_cancels_the_vocal_relative_to_a_panned_part():
    """Measured as a ratio inside one run.

    The Mel output is gain-controlled, so absolute levels are renormalised every
    frame and cannot be compared between two runs -- removing the vocal makes
    the AGC re-expand everything else, which looks like the kick got louder.
    The vocal-to-guitar ratio is immune to that: both are scaled by the same
    gain, so only their relative change shows.
    """
    # No kick here: it is centred and very loud, and the gain control would
    # normalise to it and crush both bands under test into the noise floor.
    # Low-frequency protection has its own test below.
    stereo = mix(2.0, kick=0.0)

    def ratio(amount):
        vocal = band_energy(viz(amount), stereo, 900)
        guitar = band_energy(viz(amount), stereo, 1500)
        return vocal / max(guitar, 1e-9)

    off, on = ratio(0.0), ratio(0.95)
    assert on < off * 0.5, f"vocal:guitar ratio only fell from {off:.3f} to {on:.3f}"


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

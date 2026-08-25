import numpy as np
import pytest

from ambviz.features import Features, OnsetDetector
from ambviz.settings import Settings
from ambviz.sources import make_source
from ambviz.pipeline import Visualizer


def test_thirds_splits_the_spectrum():
    f = Features(mel=np.array([1.0] * 4 + [0.5] * 4 + [0.0] * 4), volume=0.5)
    low, mid, high = f.thirds()
    assert (low, mid, high) == (1.0, 0.5, 0.0)
    assert f.bands == 12


def test_detector_is_silent_on_the_first_frame():
    d = OnsetDetector()
    assert d.update(np.zeros(8), 0.0) == (0.0, False, 0.0)


def test_detector_fires_on_a_step_and_not_on_a_hold():
    d = OnsetDetector()
    quiet, loud = np.zeros(8), np.ones(8)
    for i in range(20):
        d.update(quiet, i * 0.02)
    _, beat, flux = d.update(loud, 0.5)
    assert beat and flux > 0
    # Holding the same level produces no new flux, so no repeat trigger.
    assert not d.update(loud, 0.6)[1]


def test_refractory_period_suppresses_a_retrigger():
    d = OnsetDetector(refractory=0.5)
    quiet, loud = np.zeros(8), np.ones(8)
    for i in range(20):
        d.update(quiet, i * 0.01)
    assert d.update(loud, 1.0)[1]
    d.update(quiet, 1.05)
    assert not d.update(loud, 1.1)[1]      # inside the refractory window
    d.update(quiet, 1.55)
    assert d.update(loud, 1.6)[1]          # past it


def test_onset_strength_decays_between_beats():
    d = OnsetDetector(decay=0.2)
    quiet, loud = np.zeros(8), np.ones(8)
    for i in range(20):
        d.update(quiet, i * 0.01)
    first = d.update(loud, 1.0)[0]
    later = d.update(quiet, 1.15)[0]
    assert first > later >= 0.0
    assert d.update(quiet, 1.5)[0] == 0.0


def test_detector_locks_to_a_known_tempo(tmp_path):
    """The loop has an onset every 0.250 s: a kick on each beat of 120 BPM plus
    a hat on each offbeat. The detector should land on that grid."""
    import wave

    sr, beat = 44100, 0.5
    signal = np.zeros(int(sr * 8))
    for i in range(32):                      # an event every 0.25 s
        at = int(i * 0.25 * sr)
        env = np.exp(-np.linspace(0, 12, int(0.12 * sr)))
        signal[at:at + len(env)] += np.sin(2 * np.pi * 80 * np.arange(len(env)) / sr) * env
    pcm = (signal / np.abs(signal).max() * 0.8 * 32767).astype(np.int16)
    path = tmp_path / "click.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())

    s = Settings.load(overrides={"audio": {"source": "wav", "wav_path": str(path)}})
    v, src = Visualizer(s), make_source(s)
    gen = src.frames()
    for _ in range(40):
        v.process(next(gen))
    hits = []
    for _ in range(int(5 * s.audio.fps)):
        v.process(next(gen))
        if v.features.beat:
            hits.append(v.features.t)
    src.close()

    assert len(hits) >= 8, f"only {len(hits)} onsets found"
    gaps = np.diff(hits)
    assert 0.2 <= np.median(gaps) <= 0.3, f"median gap {np.median(gaps):.3f}s, expected ~0.25"


def test_pipeline_publishes_onset_telemetry():
    v = Visualizer(Settings.load())
    rng = np.random.default_rng(0)
    for _ in range(10):
        v.process(rng.normal(0, 3000, v.samples_per_frame))
    snap = v.snapshot()
    for key in ("onset", "beat", "beats", "flux", "effects"):
        assert key in snap, key
    assert isinstance(snap["effects"], list) and "gravcenter" in snap["effects"]


@pytest.mark.parametrize("sensitivity", [0.9, 1.0])
def test_sensitivity_must_exceed_one(sensitivity):
    with pytest.raises(ValueError, match="onset_sensitivity"):
        Settings.load(overrides={"dsp": {"onset_sensitivity": sensitivity}})

import json

import numpy as np
import pytest

from ambviz.pipeline import Visualizer
from ambviz.settings import Settings


@pytest.fixture
def visualizer():
    v = Visualizer(Settings.load())
    rng = np.random.default_rng(0)
    for _ in range(5):
        v.process(rng.normal(0, 3000, v.samples_per_frame))
    return v


def test_snapshot_is_json_serialisable(visualizer):
    json.dumps(visualizer.snapshot())


def test_snapshot_reports_the_pipeline(visualizer):
    s = visualizer.snapshot()
    assert s["frames"] == 5
    assert s["effect"] == "spectrum"
    assert s["pixels"] == 60
    assert s["bins"] == 24
    assert s["silent"] is False
    assert s["volume"] > 0
    assert len(s["mel"]) == 24


def test_band_centres_are_hertz_not_mel(visualizer):
    """Regression: compute_melmat returns centres in MEL. Publishing those as Hz
    understated the top of a 200-12000 Hz range as ~3.1 kHz."""
    centres = visualizer.snapshot()["center_frequencies"]
    assert len(centres) == 24
    assert centres == sorted(centres)
    assert 200 <= centres[0] <= 500
    assert 9000 <= centres[-1] <= 12000


def test_band_centres_follow_a_frequency_change(visualizer):
    visualizer.apply({"dsp": {"min_frequency": 80, "max_frequency": 6000}})
    centres = visualizer.snapshot()["center_frequencies"]
    assert 80 <= centres[0] <= 300
    assert 4000 <= centres[-1] <= 6000


def test_silence_is_reported(visualizer):
    # The rolling window spans `rolling_history` frames, so it takes that many
    # silent frames to flush the earlier audio out of it.
    for _ in range(visualizer.settings.audio.rolling_history):
        visualizer.process(np.zeros(visualizer.samples_per_frame))
    s = visualizer.snapshot()
    assert s["silent"] is True
    assert not any(s["mel"])


def test_fps_is_estimated():
    v = Visualizer(Settings.load())
    assert v.snapshot()["fps"] == 60.0  # seeded from settings before any frame
    rng = np.random.default_rng(0)
    for _ in range(10):
        v.process(rng.normal(0, 3000, v.samples_per_frame))
    assert v.snapshot()["fps"] > 0

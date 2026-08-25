import numpy as np
import pytest

from ambviz.control import CONTROLLABLE, CommandQueue, NotControllable
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings


@pytest.fixture
def queue():
    return CommandQueue(Settings.load())


def test_valid_patch_is_queued(queue):
    assert queue.submit({"effect": {"brightness": 0.5}}) == {"effect": {"brightness": 0.5}}
    assert len(queue) == 1
    assert queue.drain() == [{"effect": {"brightness": 0.5}}]
    assert len(queue) == 0


@pytest.mark.parametrize(
    "patch, exc, message",
    [
        ({"effect": {"brightness": 9}}, ValueError, "brightness"),
        ({"dsp": {"max_frequency": 30000}}, ValueError, "Nyquist"),
        ({"smoothing": {"red": [0.0, 0.5]}}, ValueError, "strictly between"),
        ({"output": {"pixels": 120}}, NotControllable, "set at startup"),
        ({"audio": {"rate": 22050}}, NotControllable, "set at startup"),
        ({"effect": {"nope": 1}}, KeyError, "unknown setting"),
        ({"nonsense": {"a": 1}}, KeyError, "unknown setting"),
        ({"effect": "energy"}, NotControllable, "not a settings section"),
        ({}, ValueError, "non-empty"),
        ("nope", ValueError, "non-empty"),
    ],
)
def test_bad_patches_never_queue(queue, patch, exc, message):
    with pytest.raises(exc, match=message):
        queue.submit(patch)
    assert len(queue) == 0
    assert queue.drain() == []


def test_rejection_leaves_pending_untouched(queue):
    queue.submit({"effect": {"brightness": 0.4}})
    with pytest.raises(ValueError):
        queue.submit({"effect": {"brightness": 12}})
    assert queue.pending.effect.brightness == 0.4


def test_patches_compose_on_pending(queue):
    queue.submit({"effect": {"brightness": 0.5}})
    queue.submit({"effect": {"name": "energy"}})
    assert queue.pending.effect.brightness == 0.5
    assert queue.pending.effect.name == "energy"


def test_live_settings_are_untouched_until_drained(queue):
    """The audio thread reads `settings`; the API thread must not write it."""
    queue.submit({"dsp": {"fft_bins": 8}})
    assert queue.settings.dsp.fft_bins == 24
    assert queue.pending.dsp.fft_bins == 8


def test_controllable_set_excludes_buffer_resizing_settings():
    for name in ("output.pixels", "audio.rate", "audio.fps", "audio.source",
                 "output.host", "output.port"):
        assert name not in CONTROLLABLE


# ── the run-loop side ────────────────────────────────────────────────────────
def viz(**overrides):
    return Visualizer(Settings.load(overrides=overrides))


def test_apply_rebuilds_mel_bank_only_on_frequency_change():
    v = viz()
    bank, effect = v.mel_bank.matrix, v.effect

    v.apply({"effect": {"brightness": 0.5}})
    assert v.mel_bank.matrix is bank and v.effect is effect

    v.apply({"dsp": {"max_frequency": 8000}})
    assert v.mel_bank.matrix is not bank
    assert v.effect is effect


def test_apply_rebuilds_effect_on_name_change():
    v = viz()
    effect = v.effect
    v.apply({"effect": {"name": "energy"}})
    assert v.effect is not effect
    assert type(v.effect).__name__ == "EnergyEffect"


def test_apply_mirror_change_resizes_the_effect():
    v = viz(output={"pixels": 60})
    assert v.width == 30
    v.apply({"effect": {"mirror": False}})
    assert v.width == 60
    assert v.process(np.zeros(v.samples_per_frame)).shape == (3, 60)


def test_apply_bins_change_keeps_rendering():
    v = viz()
    rng = np.random.default_rng(0)
    v.process(rng.normal(0, 3000, v.samples_per_frame))
    v.apply({"dsp": {"fft_bins": 8}})
    out = v.process(rng.normal(0, 3000, v.samples_per_frame))
    assert out.shape == (3, 60)
    assert len(v.snapshot()["mel"]) == 8


def test_apply_validates():
    v = viz()
    with pytest.raises(ValueError):
        v.apply({"effect": {"brightness": 50}})


def test_queue_to_visualizer_round_trip():
    """What the API accepts is what the run loop applies."""
    settings = Settings.load()
    v = Visualizer(settings)
    q = CommandQueue(settings)
    q.submit({"effect": {"name": "scroll", "brightness": 0.2}})
    for patch in q.drain():
        v.apply(patch)
    assert v.snapshot()["effect"] == "scroll"
    assert v.snapshot()["brightness"] == 0.2

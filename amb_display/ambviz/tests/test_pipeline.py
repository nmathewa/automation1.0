import numpy as np
import pytest

from ambviz.dsp import ExpFilter, interpolate
from ambviz.effects import EFFECTS
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings


def noise(n, rng=np.random.default_rng(0)):
    return rng.normal(0, 3000, n)


def viz(**overrides):
    return Visualizer(Settings.load(overrides=overrides))


@pytest.mark.parametrize("pixels", [1, 2, 60, 63, 107, 108])
@pytest.mark.parametrize("effect", sorted(EFFECTS))
@pytest.mark.parametrize("mirror", [True, False])
def test_output_shape_is_exact(pixels, effect, mirror):
    """Regression: mirrored effects used to emit 2 * (n // 2), so odd-length
    strips left the last pixel permanently dark."""
    v = viz(output={"pixels": pixels}, effect={"name": effect, "mirror": mirror})
    for _ in range(4):
        out = v.process(noise(v.samples_per_frame))
    assert out.shape == (3, pixels)


@pytest.mark.parametrize("effect", sorted(EFFECTS))
def test_values_stay_in_byte_range(effect):
    v = viz(effect={"name": effect})
    for _ in range(20):
        out = v.process(noise(v.samples_per_frame))
        assert out.min() >= 0 and out.max() <= 255


def test_mirror_is_symmetric():
    v = viz(output={"pixels": 60})
    for _ in range(3):
        out = v.process(noise(v.samples_per_frame))
    assert np.allclose(out[:, :30][:, ::-1], out[:, 30:])


def test_silence_blanks_the_strip():
    v = viz()
    out = v.process(np.zeros(v.samples_per_frame))
    assert v.silent
    assert not out.any()


def test_loud_input_is_not_silent():
    v = viz()
    v.process(noise(v.samples_per_frame))
    assert not v.silent
    assert v.volume > 0


def test_brightness_scales_down():
    dim, bright = viz(effect={"brightness": 0.1}), viz()
    signal = noise(bright.samples_per_frame)
    for _ in range(6):
        a, b = bright.process(signal), dim.process(signal)
    assert b.max() < a.max()


def test_set_effect_swaps_and_rejects():
    v = viz()
    v.set_effect("energy")
    assert v.settings.effect.name == "energy"
    v.process(noise(v.samples_per_frame))
    with pytest.raises(ValueError, match="unknown effect"):
        v.set_effect("strobe")


def test_effects_do_not_share_state():
    """Regression: the legacy module-level globals meant two visualizers in one
    process corrupted each other's filters."""
    a, b = viz(), viz()
    signal = noise(a.samples_per_frame)
    for _ in range(5):
        a.process(signal)
    solo = viz()
    for _ in range(5):
        expected = solo.process(signal)
        got = b.process(signal)
    assert np.allclose(expected, got)


def test_expfilter_rises_fast_and_decays_slow():
    f = ExpFilter(0.0, alpha_decay=0.01, alpha_rise=0.9)
    assert f.update(1.0) > 0.8          # snaps up
    for _ in range(3):
        f.update(0.0)
    assert f.value > 0.8                 # falls away slowly


def test_expfilter_rejects_bad_alpha():
    for bad in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError):
            ExpFilter(0.0, alpha_decay=bad)


def test_interpolate_resizes_both_ways():
    assert len(interpolate(np.arange(10.0), 3)) == 3
    assert len(interpolate(np.arange(3.0), 10)) == 10
    same = np.arange(5.0)
    assert interpolate(same, 5) is same

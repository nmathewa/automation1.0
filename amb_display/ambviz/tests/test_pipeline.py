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


# ── the effects added in wave 2 ──────────────────────────────────────────────
def _render(name, frames, **feature_kw):
    """Run one effect for N frames and return the stacked output."""
    from ambviz.effects import EFFECTS
    from ambviz.features import Features
    settings = Settings.load(overrides={"output": {"pixels": 60}})
    effect = EFFECTS[name](settings, 60)
    out = []
    for i in range(frames):
        kw = {k: (v(i) if callable(v) else v) for k, v in feature_kw.items()}
        kw.setdefault("mel", np.linspace(0.2, 0.8, settings.dsp.fft_bins))
        out.append(effect.render(Features(volume=0.3, t=i / 60.0, **kw)))
    return np.array(out)


def test_pacifica_swells_with_level_and_never_goes_fully_dark():
    """The ambient candidate: a strip that drops to black during a quiet line
    is more distracting than one that keeps moving."""
    quiet = _render("pacifica", 400, energy=0.02)
    loud = _render("pacifica", 400, energy=0.95)
    assert loud[200:].mean() > quiet[200:].mean() * 2
    assert quiet[200:].max() > 0, "quiet scene went completely black"


def test_puddles_decays_to_nothing_without_onsets():
    """The sparse candidate is the only one that is meant to go dark."""
    lit = _render("puddles", 60, beat=True, onset=0.9, centroid=0.5)
    assert lit.max() > 100, "no puddle was ever dropped"
    silent = _render("puddles", 400, beat=False, onset=0.0)
    assert silent[-1].max() < 1.0


def test_freqwave_colours_the_whole_strip_by_pitch():
    """Its whole point: hue comes from frequency, not from position."""
    low = _render("freqwave", 200, energy=0.8, centroid_hz=250.0)[-1]
    high = _render("freqwave", 200, energy=0.8, centroid_hz=9000.0)[-1]
    # Uniform along the strip: every pixel shares one hue.
    for frame in (low, high):
        lit = frame[:, frame.max(axis=0) > 20]
        ratio = lit / (lit.max(axis=0) + 1e-9)
        assert ratio.std(axis=1).max() < 0.05, "hue varied along the strip"
    # And the two pitches are visibly different colours.
    def norm(f):
        return f.mean(axis=1) / (f.mean(axis=1).sum() + 1e-9)
    assert np.abs(norm(low) - norm(high)).sum() > 0.2


def test_fire_stays_alight_under_energy_and_gutters_without_it():
    hot = _render("fire", 400, energy=0.9, onset=0.5, beat=lambda i: i % 8 == 0)
    cold = _render("fire", 400, energy=0.0, onset=0.0, beat=False)
    assert hot[200:].mean() > cold[200:].mean() * 3
    assert hot[200:].max() > 150, "flame never got hot"


def test_nothing_animates_on_frozen_audio():
    """The property that caught freqwave: it drew a hue over a travelling sine
    of fixed speed, so with the same frame fed repeatedly it still produced 56%
    of its normal motion while every other effect produced none. Motion that
    survives frozen audio is motion the music does not control."""
    from ambviz.effects import EFFECTS
    from ambviz.features import Features
    settings = Settings.load(overrides={"output": {"pixels": 60}})
    mel = np.linspace(0.2, 0.8, settings.dsp.fft_bins)
    # Exempt, and each for a stated reason rather than to make the test pass:
    #   auto      delegates to whichever animation it picked
    #   pacifica  is a slow ambient drift by construction
    #   fire      inherits Fire2012's random cooling and sparks, so it flickers
    #             whatever the audio does. Only its spark rate and cooling are
    #             music-driven. It shares freqwave's flaw and is kept out of
    #             the default shortlist partly for that reason.
    exempt = {"auto", "pacifica", "fire"}
    for name in sorted(set(EFFECTS) - exempt):
        effect = EFFECTS[name](settings, 60)
        out = [effect.render(Features(mel=np.copy(mel), volume=0.3, energy=0.5,
                                      centroid=0.4, centroid_hz=900.0, t=i / 60.0))
               for i in range(240)]
        settled = np.array(out[120:])
        moved = np.abs(np.diff(settled, axis=0)).mean()
        assert moved < 0.5, f"{name} moves {moved:.3f} on frozen audio"

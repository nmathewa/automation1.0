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
    #   fire      inherits Fire2012's random cooling and sparks, so it flickers
    #             whatever the audio does. Only its spark rate and cooling are
    #             music-driven. It shares freqwave's flaw and is kept out of
    #             the default shortlist partly for that reason.
    exempt = {"auto", "fire"}
    for name in sorted(set(EFFECTS) - exempt):
        effect = EFFECTS[name](settings, 60)
        out = [effect.render(Features(mel=np.copy(mel), volume=0.3, energy=0.5,
                                      centroid=0.4, centroid_hz=900.0, t=i / 60.0))
               for i in range(240)]
        settled = np.array(out[120:])
        moved = np.abs(np.diff(settled, axis=0)).mean()
        assert moved < 0.5, f"{name} moves {moved:.3f} on frozen audio"


def test_puddles_fades_faster_when_the_music_is_busy():
    """A fixed decay held a puddle for ~3 s whatever was playing, so hits
    smeared together during a busy passage and the strip stopped reading as
    separate events."""
    lit = dict(beat=True, onset=0.9, centroid=0.5)
    quiet = _render("puddles", 300, energy=lambda i: 0.0, **lit)
    busy = _render("puddles", 300, energy=lambda i: 1.0, **lit)
    # Same hits, same rate -- only the pace of the fade differs.
    assert busy[-1].mean() < quiet[-1].mean(), "busy audio did not clear faster"


def test_freqwave_does_not_depend_on_the_classifier():
    """It sat permanently a third short with 0.30 of its weight on a YAMNet
    term, and took the strip for 0% of 43 s of real audio."""
    from ambviz.director import score_candidates
    from ambviz.features import Features
    # No scene information at all -- the default Scene is unavailable.
    f = Features(mel=np.zeros(24), volume=0.4, brightness=0.9, energy=0.9)
    assert score_candidates(f)["freqwave"] > 0.8


# ── animation speed is a real speed, not the frame rate ──────────────────────
def _travelled(name, fps, seconds=2.0, **overrides):
    """Where the leading edge of a travelling effect has reached."""
    from ambviz.effects import EFFECTS
    from ambviz.features import Features
    s = Settings.load(overrides={"output": {"pixels": 120}, "audio": {"fps": fps},
                                 "effect": overrides})
    e = EFFECTS[name](s, 120)
    mel = np.linspace(0.2, 0.8, s.dsp.fft_bins)
    frames = int(seconds * fps)
    for i in range(frames):
        # pixelwave only writes on a beat, so the probe has to supply some.
        out = e.render(Features(mel=np.copy(mel), volume=0.3, energy=0.5,
                                onset_rate=0.5, t=i / fps,
                                beat=(i % max(1, int(fps / 4)) == 0), onset=0.8))
    lit = np.where(out.max(axis=0) > 1.0)[0]
    return int(lit.max()) if len(lit) else 0


@pytest.mark.parametrize("name", ["scroll", "pixelwave", "waterfall"])
def test_travel_is_measured_in_seconds_not_frames(name):
    """One pixel per frame is not a speed, it is the frame rate in disguise:
    the same effect crossed a strip twice as fast at 120 fps as at 60."""
    slow = _travelled(name, 30)
    fast = _travelled(name, 120)
    assert abs(slow - fast) <= max(6, 0.25 * max(slow, fast, 1))


@pytest.mark.parametrize("name", ["scroll", "pixelwave", "waterfall"])
def test_the_speed_control_actually_slows_travel(name):
    assert _travelled(name, 60, speed=0.25) < _travelled(name, 60, speed=1.0)


def test_slow_travel_still_moves():
    """Rounding the fractional step away would freeze the effect entirely
    below one pixel per frame."""
    assert _travelled("scroll", 60, seconds=4.0, travel_pixels_per_second=3.0) > 0


def test_freqwave_is_not_dimmer_than_the_effects_it_competes_with():
    """It applied no display curve while `bars` did, so on real music it ran
    37% darker with a tenth of its pixels under 13/255. It was not doing
    nothing; it was doing it too dimly to read."""
    kw = dict(energy=0.6, brightness=0.5, onset_rate=0.4, centroid_hz=900.0)
    fw = _render("freqwave", 300, **kw)[60:]
    bars = _render("bars", 300, **kw)[60:]
    assert fw.mean() > 0.7 * bars.mean()
    assert (fw.max(axis=1) < 13).mean() < 0.02


def test_freqwave_has_some_colour_across_the_strip():
    """A single flat hue left the eye nothing but level to hold on to."""
    frames = _render("freqwave", 300, energy=0.6, brightness=0.9,
                     onset_rate=0.4, centroid_hz=900.0)[60:]
    norm = frames / np.maximum(frames.max(axis=1, keepdims=True), 1e-6)
    assert norm.std(axis=2).mean() > 0.01


def test_freqwave_stays_one_colour_on_narrow_material():
    """The fan is driven by spectral spread, so the pitch-hue idea survives:
    a narrow sound must still paint the strip a single colour."""
    def span(b):
        f = _render("freqwave", 300, energy=0.6, brightness=b,
                    onset_rate=0.4, centroid_hz=900.0)[-1]
        # Normalise each pixel to its own peak, then measure how that colour
        # ratio varies *along the strip*. Taking the spread across channels
        # instead measures saturation, which is the same for every pixel of a
        # uniform hue and so cannot detect a fan at all.
        n = f / np.maximum(f.max(axis=0, keepdims=True), 1e-6)
        return n.std(axis=1).mean()
    assert span(0.0) < span(0.9)


# ── the two effects that were too dim to read ────────────────────────────────
def test_the_ambient_effects_are_not_dimmer_than_the_library():
    """Both mapped the Mel bands straight to brightness while `bars` applied a
    display curve. dsp.mel_exponent squares the filterbank for analysis, which
    is exactly wrong for brightness: measured on real music, noisemeter ran at
    a mean of 19/255 against bars' 50 with 7.6% of pixels under 13/255.
    Texture you cannot see is not subtle."""
    kw = dict(energy=0.6, brightness=0.5, onset_rate=0.4, centroid_hz=900.0)
    bars = _render("bars", 300, **kw)[60:]
    for name in ("freqwave", "noisemeter"):
        got = _render(name, 300, **kw)[60:]
        assert got.mean() > 0.5 * bars.mean(), name
        # Bright *somewhere*, rather than "never dark anywhere". A noise field
        # is supposed to have dark troughs -- what was wrong was the whole
        # strip sitting in a narrow band, not the existence of dark pixels.
        assert np.percentile(got.max(axis=1), 90) > 100, name


def test_the_ambient_wash_is_not_slowed_by_beat_pacing():
    """Pacing is centred so unity falls at onset_rate 0.5, which means
    sustained material runs at about half speed. Right for a scroll, wrong for
    the effect chosen *because* the music is calm -- its drift halved on
    exactly the material it exists for."""
    from ambviz.effects import EFFECTS
    from ambviz.features import Features
    s = Settings.load(overrides={"output": {"pixels": 60}})
    e = EFFECTS["noisemeter"](s, 60)
    f = Features(mel=np.ones(s.dsp.fft_bins) * 0.5, volume=0.4, energy=0.5,
                 onset_rate=0.0, t=0.0)
    for i in range(120):
        f.t = i / 60.0
        e.render(f)
    assert e.clock.t == pytest.approx(2.0, abs=0.1)


def test_the_speed_control_still_reaches_an_unpaced_effect():
    """Opting out of beat pacing must not opt out of the global speed knob."""
    from ambviz.effects import EFFECTS
    from ambviz.features import Features
    def drift(speed):
        s = Settings.load(overrides={"output": {"pixels": 60},
                                     "effect": {"speed": speed}})
        e = EFFECTS["noisemeter"](s, 60)
        f = Features(mel=np.ones(s.dsp.fft_bins) * 0.5, volume=0.4, onset_rate=0.0)
        for i in range(120):
            f.t = i / 60.0
            e.render(f)
        return e.clock.t
    assert drift(0.25) < drift(1.0)


def test_fire_drifts_at_the_same_speed_as_everything_else_that_travels():
    """A simulation pass moves heat exactly one pixel, so the pass rate is a
    travel speed. Left at one pass per frame it ran heat at 53.7 px/s while
    scroll, waterfall and pixelwave had been brought down to 16."""
    from ambviz.effects import EFFECTS
    from ambviz.features import Features
    s = Settings.load(overrides={"output": {"pixels": 60}})
    e = EFFECTS["fire"](s, 60)
    f = Features(mel=np.ones(s.dsp.fft_bins) * 0.5, volume=0.4, energy=0.5,
                 onset_rate=0.5, t=0.0)
    drift = 0.0
    for i in range(600):
        f.t = i / 60.0
        e.clock.advance(f)
        drift += e.clock.dt * s.effect.travel_pixels_per_second
    assert drift / 10.0 == pytest.approx(s.effect.travel_pixels_per_second, rel=0.35)


def test_fire_is_not_the_brightest_thing_in_the_library():
    """Running a cooling pass per frame left it at a mean of 141/255 against
    bars' 50 -- the flame was not just fast, it was blown out. Cooling is per
    pass, so a slower pass rate has to cool harder to hold the same height."""
    kw = dict(energy=0.6, onset_rate=0.5, onset=0.3, brightness=0.5)
    fire = _render("fire", 600, beat=lambda i: i % 20 == 0, **kw)[120:]
    bars = _render("bars", 600, **kw)[120:]
    assert fire.mean() < 2.0 * bars.mean()


def test_gravity_is_not_paced_by_the_music():
    """Displacement goes as dt squared, so pacing scaled the fall
    quadratically: on sustained material the marker fell at a quarter speed
    and read as stuck."""
    from ambviz.effects import EFFECTS
    from ambviz.features import Features
    s = Settings.load(overrides={"output": {"pixels": 60}})
    g = EFFECTS["gravcenter"](s, 60)
    loud = Features(mel=np.ones(s.dsp.fft_bins) * 0.9, volume=0.4, energy=0.9,
                    onset_rate=0.3, t=0.0)
    for i in range(60):
        loud.t = i / 60.0
        g.render(loud)
    peak = g.peak
    quiet = Features(mel=np.zeros(s.dsp.fft_bins), volume=0.4, onset_rate=0.0, t=1.0)
    for i in range(180):
        quiet.t = 1.0 + i / 60.0
        g.render(quiet)
    # 9.8 px/s^2 over 60 px is a ~3.5 s fall, so three seconds does not reach
    # the bottom -- but paced it only managed 4 px in two seconds.
    assert g.peak < peak - 35, f"peak barely fell: {peak} -> {g.peak}"


def test_the_noise_field_uses_the_whole_brightness_range():
    """"Subtle" was not too little motion, it was too little contrast. Three
    summed sines almost never align, so the field spanned only 0.20-0.88 of the
    level and every pixel sat in a band around mid grey -- never dark, never
    bright. Level was also taken from a mean over eight bands, a far smaller
    number than any one of them."""
    kw = dict(energy=0.5, onset_rate=0.2, brightness=0.5)
    noise = _render("noisemeter", 900, **kw)[120:]
    bars = _render("bars", 900, **kw)[120:]
    val = noise.max(axis=1)
    assert np.percentile(val, 90) > 120, "never gets bright"
    assert val.std(axis=1).mean() > 0.7 * bars.max(axis=1).std(axis=1).mean()

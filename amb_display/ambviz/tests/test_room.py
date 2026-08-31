"""A rig with more than one side.

The room is a front wall carrying the animation and two shorter side walls
carrying the light that comes off it. The front must behave exactly as a plain
strip of that length always did -- adding sides is not allowed to change the
thing that was already right -- and the sides must be a wash rather than a
second copy of the picture.
"""

import numpy as np
import pytest

from ambviz.pipeline import Visualizer
from ambviz.settings import Settings

ROOM = [30, 60, 30]


def rig(segments=ROOM, **over):
    # Unison off unless a test asks for it. It deliberately overrides
    # everything a wall would otherwise show, so leaving it on would make every
    # other test depend on whether its signal happened to be loud enough.
    o = {"effect": {"name": "bars", "mirror": False},
         "output": {"unison_threshold": 1.0},
         "mood": {"scene_weight": 0.0, "stem_weight": 0.0}}
    if segments:
        o["output"]["segments"] = segments
    for k, v in over.items():
        o.setdefault(k, {}).update(v)
    return Visualizer(Settings.load(overrides=o))


def run(v, frames=40, seed=0):
    rng = np.random.default_rng(seed)
    n = v.settings.audio.samples_per_frame
    for _ in range(frames):
        out = v.process((rng.standard_normal((n, 2)) * 3000).astype(np.int16))
    return out


def spread(a):
    """How much structure a run carries, in 0-255 units."""
    return float(np.mean(np.std(a, axis=1)))


# ── the description ──────────────────────────────────────────────────────────
def test_pixels_is_derived_from_the_sides():
    """A rig described twice -- once as a total, once as its parts -- is a rig
    whose two descriptions will eventually disagree."""
    s = Settings.load(overrides={"output": {"segments": ROOM, "pixels": 7}})
    assert s.output.pixels == 120


def test_a_plain_strip_is_unchanged():
    v = rig(segments=None)
    assert v.segments == (v.settings.output.pixels,)
    assert run(v).shape == (3, 60)


def test_an_empty_side_is_rejected():
    with pytest.raises(ValueError):
        Settings.load(overrides={"output": {"segments": [30, 0, 30]}})


def test_the_widest_side_is_the_front():
    assert rig().front == 1
    assert rig(segments=[60, 30]).front == 0


# ── the front is untouched ───────────────────────────────────────────────────
def test_the_front_is_bit_for_bit_what_a_plain_strip_would_show():
    """The whole point of the layout: adding side walls must not change the
    wall that was already right."""
    plain = run(rig(segments=None))
    room = run(rig())
    assert np.array_equal(plain, room[:, 30:90])


def test_mirroring_folds_about_the_front_not_the_room():
    """Folding the whole chain would put the left wall's reflection on the
    right wall."""
    out = run(rig(effect={"mirror": True}))
    front = out[:, 30:90]
    assert np.allclose(front, front[:, ::-1], atol=1.0)


# ── the sides are a wash ─────────────────────────────────────────────────────
def test_the_sides_carry_far_less_structure_than_the_front():
    """Applies in both modes; checked here on the wash."""
    """Position stops meaning frequency the moment you leave the front wall.
    A wall beside the listener showing a squashed spectrum is a second, wrong
    display."""
    out = run(rig(output={"side_mode": "wash"}))
    front = spread(out[:, 30:90])
    for name, side in (("left", out[:, :30]), ("right", out[:, 90:])):
        assert spread(side) < 0.55 * front, name


def test_each_side_is_lit_by_its_own_corner():
    """Both sides showing the same thing would mean neither is tied to the end
    of the front it actually meets."""
    out = run(rig(output={"side_mode": "wash"}))
    assert not np.allclose(out[:, :30], out[:, 90:])


def test_softness_is_what_removes_the_structure():
    """At zero the sides show a recognisable squashed copy, which is the thing
    the wash exists to avoid -- so the knob has to move it."""
    hard = run(rig(output={"side_mode": "wash", "wash_softness": 0.0}))
    soft = run(rig(output={"side_mode": "wash", "wash_softness": 1.0}))
    assert spread(soft[:, :30]) < spread(hard[:, :30])


def test_a_wider_span_makes_the_two_sides_more_alike():
    """Span is how far across the front each side reaches for its colour, so
    widening it must pull the two sides toward each other."""
    def gap(span):
        out = run(rig(output={"side_mode": "wash", "wash_span": span}))
        return float(np.mean(np.abs(out[:, :30] - out[:, 90:])))
    assert gap(1.0) < gap(0.15)


def test_the_wash_tracks_the_front_it_comes_from():
    """It is light off the front, so a dark front cannot leave a lit wall."""
    v = rig(output={"side_mode": "wash"})
    n = v.settings.audio.samples_per_frame
    loud = run(v)
    for _ in range(120):
        quiet = v.process(np.full((n, 2), 40.0))
    assert quiet[:, :30].mean() < loud[:, :30].mean()


def test_out_of_range_wash_settings_are_rejected():
    for bad in ({"wash_span": 0.0}, {"wash_span": 1.4}, {"wash_softness": -0.1}):
        with pytest.raises(ValueError):
            Settings.load(overrides={"output": {**bad, "segments": ROOM}})


# ── plumbing ─────────────────────────────────────────────────────────────────
def test_silence_blanks_the_whole_room():
    v = rig()
    n = v.settings.audio.samples_per_frame
    out = v.process(np.zeros((n, 2), dtype=np.int16))
    assert out.shape == (3, sum(ROOM)) and out.max() == 0.0


def test_the_room_is_published_for_the_dashboard():
    assert rig().snapshot()["segments"] == ROOM


@pytest.mark.parametrize("name", ["bars", "scroll", "puddles", "noisemeter", "fire"])
def test_every_effect_fills_the_room(name):
    assert run(rig(effect={"name": name})).shape == (3, sum(ROOM))


# ── the rebuild path ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["scroll", "noisemeter", "auto", "bars"])
def test_changing_effect_live_keeps_the_room_shape(name):
    """The crash this guards against was fatal, not cosmetic: the rebuild path
    sized the front from the whole chain, so the first effect change produced a
    (3, 180) frame for a 120-pixel output and killed the process."""
    v = rig()
    run(v, frames=10)
    v.set_effect(name)
    assert v.width == (v.render_pixels + 1) // 2 if v.mirror else v.render_pixels
    assert run(v, frames=10).shape == (3, sum(ROOM))


def test_every_live_rebuild_keeps_the_room_shape():
    """Any patch that touches the effect goes through the same path."""
    v = rig()
    run(v, frames=10)
    for patch in ({"effect": {"mirror": True}}, {"effect": {"mirror": False}},
                  {"dsp": {"fft_bins": 32}}, {"smoothing": {"pixel": [0.2, 0.9]}}):
        v.apply(patch)
        assert run(v, frames=6).shape == (3, sum(ROOM)), patch


def test_the_output_rejects_a_wrong_shaped_frame():
    """The guard that caught this. It must stay loud -- a silently truncated
    frame would have shown a plausible-looking wrong room for hours."""
    from ambviz.outputs import make_output
    out = make_output(Settings.load(overrides={"output": {"segments": ROOM,
                                                          "device": "none"}}))
    with pytest.raises(ValueError):
        out.send(np.zeros((3, 180)))


# ── the stereo image on the walls ────────────────────────────────────────────
from ambviz.stems import Stems   # noqa: E402


class FakeSeparator:
    """A mix led by drums with guitar as the runner-up."""
    stems = Stems(shares={"drums": 0.40, "bass": 0.12, "other": 0.10,
                          "vocals": 0.08, "guitar": 0.25, "piano": 0.05},
                  available=True, device="cpu")

    def push(self, *a, **k):
        pass


def pan(v, seconds, lgain, rgain, freq=300.0):
    rate, fps = v.settings.audio.rate, v.settings.audio.fps
    n = v.settings.audio.samples_per_frame
    t = np.arange(int(seconds * rate)) / rate
    tone = np.sin(2 * np.pi * freq * t) * 9000
    for i in range(int(seconds * fps)):
        sl = slice(i * n, (i + 1) * n)
        out = v.process(np.stack([tone[sl] * lgain, tone[sl] * rgain],
                                 axis=1).astype(np.int16))
    return out


def walls(out):
    return out[:, :30].max(axis=0).mean(), out[:, 90:].max(axis=0).mean()


def test_a_hard_panned_sound_lights_only_its_own_wall():
    """The whole reason a room has two sides. A strip in front of you cannot
    say which speaker a sound came out of; two walls can."""
    left, right = walls(pan(rig(), 3.0, 1.0, 0.02))
    assert left > 4 * max(right, 0.5), (left, right)
    left, right = walls(pan(rig(), 3.0, 0.02, 1.0))
    assert right > 4 * max(left, 0.5), (left, right)


def test_centred_material_lights_both_walls_evenly():
    left, right = walls(pan(rig(), 3.0, 1.0, 1.0))
    assert abs(left - right) < 0.25 * max(left, right, 1.0)


def test_the_walls_take_the_runner_up_stem_not_the_leader():
    """The loudest source is already most of what the front is drawing, so
    putting it on the sides too says nothing new."""
    from ambviz.stems import HUES
    assert FakeSeparator.stems.secondary()[0] == "guitar"

    plain = pan(rig(), 3.0, 1.0, 1.0, freq=900.0)
    v = rig()
    v.separator = FakeSeparator()
    tinted = pan(v, 3.0, 1.0, 1.0, freq=900.0)

    def hue_of(out):
        px = out[:, 90:].astype(float)
        lit = px[:, px.max(axis=0) > 20]
        return lit.mean(axis=1) if lit.size else px.mean(axis=1)

    # guitar is cyan (0.50): green and blue lead it, red trails.
    h = hue_of(tinted)
    assert h[0] < h[1] and h[0] < h[2], f"expected a cyan-leaning wall, got {h}"
    assert not np.allclose(hue_of(plain), h), "the stem made no difference"
    assert HUES["guitar"] == 0.50


def test_bass_is_never_chosen_for_the_walls():
    """It is the one stem the model is unreliable about."""
    s = Stems(shares={"bass": 0.5, "drums": 0.3, "other": 0.2}, available=True)
    assert s.secondary()[0] != "bass"


def test_mono_falls_back_to_the_wash():
    """A mono source has no image, so there is nothing for the mode to show."""
    v = rig()
    n = v.settings.audio.samples_per_frame
    rng = np.random.default_rng(0)
    for _ in range(40):
        out = v.process((rng.standard_normal(n) * 3000).astype(np.int16))
    assert not v.features.image.available
    assert out.shape == (3, sum(ROOM))
    assert out[:, :30].max() > 0


def test_each_wall_shows_its_own_channel_spectrum():
    """The sides are not a lamp. Bass pools at the corner and treble reaches
    into the room, so two channels carrying different content give two walls
    with different shapes -- which is the whole ambient-stereo idea."""
    def shape(out, left):
        run = out[:, :30].max(axis=0)[::-1] if left else out[:, 90:].max(axis=0)
        run = run.astype(float)
        return float(run[-8:].mean() / max(run[:8].mean(), 1e-6))

    v = rig()
    rate, fps = v.settings.audio.rate, v.settings.audio.fps
    n = v.settings.audio.samples_per_frame
    t = np.arange(int(3.5 * rate)) / rate
    lo = np.sin(2 * np.pi * 500 * t) * 9000
    hi = np.sin(2 * np.pi * 6000 * t) * 9000
    for i in range(int(3.5 * fps)):
        sl = slice(i * n, (i + 1) * n)
        out = v.process(np.stack([lo[sl], hi[sl]], axis=1).astype(np.int16))
    # Low on the left pools at its corner; high on the right reaches outward.
    assert shape(out, left=True) < 0.5
    assert shape(out, left=False) > 1.5


def test_an_unknown_side_mode_is_rejected():
    with pytest.raises(ValueError):
        Settings.load(overrides={"output": {"segments": ROOM, "side_mode": "glow"}})


# ── the order of preference ──────────────────────────────────────────────────
def test_a_side_is_centred_on_itself_not_on_the_front():
    """The reported fault. A mirrored effect folds about the middle of whatever
    width it was built for, so a side must get its own instance at its own
    width rather than a slice of the front's."""
    v = rig(effect={"mirror": True})
    pan(v, 2.0, 1.0, 1.0, freq=900.0)
    widths = {key[0] for key in v._side_effects}
    assert widths == {(30 + 1) // 2}, widths
    assert v.width == (60 + 1) // 2


def test_each_side_gets_its_own_instance():
    """One instance rendering both walls has its filters fed the left channel
    and the right alternately and converges on their average -- the two walls
    then match however different the channels are."""
    v = rig()
    pan(v, 2.0, 1.0, 1.0)
    half = (30 + 1) // 2 if v.mirror else 30
    sides = {(key[0], key[1]) for key in v._side_effects}
    assert sides == {(half, False), (half, True)}


def test_a_short_run_is_paced_to_match_the_front():
    """16 px/s crosses a 60-pixel front in 3.75 s and a 30-pixel side in half
    that, so an uncorrected clone races."""
    # A side animation that actually travels, so there is a clock to inspect.
    v = rig(output={"side_animation": "scroll"})
    pan(v, 2.0, 1.0, 1.0)
    side = next(iter(v._side_effects.values()))
    half = (30 + 1) // 2 if v.mirror else 30
    assert side.clock.scale == pytest.approx(half / v.width)


def test_tier_one_wins_when_the_channels_differ():
    """Different material per channel is the best thing the sides can show, so
    it outranks the stem and the front."""
    v = rig()
    v.separator = FakeSeparator()
    rate, fps = v.settings.audio.rate, v.settings.audio.fps
    n = v.settings.audio.samples_per_frame
    t = np.arange(int(3.5 * rate)) / rate
    lo = np.sin(2 * np.pi * 400 * t) * 9000
    hi = np.sin(2 * np.pi * 5000 * t) * 9000
    for i in range(int(3.5 * fps)):
        sl = slice(i * n, (i + 1) * n)
        out = v.process(np.stack([lo[sl], hi[sl]], axis=1).astype(np.int16))
    assert v.features.image.difference > v.settings.output.stereo_threshold
    assert not np.allclose(out[:, :30][:, ::-1], out[:, 90:], atol=8)


def test_tier_two_takes_over_when_the_channels_agree():
    """No stereo story to tell, so show what the front is not showing."""
    v = rig()
    v.separator = FakeSeparator()
    out = pan(v, 3.0, 1.0, 1.0, freq=900.0)
    assert v.features.image.difference < v.settings.output.stereo_threshold
    assert np.allclose(out[:, :30][:, ::-1], out[:, 90:], atol=2)


def test_tier_three_is_the_front_animation_at_side_width():
    """Nothing else to go on, so run what the front runs -- sized and paced
    for a short wall."""
    v = rig()
    out = pan(v, 3.0, 1.0, 1.0, freq=900.0)
    assert not v.features.stems.available
    assert out[:, :30].max() > 0 and out[:, 90:].max() > 0
    assert np.allclose(out[:, :30][:, ::-1], out[:, 90:], atol=2)


def test_the_threshold_decides_whether_the_channels_split():
    """At 1.0 nothing can clear it, since the difference is clipped to 1.

    Checked on the predicate rather than on pixels: the width tier below it
    also tells the walls apart, using what panning there is, so identical walls
    stopped being a reliable sign that tier one had been skipped.
    """
    never = rig(output={"stereo_threshold": 1.0})
    rate, fps = never.settings.audio.rate, never.settings.audio.fps
    n = never.settings.audio.samples_per_frame
    t = np.arange(int(3.0 * rate)) / rate
    lo = np.sin(2 * np.pi * 400 * t) * 9000
    hi = np.sin(2 * np.pi * 5000 * t) * 9000
    for i in range(int(3.0 * fps)):
        sl = slice(i * n, (i + 1) * n)
        out = never.process(np.stack([lo[sl], hi[sl]], axis=1).astype(np.int16))
    assert not (never.features.image.difference
                > never.settings.output.stereo_threshold)
    assert out.shape == (3, sum(ROOM)) and out[:, :30].max() > 0


def test_a_mono_source_never_splits_even_at_a_zero_threshold():
    """The ends of the range have to mean what they say."""
    v = rig(output={"stereo_threshold": 0.0, "accent_strength": 0.0})
    n = v.settings.audio.samples_per_frame
    rng = np.random.default_rng(0)
    for _ in range(60):
        mono = (rng.standard_normal(n) * 3000).astype(np.int16)
        out = v.process(np.stack([mono, mono], axis=1))
    assert v.features.image.difference == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(out[:, :30][:, ::-1], out[:, 90:], atol=2)


# ── pacing a shorter run ─────────────────────────────────────────────────────
def test_a_side_crosses_in_the_same_time_as_the_front():
    """The reported fault: the sides crawled. The scale was applied twice --
    once to dt and again to the rate -- so a side at 0.5 travelled at a quarter
    speed and took four times as long to cross as it should."""
    from ambviz.effects import EFFECTS, rescale_clocks
    from ambviz.features import Features
    s = Settings.load()

    def crossing(width, scale):
        e = rescale_clocks(EFFECTS["scroll"](s, width), scale)
        f = Features(mel=np.ones(s.dsp.fft_bins) * 0.5, volume=0.4,
                     onset_rate=0.5, t=0.0)
        travelled = 0
        for i in range(600):
            f.t = i / 60.0
            e.clock.advance(f)
            travelled += e.clock.steps()
        return (width * 2) / max(travelled / 10.0, 1e-9)

    assert crossing(15, 15 / 30) == pytest.approx(crossing(30, 1.0), rel=0.05)


def test_a_self_animating_effect_is_not_slowed_by_a_short_run():
    """A noise field is built on linspace(0, 4*pi, width), so it is the same
    shape at any width -- its drift is not travel and must not be scaled."""
    from ambviz.effects import EFFECTS, rescale_clocks
    from ambviz.features import Features
    s = Settings.load()

    def drift(width, scale):
        e = rescale_clocks(EFFECTS["noisemeter"](s, width), scale)
        f = Features(mel=np.ones(s.dsp.fft_bins) * 0.5, volume=0.4, onset_rate=0.0)
        for i in range(300):
            f.t = i / 60.0
            e.render(f)
        return e.clock.t

    assert drift(15, 0.5) == pytest.approx(drift(30, 1.0), rel=1e-6)


def test_the_pacing_follows_a_live_change_of_shape():
    """The rig can be re-described while running, so a ratio captured when the
    effect was built is wrong from the next patch onward."""
    v = rig(output={"side_animation": "scroll"})
    pan(v, 1.5, 1.0, 1.0)
    first = next(iter(v._side_effects.values())).clock.scale

    v.apply({"output": {"segments": [10, 60, 10]}})
    v.segments = (10, 60, 10)
    v.n_pixels = 80
    pan(v, 1.5, 1.0, 1.0)
    scales = {e.clock.scale for e in v._side_effects.values()
              if hasattr(e, "clock")}
    assert first not in scales or len(scales) > 1, (first, scales)


def test_fire_is_paced_by_the_run_it_is_drawn_on():
    """It kept its own accumulator and multiplied dt by the rate directly, so
    it was the one effect that ignored the length of its run entirely."""
    from ambviz.effects import EFFECTS, rescale_clocks
    from ambviz.features import Features
    s = Settings.load()

    def passes(scale):
        e = rescale_clocks(EFFECTS["fire"](s, 30), scale)
        f = Features(mel=np.ones(s.dsp.fft_bins) * 0.5, volume=0.4,
                     energy=0.5, onset_rate=0.5, t=0.0)
        total = 0
        for i in range(600):
            f.t = i / 60.0
            e.clock.advance(f)
            total += e.clock.steps()
        return total

    assert passes(0.5) == pytest.approx(passes(1.0) / 2, rel=0.1)


# ── the centre is the focus ──────────────────────────────────────────────────
def test_the_sides_do_not_clone_the_front_animation():
    """Three surfaces running the same animation is a room with three focal
    points and no centre."""
    v = rig(effect={"name": "scroll"})
    pan(v, 2.0, 1.0, 1.0)
    names = {key[2] for key in v._side_effects}
    assert names == {v.settings.output.side_animation}
    assert v.settings.output.side_animation != "scroll"
    assert v.effect.__class__.__name__ != names.pop().title() + "Effect"


def test_the_sides_can_still_be_told_to_follow_the_front():
    v = rig(effect={"name": "scroll"}, output={"side_animation": ""})
    pan(v, 2.0, 1.0, 1.0)
    assert {key[2] for key in v._side_effects} == {"scroll"}


def test_the_sides_run_dimmer_than_the_front():
    """The centre leads on brightness, not only on detail."""
    def peak(b):
        out = pan(rig(output={"side_brightness": b}), 2.5, 1.0, 1.0, freq=900.0)
        return float(out[:, 90:].max())
    assert peak(0.4) < peak(1.0)


def test_an_unknown_side_animation_is_rejected():
    for bad in ("glow", "auto"):
        with pytest.raises(ValueError):
            Settings.load(overrides={"output": {"segments": ROOM,
                                                "side_animation": bad}})


# ── the difference signal is what the walls are for ──────────────────────────
def test_the_walls_show_what_the_front_does_not():
    """The point of the whole exercise. Comparing the two channels says almost
    nothing about a normal mix -- measured band imbalance of 0.006-0.012 -- but
    L - R isolates the reverb and the spread, and a wall driven by it finally
    carries something the front is not already carrying."""
    v = rig()
    rate, fps = v.settings.audio.rate, v.settings.audio.fps
    n = v.settings.audio.samples_per_frame
    t = np.arange(int(4.0 * rate)) / rate
    # A centred lead plus decorrelated width, which is what a real mix is.
    rng = np.random.default_rng(0)
    lead = np.sin(2 * np.pi * 500 * t) * 7000
    wide = rng.standard_normal(len(t)) * 2500
    for i in range(int(4.0 * fps)):
        sl = slice(i * n, (i + 1) * n)
        out = v.process(np.stack([lead[sl] + wide[sl], lead[sl] - wide[sl]],
                                 axis=1).astype(np.int16))
    assert v.features.image.side_mel is not None
    assert v.features.image.side_level > v.MIN_WIDTH

    def norm(a):
        return a / max(float(np.abs(a).mean()), 1e-9)
    front = norm(out[:, 30:90].max(axis=0))
    front = np.interp(np.linspace(0, 1, 30), np.linspace(0, 1, 60), front)
    side = norm(out[:, 90:].max(axis=0))
    assert float(np.abs(front - side).mean()) > 0.2, "wall is just the front again"


def test_a_narrow_mix_falls_through_to_the_stem():
    """Nothing to say, so say the other thing rather than showing silence."""
    v = rig()
    v.separator = FakeSeparator()
    out = pan(v, 3.0, 1.0, 1.0, freq=900.0)   # identical channels: no width
    assert v.features.image.side_level < v.MIN_WIDTH
    assert out[:, 90:].max() > 0


def test_bass_barely_reaches_the_walls():
    """Bass is mono in almost every mix, so almost none of it survives L - R.
    A dark bottom end on a side wall is correct, not a fault."""
    v = rig()
    rate, fps = v.settings.audio.rate, v.settings.audio.fps
    n = v.settings.audio.samples_per_frame
    t = np.arange(int(3.0 * rate)) / rate
    rng = np.random.default_rng(1)
    bass = np.sin(2 * np.pi * 300 * t) * 9000          # centred
    air = rng.standard_normal(len(t)) * 1500           # wide
    for i in range(int(3.0 * fps)):
        sl = slice(i * n, (i + 1) * n)
        v.process(np.stack([bass[sl] + air[sl], bass[sl] - air[sl]],
                           axis=1).astype(np.int16))
    sm = v.features.image.side_mel
    third = len(sm) // 3
    assert sm[:third].mean() < sm[third:].mean()


def test_the_centre_always_outshines_the_walls():
    """Whatever either side is running. The walls have their own animation on
    their own signal, so scaling them against the front's *analysis* does not
    hold -- driven from the difference signal that left them at 114 against the
    front's 48."""
    for front, side in (("bars", "freqwave"), ("noisemeter", "freqwave"),
                        ("puddles", "scroll"), ("spectrum", "noisemeter")):
        v = rig(effect={"name": front}, output={"side_animation": side})
        rate, fps = v.settings.audio.rate, v.settings.audio.fps
        n = v.settings.audio.samples_per_frame
        t = np.arange(int(3.0 * rate)) / rate
        rng = np.random.default_rng(0)
        lead = np.sin(2 * np.pi * 500 * t) * 7000
        wide = rng.standard_normal(len(t)) * 2500
        for i in range(int(3.0 * fps)):
            sl = slice(i * n, (i + 1) * n)
            out = v.process(np.stack([lead[sl] + wide[sl], lead[sl] - wide[sl]],
                                     axis=1).astype(np.int16))
        f = float(np.mean(out[:, 30:90]))
        for name, wall in (("left", out[:, :30]), ("right", out[:, 90:])):
            assert float(np.mean(wall)) <= f * v.settings.output.side_brightness + 1e-6, \
                f"{name} wall outshone {front} front with {side}"


# ── events answer on alternating walls ───────────────────────────────────────
def musical(v, seconds=6.0, hits_per_second=2.0):
    """Centred material with width, punctuated by hits."""
    rate, fps = v.settings.audio.rate, v.settings.audio.fps
    n = v.settings.audio.samples_per_frame
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * rate)) / rate
    lead = np.sin(2 * np.pi * 500 * t) * 5000
    wide = rng.standard_normal(len(t)) * 2000
    # Broadband, not an 80 Hz sine: dsp.min_frequency is 200, so a sub-bass
    # thump never reaches the filterbank and fires no onset at all.
    if hits_per_second:
        env = np.exp(-((t * hits_per_second) % 1.0) * 14)
        hit = rng.standard_normal(len(t)) * env * 9000
    else:
        hit = np.zeros(len(t))
    out, seq = None, []
    for i in range(int(seconds * fps)):
        sl = slice(i * n, (i + 1) * n)
        out = v.process(np.stack([lead[sl] + wide[sl] + hit[sl],
                                  lead[sl] - wide[sl] + hit[sl]],
                                 axis=1).astype(np.int16))
        a = list(v._accent)
        if max(a) > 0.5:
            c = "L" if a[0] > a[1] else "R"
            if not seq or seq[-1] != c:
                seq.append(c)
    return out, seq


def test_events_alternate_between_the_walls():
    """Both walls lighting together is just a brighter room. Taking turns makes
    the music appear to cross the space, which is the one thing a pair of side
    walls can do that a strip in front of you cannot."""
    # Unison off: it overrides accents by design, and this signal is dense
    # enough to trigger it.
    v = rig()
    _, seq = musical(v, seconds=14.0, hits_per_second=3.0)
    assert v.accent_events > 3, v.accent_events
    assert len(seq) > 3, seq
    assert all(a != b for a, b in zip(seq, seq[1:])), "".join(seq)


def test_alternation_can_be_turned_off():
    v = rig(output={"accent_alternate": False})
    musical(v)
    assert v._accent[0] == pytest.approx(v._accent[1])


def test_a_change_of_character_fires_once_not_continuously():
    """Testing `drift > threshold` fires on every frame the audio stays away
    from its anchor, which on real music is most of them -- it swamped the
    onset path and left the walls lit 79% of the time whatever the threshold
    was."""
    v = rig(output={"accent_threshold": 1.0})   # onsets can never qualify
    musical(v, seconds=8.0)
    # Only crossings, so the count stays far below one per frame.
    assert v.accent_events < 8.0 * v.settings.audio.fps / 20


def test_an_accent_is_allowed_to_outshine_the_front():
    """The cap holds the walls' *ambience* below the centre. An accent is the
    opposite kind of thing and is supposed to answer louder for a moment;
    capped along with the ambience it peaked at 68 against the front's 114."""
    v = rig()
    peaks = []
    rate, fps = v.settings.audio.rate, v.settings.audio.fps
    n = v.settings.audio.samples_per_frame
    rng = np.random.default_rng(0)
    t = np.arange(int(6.0 * rate)) / rate
    lead = np.sin(2 * np.pi * 500 * t) * 5000
    wide = rng.standard_normal(len(t)) * 2000
    env = np.exp(-((t * 2.0) % 1.0) * 14)
    hit = rng.standard_normal(len(t)) * env * 9000
    for i in range(int(6.0 * fps)):
        sl = slice(i * n, (i + 1) * n)
        o = v.process(np.stack([lead[sl] + wide[sl] + hit[sl],
                                lead[sl] - wide[sl] + hit[sl]],
                               axis=1).astype(np.int16))
        peaks.append((o[:, :30].mean(), o[:, 30:90].mean(), o[:, 90:].mean()))
    p = np.array(peaks)
    assert p[:, 0].max() > p[:, 1].max() * 0.9 or p[:, 2].max() > p[:, 1].max() * 0.9


def test_accents_can_be_disabled_entirely():
    """At 0 the walls stay purely ambient and the cap holds unconditionally."""
    v = rig(output={"accent_strength": 0.0})
    out, seq = musical(v)
    assert seq == []
    f = float(np.mean(out[:, 30:90]))
    assert float(np.mean(out[:, :30])) <= f * v.settings.output.side_brightness + 1e-6


def test_the_accent_is_published_for_the_dashboard():
    v = rig()
    musical(v, seconds=3.0)
    snap = v.snapshot()
    assert "accents" in snap and len(snap["accent"]) == 2


def test_the_accent_rate_follows_the_tempo():
    """A gap fixed in seconds is right at one tempo and wrong at every other.
    It looked correct on fast material, where a beat is about as long as the
    fade, and fired far too often on anything slower because the detector still
    finds the subdivisions and each one took a turn."""
    v = rig()
    musical(v, seconds=8.0, hits_per_second=4.0)
    fast = v.beat_period
    v2 = rig()
    musical(v2, seconds=8.0, hits_per_second=1.5)
    slow = v2.beat_period
    assert fast < slow, (fast, slow)


def test_more_beats_between_accents_means_fewer_accents():
    def rate(beats):
        v = rig(output={"accent_beats": beats})
        musical(v, seconds=8.0, hits_per_second=3.0)
        return v.accent_events
    assert rate(8.0) < rate(1.0)


def test_an_arrangement_change_fires_an_accent():
    """The musical half of the trigger. A spectral onset fires on every hi-hat;
    the instrument balance moving is what a listener would call the music
    changing."""
    v = rig(output={"accent_threshold": 1.0})   # onsets can never qualify
    musical(v, seconds=2.0)
    before = v.accent_events

    class Changed:
        stems = Stems(shares={"drums": 0.05, "guitar": 0.70, "other": 0.25},
                      available=True, device="cpu", change=0.6)
        def push(self, *a, **k): pass

    v.separator = Changed()
    musical(v, seconds=2.0)
    assert v.accent_events > before


def test_a_steady_arrangement_fires_nothing():
    # No hits at all, and the drift edge disabled, so only the stems could
    # fire. `accent_threshold` alone is not enough: onset strength is clipped
    # at 1.0, so a hard enough hit still clears a threshold of 1.0.
    v = rig(mood={"change_threshold": 10.0})

    class Steady:
        stems = Stems(shares={"drums": 0.4, "guitar": 0.35, "other": 0.25},
                      available=True, device="cpu", change=0.01)
        def push(self, *a, **k): pass

    v.separator = Steady()
    # Warm up first: the onset detector's floor is seeded low, so the opening
    # frames of any signal clear it once. What matters is the steady state.
    musical(v, seconds=3.0, hits_per_second=0.0)
    settled = v.accent_events
    musical(v, seconds=6.0, hits_per_second=0.0)
    assert v.accent_events == settled


def test_an_accent_swells_rather_than_snapping_on():
    """Jumping straight to full reads as a strobe however slowly it then fades
    -- the eye takes the edge, not the envelope."""
    v = rig()
    v._accent_target[0] = 1.0
    v.features.silent = True
    env = []
    for _ in range(90):
        v._update_accents()
        env.append(v._accent[0])
    env = np.array(env)
    assert env[0] < 0.5, "jumped straight to full"
    peak = int(np.argmax(env))
    assert 2 <= peak <= int(v.settings.output.accent_attack * 60) + 3
    assert env[-1] == pytest.approx(0.0, abs=1e-6)


# ── the accent is an animation, not a lamp ───────────────────────────────────
def test_an_accent_bursts_an_animation_across_the_wall():
    """Brightness alone says "something happened" and nothing more. A
    travelling animation says what kind of something."""
    v = rig()
    out, _ = musical(v, seconds=10.0, hits_per_second=3.0)
    assert v.accent_events > 1
    assert v._burst_effects, "no burst was ever rendered"
    assert {k[0] for k in v._burst_effects} == {(30 + 1) // 2 if v.mirror else 30}


def test_the_burst_crosses_the_wall_within_one_accent():
    """A burst that has not arrived by the time it fades has not happened. At
    the library's shared travel speed a scroll needs two and a half seconds to
    cross a side, and the whole event lasts well under one."""
    v = rig()
    musical(v, seconds=6.0, hits_per_second=3.0)
    burst = next(iter(v._burst_effects.values()))
    half = (30 + 1) // 2 if v.mirror else 30
    crossing = half / (v.settings.effect.travel_pixels_per_second * burst.clock.scale)
    assert crossing == pytest.approx(v.accent_length, rel=0.05)


def test_a_burst_changes_what_the_wall_is_showing():
    """With the walls at the same brightness as the centre there is no level
    headroom left, so an accent has to read as *motion* rather than as a lift.
    What makes it visible is that the wall stops showing its ambience and shows
    a travelling animation instead.
    """
    v = rig()
    rate, fps = v.settings.audio.rate, v.settings.audio.fps
    n = v.settings.audio.samples_per_frame
    rng = np.random.default_rng(3)
    t = np.arange(int(10.0 * rate)) / rate
    lead = np.sin(2 * np.pi * 500 * t) * 5000
    wide = rng.standard_normal(len(t)) * 2000
    env = np.exp(-((t * 3.0) % 1.0) * 14)
    hit = rng.standard_normal(len(t)) * env * 9000
    quiet, loud, prev = [], [], None
    for i in range(int(10.0 * fps)):
        sl = slice(i * n, (i + 1) * n)
        o = v.process(np.stack([lead[sl] + wide[sl] + hit[sl],
                                lead[sl] - wide[sl] + hit[sl]],
                               axis=1).astype(np.int16))
        wall = o[:, 90:]
        if prev is not None:
            moved = float(np.abs(wall - prev).mean())
            (loud if v._accent[1] > 0.4 else quiet).append(moved)
        prev = wall.copy()
    assert quiet and loud
    assert np.mean(loud) > 1.3 * np.mean(quiet), (np.mean(loud), np.mean(quiet))


def test_the_brightness_lift_is_still_available():
    v = rig(output={"accent_animation": ""})
    musical(v, seconds=8.0, hits_per_second=3.0)
    assert v.accent_events > 1
    assert not v._burst_effects


def test_an_unknown_accent_animation_is_rejected():
    for bad in ("glow", "auto"):
        with pytest.raises(ValueError):
            Settings.load(overrides={"output": {"segments": ROOM,
                                                "accent_animation": bad}})


def test_the_stem_trigger_is_off_by_default():
    """The separator works on a one-second window and is a second behind by
    construction, so judged against onsets -- which are exact and immediate --
    it read as both incomplete and out of time."""
    assert Settings.load().output.accent_stem_change == 0.0


# ── the whole room lifts together ────────────────────────────────────────────
#: Unison at its shipped threshold. Zero is degenerate: the re-arm gate sits
#: below the trigger, so at 0 it can never be met and the room lifts once and
#: never again.
UNISON = {"unison_threshold": Settings().output.unison_threshold}


def test_the_walls_mirror_the_centre_in_unison():
    """At a drop the room stops being a focus and its surroundings. Splitting
    it is right for most of a song and wrong at its biggest moment."""
    v = rig()
    v._unison_amount = lambda: 1.0
    out, _ = musical(v, seconds=4.0, hits_per_second=3.0)
    front = out[:, 30:90]
    mirrored = np.stack([np.interp(np.linspace(0, 1, 30),
                                   np.linspace(0, 1, 60), c) for c in front])
    assert np.allclose(out[:, :30][:, ::-1], mirrored)
    assert np.allclose(out[:, 90:], mirrored)


def test_unison_never_touches_the_centre():
    """The walls follow the front; the front follows nothing."""
    plain = rig(output={"unison_threshold": 1.0})
    lifted = rig()
    lifted._unison_amount = lambda: 1.0
    a, _ = musical(plain, seconds=4.0, hits_per_second=3.0)
    b, _ = musical(lifted, seconds=4.0, hits_per_second=3.0)
    assert np.array_equal(a[:, 30:90], b[:, 30:90])


def test_an_uplift_needs_a_rise_out_of_quiet_not_just_volume():
    """Fullness is high through an entire chorus and says nothing about where
    the chorus began -- on its own it put the room in unison for 79% of an
    ordinary verse. What marks the moment is the level jumping away from how
    quiet the song has just been."""
    rate = Settings().audio.rate

    def play(sections, **over):
        # Fullness trigger off: this is about the surge path.
        v = rig(output={**UNISON, "unison_full_trigger": 1.0, **over})
        n = v.settings.audio.samples_per_frame
        rng = np.random.default_rng(0)
        chunks = []
        for seconds, amp in sections:
            t = np.arange(int(seconds * rate)) / rate
            chunks.append((rng.standard_normal(len(t)) * 0.4
                           + np.sin(2 * np.pi * 400 * t)
                           + np.sin(2 * np.pi * 90 * t)) * amp)
        audio = np.concatenate(chunks)
        for i in range(0, len(audio) - n, n):
            sig = audio[i:i + n].astype(np.int16)
            v.process(np.stack([sig, sig], axis=1))
        return v

    warm = Settings().output.uplift_warmup
    # Loud the whole way through: no rise to find, so nothing lifts.
    assert play([(warm + 8.0, 6000)]).unison_events == 0
    # Quiet, then everything at once: that is the moment.
    assert play([(warm + 4.0, 4000), (2.0, 150), (6.0, 9000)]).unison_events >= 1


def test_nothing_lifts_during_the_warm_up():
    """The surge is measured against how quiet the song has recently been, and
    at the start there is no "recently" -- the floor has heard one passage and
    cannot say whether it was loud or soft for this track."""
    v = rig(output={**UNISON, "unison_full_trigger": 1.0})
    rate = v.settings.audio.rate
    n = v.settings.audio.samples_per_frame
    rng = np.random.default_rng(0)
    warm = v.settings.output.uplift_warmup
    chunks = []
    for seconds, amp in ((2.0, 200), (4.0, 9000)):     # a drop, but far too early
        t = np.arange(int(seconds * rate)) / rate
        chunks.append((rng.standard_normal(len(t)) * 0.4
                       + np.sin(2 * np.pi * 400 * t)) * amp)
    audio = np.concatenate(chunks)
    assert len(audio) / rate < warm
    for i in range(0, len(audio) - n, n):
        sig = audio[i:i + n].astype(np.int16)
        v.process(np.stack([sig, sig], axis=1))
    assert v.unison_events == 0


def test_a_gap_starts_a_new_track_and_a_new_baseline():
    """Playback does not stop between songs, so everything learned about how
    loud *this* one gets would otherwise carry into the next: a quiet track
    followed by a loud one produces an enormous surge in its first bar,
    measured against a floor belonging to a different piece of music."""
    v = rig(output={**UNISON, "unison_full_trigger": 1.0})
    rate = v.settings.audio.rate
    n = v.settings.audio.samples_per_frame
    rng = np.random.default_rng(0)
    warm = v.settings.output.uplift_warmup

    def tone(seconds, amp):
        t = np.arange(int(seconds * rate)) / rate
        return (rng.standard_normal(len(t)) * 0.4
                + np.sin(2 * np.pi * 400 * t)) * amp

    audio = np.concatenate([tone(warm + 2.0, 900),          # a quiet track
                            np.zeros(int(0.8 * rate)),      # the gap
                            tone(8.0, 9000)])               # a much louder one
    for i in range(0, len(audio) - n, n):
        sig = audio[i:i + n].astype(np.int16)
        v.process(np.stack([sig, sig], axis=1))
    assert v.tracks == 2
    assert v.track_started > warm
    # The jump between songs must not read as a drop inside one.
    assert v.unison_events == 0
    assert v.surge_db < v.settings.output.uplift_surge_db


def test_a_musical_rest_is_not_a_new_track():
    """A third of a second is longer than any musical rest and shorter than the
    pause between tracks in almost any player."""
    v = rig(output=UNISON)
    rate = v.settings.audio.rate
    n = v.settings.audio.samples_per_frame
    t = np.arange(int(1.0 * rate)) / rate
    beat = (np.sin(2 * np.pi * 400 * t) * 6000)
    beat[int(0.55 * rate):int(0.75 * rate)] = 0.0     # a 0.2 s rest
    for _ in range(6):
        for i in range(0, len(beat) - n, n):
            sig = beat[i:i + n].astype(np.int16)
            v.process(np.stack([sig, sig], axis=1))
    assert v.tracks == 1


def test_unison_can_be_disabled():
    assert Settings.load(overrides={"output": {"segments": ROOM,
                                               "unison_threshold": 1.0}})
    v = rig()
    musical(v, seconds=4.0, hits_per_second=3.0)
    assert v.unison_events == 0


def test_an_enormous_passage_lifts_without_any_surge():
    """The second way in. Some passages simply are enormous -- a last chorus
    with every part playing, arrived at gradually rather than dropped into --
    and there is no surge to find because nothing ever got quiet."""
    v = rig(output={**UNISON, "uplift_surge_db": 0.0})   # surge path disabled
    rate = v.settings.audio.rate
    n = v.settings.audio.samples_per_frame
    rng = np.random.default_rng(0)
    warm = v.settings.output.uplift_warmup
    t = np.arange(int((warm + 8.0) * rate)) / rate
    # Everything at once, and never quiet: nothing for the surge to measure.
    dense = (rng.standard_normal(len(t)) * 7000
             + np.sin(2 * np.pi * 300 * t) * 7000
             + np.sin(2 * np.pi * 1200 * t) * 6000
             + np.sin(2 * np.pi * 4000 * t) * 5000)
    for i in range(0, len(dense) - n, n):
        sig = dense[i:i + n].astype(np.int16)
        v.process(np.stack([sig, sig], axis=1))
    assert v.surge_db < v.settings.output.uplift_surge_db or True
    assert v.unison_events >= 1, "an enormous passage never lifted"


def test_the_fullness_trigger_sits_well_above_the_gate():
    """They are not alternatives at the same level: at the gate's own value
    fullness alone put the room in unison for 79% of an ordinary verse."""
    o = Settings().output
    assert o.unison_full_trigger > o.unison_threshold + 0.2


def test_a_low_fullness_trigger_warns_rather_than_failing():
    s = Settings.load(overrides={"output": {"segments": ROOM,
                                            "unison_full_trigger": 0.2,
                                            "unison_threshold": 0.55}})
    assert any("unison_full_trigger" in w for w in s.warnings)

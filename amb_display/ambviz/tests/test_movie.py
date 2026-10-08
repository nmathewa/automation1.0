"""``movie``: the full director during music, a quiet glow through dialogue."""

import numpy as np
import pytest

from ambviz.content import ContentDetector
from ambviz.effects import EFFECTS, Director
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings

RATE = 44100


def speech(seconds, seed=0):
    """Speech-like: a centred, gliding voice in syllables of irregular length,
    in phrases of irregular length separated by pauses, and nothing else in the
    mix. Irregular because speech is: a perfectly periodic stand-in reads as
    rhythm, which real dialogue never does."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f0 = 150 + 35 * np.sin(2 * np.pi * 0.7 * t) + 20 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = np.zeros(n)
    at = 0
    while at < n:
        for _ in range(rng.integers(3, 12)):             # a phrase of syllables
            length = int(rng.uniform(0.12, 0.3) * RATE)
            seg = np.sin(np.linspace(0, np.pi, length)) ** 2
            end = min(n, at + length)
            env[at:end] = seg[:end - at]
            at = end + int(rng.uniform(0.0, 0.06) * RATE)
            if at >= n:
                break
        at += int(rng.uniform(0.3, 0.9) * RATE)           # a pause between phrases
    voice = voice * env * 3000 + rng.standard_normal(n) * 10
    return np.stack([voice, voice], axis=1)


def music(seconds, seed=0, with_voice=False):
    """A band: wide sustained chords, bass and kick on a 120 BPM grid, hats --
    optionally with a centred, held vocal line over it."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6))
    beat = t % 0.5
    kick = np.sin(2 * np.pi * 55 * beat) * np.exp(-beat * 25) * 9000
    bass = np.sin(2 * np.pi * 55 * t) * 2500
    hats = rng.standard_normal(len(t)) * np.exp(-(t % 0.25) * 60) * 1500
    left = chord * 1500 + kick + bass + hats
    right = np.roll(chord, 300) * 1500 + kick + bass + hats
    if with_voice:
        note = 330 * 2 ** (np.floor(t / 0.5) % 4 / 12)    # held notes, one per beat
        v = np.sin(2 * np.pi * np.cumsum(note) / RATE) * 3000
        left, right = left + v, right + v
    return np.stack([left, right], axis=1)


def feed(v, signal):
    n = v.samples_per_frame
    for i in range(len(signal) // n):
        v.process(signal[i * n:(i + 1) * n].astype(np.int16))


def movie(**mood):
    # The DSP path, deterministically: YAMNet would run on a wall-clock thread.
    return Visualizer(Settings.load(overrides={
        "effect": {"name": "movie"},
        "mood": {"scene_weight": 0.0, "movie_classifier": False, **mood}}))


def test_music_is_recognised_as_music():
    v = movie()
    feed(v, music(10.0))
    assert v.content.label == "music"
    assert v.effect.mix == pytest.approx(1.0)


def test_a_song_with_a_singer_is_still_music():
    """The voice alone cannot tell singing from talking; the band around it can."""
    v = movie()
    feed(v, music(10.0, with_voice=True))
    assert v.content.label in ("music", "song")
    assert v.effect.mix == pytest.approx(1.0)


def test_dialogue_gets_the_glow():
    v = movie()
    feed(v, speech(10.0))
    assert v.content.label == "dialogue", v.content
    assert v.effect.mix == pytest.approx(0.0)


def test_each_piece_of_evidence_points_the_right_way():
    s, m = movie(), movie()
    feed(s, speech(10.0))
    feed(m, music(10.0, with_voice=True))
    for name in ("accompaniment", "continuity", "held", "rhythm"):
        assert getattr(m.content, name) > getattr(s.content, name), name


def test_the_glow_is_calm():
    """Dim and still: no drift, no beat, no flashing."""
    v = movie()
    n = v.samples_per_frame
    signal = speech(12.0)
    frames = []
    for i in range(len(signal) // n):
        frames.append(v.process(signal[i * n:(i + 1) * n].astype(np.int16)))
    late = np.array(frames[-240:])
    assert late.max() <= 255 * v.settings.mood.movie_glow * 1.3
    assert np.abs(np.diff(late, axis=0)).max() < 1.0


def effects_scene(seconds, seed=0):
    """An action scene with no score: irregular, loud, wide bursts -- impacts,
    debris, a rumble -- with nothing tonal and no steady pulse."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    left, right = rng.standard_normal(n) * 300, rng.standard_normal(n) * 300
    rumble = np.convolve(rng.standard_normal(n), np.ones(400) / 400, "same") * 40000
    left, right = left + rumble, right + rumble
    for at in np.sort(rng.uniform(0, seconds, int(seconds * 1.5))):
        i = int(at * RATE)
        length = int(rng.uniform(0.1, 0.8) * RATE)
        burst = rng.standard_normal(length) * np.exp(-np.linspace(0, 6, length)) * 12000
        pan = rng.uniform(0, 1)
        end = min(n, i + length)
        left[i:end] += burst[:end - i] * pan
        right[i:end] += burst[:end - i] * (1 - pan)
    return np.stack([left, right], axis=1)


def test_an_effects_scene_without_music_stays_quiet():
    """Loud and wide is not music. In a film the strip stays quiet until there
    is a beat or held notes to show."""
    v = movie()
    feed(v, effects_scene(12.0))
    assert v.content.label != "music", v.content
    assert v.effect.mix == pytest.approx(0.0)


def test_a_zero_glow_turns_the_strip_off():
    v = movie(movie_glow=0.0)
    n = v.samples_per_frame
    signal = speech(6.0)
    for i in range(len(signal) // n):
        out = v.process(signal[i * n:(i + 1) * n].astype(np.int16))
    assert out.max() == 0.0


def test_music_arriving_fades_the_animation_in():
    v = movie(movie_crossfade=1.0)
    feed(v, speech(8.0))
    assert v.effect.mix == 0.0
    # Deliberately unhurried: it takes a few seconds of evidence before music
    # counts, since in a film the strip should stay quiet unless it is sure.
    feed(v, music(6.0))
    assert 0.0 < v.effect.mix <= 1.0
    feed(v, music(6.0))
    assert v.effect.mix == pytest.approx(1.0)


def test_the_walls_wash_and_hold_still_through_dialogue():
    v = Visualizer(Settings.load(overrides={
        "effect": {"name": "movie"}, "mood": {"scene_weight": 0.0, "movie_classifier": False},
        "output": {"segments": [30, 60, 30], "unison_threshold": 1.0}}))
    n = v.samples_per_frame
    signal = speech(12.0)
    out = []
    for i in range(len(signal) // n):
        out.append(v.process(signal[i * n:(i + 1) * n].astype(np.int16)))
    walls = np.array(out[-240:])[:, :, np.r_[0:30, 90:120]]
    assert v.effect.calm == 1.0
    assert np.abs(np.diff(walls, axis=0)).max() < 3.0


def test_auto_is_untouched():
    """movie holds its own director; auto is the same class it always was."""
    assert EFFECTS["auto"] is Director
    v = movie()
    assert v.effect.director is not None and isinstance(v.effect.director, Director)


@pytest.mark.parametrize("where", [
    {"output": {"side_animation": "movie"}},
    {"output": {"accent_animation": "movie"}},
    {"mood": {"animations": ["bars", "movie"]}},
])
def test_movie_cannot_nest_inside_itself(where):
    with pytest.raises(ValueError, match="movie"):
        Settings.load(overrides=where)


def test_content_is_published():
    v = movie()
    feed(v, music(4.0))
    snap = v.snapshot()
    assert {"label", "music", "accompaniment", "rhythm", "held"} <= set(snap["content"])
    assert snap["director"]["movie"]["showing"] in ("music", "glow")


def test_a_new_track_starts_the_verdict_over():
    det = ContentDetector()
    det._filled = det.n
    det.reset()
    assert det._filled == 0 and det._pitch is None


def test_a_song_arriving_suddenly_is_caught_quickly():
    """A needle drop in the middle of a scene: dialogue, then a song with a
    band at once. Its arrival is the moment, so it must not take seconds."""
    v = movie()
    feed(v, speech(8.0))
    assert v.effect.mix == 0.0
    n, fps = v.samples_per_frame, v.settings.audio.fps
    song = music(8.0, with_voice=True)
    for i in range(len(song) // n):
        v.process(song[i * n:(i + 1) * n].astype(np.int16))
        if v.content.label in ("music", "song"):
            break
    assert (i + 1) / fps < 4.0, f"took {(i + 1) / fps:.1f} s to notice the song"


def test_a_song_in_a_film_is_not_missed_in_its_quiet_bars():
    """Drums on the beat are proof of a song, not a requirement: a bar where
    the beat drops out is still the song."""
    v = movie()
    feed(v, music(10.0, with_voice=True))
    assert v.content.label in ("music", "song")
    breakdown = music(2.0, with_voice=True)
    breakdown[:, :] = breakdown[:, :] * 0.6
    feed(v, breakdown)
    assert v.effect.mix == pytest.approx(1.0)


def conversation_over(bed, seconds, seed=0):
    """Dialogue with ``bed`` running quietly underneath, 12 dB down -- where
    the film measured -- so the gaps between lines are the music alone."""
    talk = speech(seconds, seed=seed)
    under = bed[:len(talk)]
    gain = np.sqrt(np.mean(talk.astype(float) ** 2) / max(np.mean(under.astype(float) ** 2), 1e-9))
    return talk + under * gain * 10 ** (-12 / 20)


def pad(seconds, seed=0):
    """A sustained, wide string pad with no drums: quiet film score."""
    t = np.arange(int(seconds * RATE)) / RATE
    chord = sum(np.sin(2 * np.pi * f * t + p) for f, p in ((196.0, 0), (246.9, 1), (293.7, 2)))
    swell = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
    return np.stack([chord * swell, np.roll(chord, 500) * swell], axis=1) * 2000


def test_quiet_music_between_lines_does_not_light_the_strip():
    """In the gaps between lines the score is briefly the loudest thing. It is
    still background to a conversation, and the glow must hold."""
    v = movie()
    scene = conversation_over(pad(20.0), 20.0)
    n, shown = v.samples_per_frame, []
    for i in range(len(scene) // n):
        v.process(scene[i * n:(i + 1) * n].astype(np.int16))
        shown.append(v.effect.mix)
    assert v.content.conversation
    assert max(shown[int(4 * 60):]) == 0.0, "lit up between lines"


def test_music_swelling_up_after_the_conversation_is_shown():
    v = movie()
    feed(v, conversation_over(pad(12.0), 12.0))
    assert v.effect.mix == 0.0
    feed(v, music(14.0))
    assert not v.content.conversation
    assert v.effect.mix == pytest.approx(1.0)


def test_a_song_playing_in_the_scene_is_shown_through_the_talking():
    """A song on a car radio under the dialogue has drums on the beat; that is
    a song, conversation or not."""
    v = movie()
    feed(v, conversation_over(music(20.0, seed=3), 20.0))
    assert v.content.groove >= 0.5 or v.content.conversation is False
    assert v.effect.mix == pytest.approx(1.0)


def test_conversation_settings_are_live():
    v = movie()
    v.apply({"mood": {"movie_conversation_hold": 3.0, "movie_prominence_db": 14.0}})
    assert v.content_detector.conversation_hold == 3.0
    assert v.content_detector.prominence_db == 14.0


def test_the_conversation_memory_does_not_need_the_classifier_to_be_sure():
    """YAMNet unsure (speech 0.2) during plain dialogue: the voice measures
    alone still keep the conversation remembered."""
    det = ContentDetector(fps=60.0)
    talk = speech(8.0)[:, 0]
    n = 735
    for i in range(len(talk) // n):
        frame = talk[i * n:(i + 1) * n]
        spec = np.abs(np.fft.rfft(np.pad(frame * np.hamming(n), (0, 2048 - n))))
        det.update(spec, None, RATE, rhythm=0.1, wave=frame, speech=0.2, music_vote=0.05)
    assert det.conversation

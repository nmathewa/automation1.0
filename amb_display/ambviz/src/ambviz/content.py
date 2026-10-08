"""Is this music, or someone talking? -- the question ``movie`` switches on.

A film is mostly dialogue and ambience with music in stretches. A strip that
animates through all of it throbs under every line; one that is always subtle
wastes the score and the songs. ``movie`` runs the full director only while
music plays, which needs this answer continuously and without a model.

Four pieces of evidence, each cheap and each fooled by something the others are
not:

``accompaniment``
    The voice compared with everything else. Speech and singing both sit in the
    centre of the mix and inside 300-3400 Hz, so a voice alone cannot say which
    it is -- but singing comes *with* a band and dialogue mostly does not. This
    measures how much energy is not a centred voice: the side channel (anything
    panned), the bottom (bass and kick, below 80 Hz -- under even a low male
    voice's pitch, which can reach down to about 85 Hz) and the top (cymbals,
    air; above 6 kHz). Energy-weighted over half a second, so the near-silence
    between words -- where room tone is all there is -- barely counts. A singer
    over a band reads high, a line of dialogue reads low.

``rhythm``
    The tempo tracker's confidence. Music repeats at a steady period; speech
    does not, and neither does most ambience.

``continuity``
    One minus the low-energy ratio -- the share of frames well below the
    recent average (Scheirer & Slaney 1997). Speech is full of short pauses
    between words and phrases; music rarely drops out.

``syllables``
    How much the voice band's level moves at 3-6 Hz, the rate of syllables
    (Houtgast & Steeneken's modulation index; Scheirer & Slaney's 4 Hz energy).
    Strong in speech, weaker in music -- though not absent, since eighth notes
    at 120 BPM are also 4 Hz, which is why it carries the least weight.

``held``
    How often the pitch in the frame is held from one frame to the next. Singing
    is about 95% voiced and sits on notes; speech is about 60% voiced and its
    pitch glides and resets between syllables (arXiv 2204.03166). Measured by
    autocorrelation over 80-1000 Hz on the mid channel, so a steady instrument
    counts too -- which points the right way, since a held note is music
    whoever plays it.

``groove``
    Drums landing on a steady beat: the share of kick, snare and hat hits that
    fall on the tempo grid while the tracker is locked. Measured on a pop song,
    locked 88% of the time with 42% of hits on the grid; on film dialogue,
    locked 0% and 0%. Raw hit counts say nothing -- consonants are percussive,
    and dialogue fired the detectors 5.8 times a second against the song's 6.8
    -- but speech never repeats at a steady period. The strongest cue for "this
    is a song", and sufficient on its own -- combined as a noisy-OR with
    the rest, so a breakdown that drops the beat still counts. Its blind spot is a score *under* dialogue: the speech breaks
    the pulse up, and the same score locked 100% once Demucs removed the voice.

**Conversations.** Quiet score under a dialogue scene is the hard case: in
the gaps between lines the music is briefly the loudest thing, and judged a
second at a time it *is* dominant. Measured on a film, YAMNet alternated
music/speech every second for 48 s while the score sat 12.6 dB under the
dialogue. So the detector remembers how loud the dialogue was -- the loud part
of each line, held for ``dialogue_memory`` seconds after the last one -- and
music counts only if, measured where nobody is talking, it comes within
``prominence_db`` of that. A song at normal volume clears it; background music,
quiet songs in a scene included, does not. With no dialogue remembered there is
nothing to compare against, and music shows.

When YAMNet runs -- always, under ``movie``, if it is installed -- it decides:
music counts only while its music score clearly leads its speech score. That is
the question a film needs answered ("is the music *dominant*?"), and on a score
mixed into the centre the cues above can see the music but cannot weigh it
against the voice. The Demucs vocal share, when running, nudges the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ambviz.dsp import EPS, ExpFilter


#: Seconds after a line ends before the loudness counts as the music's.
SETTLE_SECONDS = 0.5

#: A vote at least this strong, held for SNAP_SECONDS, rises in SNAP_SECONDS
#: more instead of over ``attack``.
STRONG = 0.8
SNAP_SECONDS = 0.3


@dataclass
class Content:
    """What the audio is, as the slow layer sees it."""

    music: float = 0.0
    """0-1. How sure we are that music is playing, smoothed over seconds."""

    label: str = "quiet"
    """``music``, ``song`` (music with a voice in it), ``dialogue`` or ``quiet``."""

    accompaniment: float = 0.0
    rhythm: float = 0.0
    continuity: float = 0.0
    syllables: float = 0.0
    held: float = 0.0
    groove: float = 0.0
    voice: float = 0.0
    conversation: bool = False
    """A dialogue level is remembered; music must come within
    ``prominence_db`` of it to count."""

    dialogue_db: float | None = None
    """The remembered loudness of the dialogue, dB, or None."""

    music_db: float | None = None
    """Loudness where nobody is talking, dB, or None before any such moment."""
    """How much of the energy is a centred voice band, 0-1."""

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class ContentDetector:
    """Frame-rate evidence in, a smoothed music/dialogue verdict out."""

    #: How much each piece of DSP evidence counts. Accompaniment is the
    #: strongest single cue for "singing, not talking"; rhythm the strongest for
    #: "music at all"; syllables the weakest, for the reason in the module doc.
    WEIGHTS = {"accompaniment": 0.30, "rhythm": 0.25, "held": 0.20,
               "continuity": 0.15, "syllables": 0.10}

    #: Above ``ON`` it is music; it stays music until below ``OFF``. A single
    #: threshold flaps on every pause in a ballad. ``ON`` is set high because
    #: in a film the strip should be quiet most of the time: a busy effects
    #: scene -- loud and wide, but with no steady beat and no held notes --
    #: reaches about half the evidence and must stay below it.
    ON, OFF = 0.62, 0.45

    def __init__(self, fps: float = 60.0, window: float = 3.0,
                 attack: float = 2.0, release: float = 1.5,
                 dialogue_memory: float = 60.0, prominence_db: float = 6.0):
        self.fps = float(fps)
        self.dialogue_memory = float(dialogue_memory)
        self.prominence_db = float(prominence_db)
        self._db = np.full(int(1.0 * fps), -120.0)
        self._since_speech = 1e9
        self.conversation = False
        self.dialogue_db: float | None = None
        self.music_db: float | None = None
        a_level = float(np.clip(1.0 / (2.0 * fps), 1e-4, 0.5))
        self._a_dialogue = a_level
        self._a_music = float(np.clip(1.0 / (0.5 * fps), 1e-4, 0.5))
        self.n = max(16, int(window * fps))
        self._level = np.zeros(self.n)
        self._voice = np.zeros(self.n)
        self._filled = 0
        self._masks: dict[int, tuple[np.ndarray, ...]] = {}
        a_fast = float(np.clip(1.0 / max(0.5 * fps, 1.0), 1e-4, 0.5))
        # Energies, averaged separately and divided afterwards: averaging the
        # per-frame *ratio* gives a pause between words, where room tone is all
        # there is, as much say as the words.
        self._other = ExpFilter(0.0, alpha_decay=a_fast, alpha_rise=a_fast)
        self._voiced = ExpFilter(0.0, alpha_decay=a_fast, alpha_rise=a_fast)
        self._energy = ExpFilter(0.0, alpha_decay=a_fast, alpha_rise=a_fast)
        # Asymmetric on purpose: music arriving should show quickly, and a line
        # of dialogue over a song should not end the song.
        self._music = ExpFilter(0.0,
                                alpha_decay=float(np.clip(1.0 / (release * fps), 1e-4, 0.5)),
                                alpha_rise=float(np.clip(1.0 / (attack * fps), 1e-4, 0.5)))
        # A sudden, unmistakable arrival -- a sting, a needle drop -- is taken
        # at once rather than over ``attack``: the moment is the point. Only a
        # strong vote earns it, so borderline evidence still has to persist.
        self._snap = float(np.clip(1.0 / (SNAP_SECONDS * fps), 1e-4, 0.5))
        self._slow_rise = self._music.alpha_rise
        # Strong for this long before it counts as an arrival: a syllable, a
        # crash or a door slam can spike one frame's vote, and an arrival that
        # lasts a third of a second is still caught almost at once.
        self._strong_frames = 0
        self._db[:] = -120.0
        self._since_speech = 1e9
        self.dialogue_db = None
        self.music_db = None
        self._strong_needed = max(1, int(SNAP_SECONDS * fps))
        self.is_music = False
        self.content = Content()
        self._pitch: float | None = None
        self._held = ExpFilter(0.0, alpha_decay=a_fast * 0.5, alpha_rise=a_fast * 0.5)
        # Per hit rather than per frame, so it is a share of hits; ~3 s memory
        # at the six or so hits a second music produces.
        self._groove = ExpFilter(0.0, alpha_decay=0.06, alpha_rise=0.06)
        self._locked = ExpFilter(0.0, alpha_decay=a_fast * 0.3, alpha_rise=a_fast * 0.3)

    def reset(self) -> None:
        self._level[:] = 0.0
        self._voice[:] = 0.0
        self._filled = 0
        self._pitch = None
        self._groove.value = 0.0
        self._locked.value = 0.0
        self._strong_frames = 0

    def _track_pitch(self, wave: np.ndarray, rate: float) -> None:
        """Note whether this frame's pitch is held from the last one."""
        x = wave - wave.mean()
        n = len(x)
        spec = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        if ac[0] <= EPS:
            self._pitch = None
            return
        lo, hi = int(rate / 1000.0), min(n - 1, int(rate / 80.0))
        lag = lo + int(np.argmax(ac[lo:hi]))
        voiced = ac[lag] / ac[0] > 0.45
        pitch = rate / lag if voiced else None
        held = (pitch is not None and self._pitch is not None
                and abs(pitch - self._pitch) / self._pitch < 0.03)
        self._held.update(1.0 if held else 0.0)
        self._pitch = pitch

    def _bands(self, bins: int, rate: float) -> tuple[np.ndarray, ...]:
        if bins not in self._masks:
            f = np.fft.rfftfreq(2 * (bins - 1), 1.0 / rate)
            self._masks[bins] = ((f >= 300) & (f <= 3400),      # voice
                                 (f >= 30) & (f < 80),           # bass and kick
                                 f > 6000,                       # cymbals, air
                                 f >= 30)                        # everything audible
        return self._masks[bins]

    def update(self, mid: np.ndarray, side: np.ndarray | None, rate: float,
               rhythm: float, wave: np.ndarray | None = None,
               hit: bool = False, on_grid: bool = False, locked: bool = False, speech: float | None = None, music_vote: float | None = None,
               vocal_share: float | None = None) -> Content:
        """One frame of evidence.

        ``mid`` and ``side`` are magnitude spectra; ``side`` is None for mono,
        which loses the panning half of ``accompaniment`` but nothing else.
        ``wave`` is the mid channel's latest samples, for the pitch; ``hit``,
        ``on_grid`` and ``locked`` describe this frame's drum hits against the
        tempo grid.
        ``speech`` and ``music_vote`` are YAMNet group scores and
        ``vocal_share`` the Demucs vocal prominence, each None when not running.
        """
        voice_m, low_m, high_m, all_m = self._bands(len(mid), rate)
        p = mid.astype(np.float64) ** 2
        total = float(p[all_m].sum())
        side_e = float((side.astype(np.float64) ** 2)[all_m].sum()) if side is not None else 0.0
        energy = total + side_e
        voice_e = float(p[voice_m].sum())
        other = side_e + float(p[low_m].sum()) + float(p[high_m].sum())

        self._level[:-1] = self._level[1:]
        self._level[-1] = energy
        self._db[:-1] = self._db[1:]
        self._db[-1] = 10.0 * np.log10(energy + 1e-12)
        self._voice[:-1] = self._voice[1:]
        self._voice[-1] = np.sqrt(voice_e)
        self._filled = min(self.n, self._filled + 1)

        other_s = float(self._other.update(other))
        voice_s = float(self._voiced.update(voice_e))
        energy_s = float(self._energy.update(energy))
        # Relative to the voice band rather than the total: what matters is
        # whether there is a band *around* the voice, not how loud it is.
        acc = other_s / (other_s + voice_s) if other_s + voice_s > EPS else 0.0
        accompaniment = float(np.clip(acc / 0.5, 0.0, 1.0))

        if wave is not None:
            self._track_pitch(np.asarray(wave, dtype=np.float64), rate)
        lock = float(self._locked.update(1.0 if locked else 0.0))
        if hit:
            self._groove.update(1.0 if (locked and on_grid) else 0.0)
        # Locked most of the time *and* most hits on the grid: about 0.37 on
        # the pop song, 0 on dialogue.
        groove = float(np.clip(lock * self._groove.value / 0.3, 0.0, 1.0))
        # Singing holds about two frames in three at 60 fps; speech about one
        # in five.
        held = float(np.clip((self._held.value - 0.15) / 0.45, 0.0, 1.0))

        if self._filled < self.n // 2:
            return self.content
        level = self._level[-self._filled:]
        mean = float(level.mean())
        low_ratio = float(np.mean(level < 0.5 * mean)) if mean > EPS else 1.0
        # Speech sits around 0.3-0.5, continuous music under 0.15.
        continuity = float(np.clip(1.0 - (low_ratio - 0.1) / 0.35, 0.0, 1.0))

        env = self._voice[-self._filled:]
        env = env - env.mean()
        spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
        freqs = np.fft.rfftfreq(len(env), 1.0 / self.fps)
        band = spec[(freqs >= 3.0) & (freqs <= 6.0)].sum()
        modulation = spec[(freqs >= 0.5) & (freqs <= 15.0)].sum()
        syl = float(band / modulation) if modulation > EPS else 0.0
        # Speech puts roughly half its modulation energy at syllable rate.
        syllables = float(np.clip((0.5 - syl) / 0.3, 0.0, 1.0))

        rhythm = float(np.clip(rhythm / 0.6, 0.0, 1.0))
        # Accompaniment and syllables compare things *with a voice*. Without
        # one they read as maximally musical -- everything is "around the
        # voice", and nothing moves at syllable rate -- which is exactly what
        # an explosion or a car chase looks like too. Absent a voice they are
        # neutral, and held notes, rhythm and drums have to make the case.
        presence = float(np.clip((voice_s / energy_s if energy_s > EPS else 0.0) / 0.15, 0.0, 1.0))
        accompaniment = presence * accompaniment + (1.0 - presence) * 0.5
        syllables = presence * syllables + (1.0 - presence) * 0.5
        w = self.WEIGHTS
        evidence = (w["accompaniment"] * accompaniment + w["rhythm"] * rhythm
                    + w["held"] * held + w["continuity"] * continuity
                    + w["syllables"] * syllables)
        # Drums on the beat are proof of a song, not a requirement for one: a
        # breakdown drops the pulse and is still the song. Either case can
        # carry the verdict (a noisy-OR); dialogue has no groove, so this
        # leaves it exactly where the other cues put it.
        evidence = 1.0 - (1.0 - evidence) * (1.0 - groove)
        # The classifier, when running, decides; the DSP only nudges.
        #
        # Measured on 90 s of a film whose score is mixed into the centre with
        # the dialogue: in the seconds YAMNet called music-dominant the DSP's
        # accompaniment read 0.12 against 0.04 for pure dialogue, and rhythm
        # 0.30 against 0.20 -- real differences, far too small to threshold.
        # The question is "is the music *dominant*", not "is there music", so
        # the vote is music's lead over speech, and nothing without a clear
        # music score counts at all.
        vote = evidence
        if speech is not None and music_vote is not None:
            lead = music_vote - speech
            dominant = float(np.clip(0.5 + lead, 0.0, 1.0)) if music_vote > 0.25 else 0.0
            vote = 0.8 * dominant + 0.2 * evidence
        if vocal_share is not None:
            vote = 0.7 * vote + 0.3 * float(np.clip(1.0 - vocal_share, 0.0, 1.0))

        # Is a conversation going on? Either witness is enough: YAMNet hearing
        # speech, or a voice carrying the energy with almost nothing around it.
        # YAMNet alone let the memory lapse live, mid-dialogue, whenever it was
        # less than sure -- while the voice measured 0.73 of the energy and
        # the accompaniment 0.01.
        talking = presence >= 1.0 and accompaniment < 0.35
        if speech is not None and music_vote is not None:
            talking = talking or (speech >= 0.4 and speech >= music_vote)
        self._since_speech = 0.0 if talking else self._since_speech + 1.0 / self.fps
        level = float(self._db[-1])
        if talking:
            # The loud part of the line, not its pauses: the 90th percentile
            # of the last second, averaged over a couple of seconds of talk.
            loud = float(np.percentile(self._db, 90))
            self.dialogue_db = (loud if self.dialogue_db is None
                                else self.dialogue_db + self._a_dialogue * (loud - self.dialogue_db))
        elif level > -110.0 and self._since_speech >= SETTLE_SECONDS:
            # Only where nobody is talking, and not in the first moments after
            # a line: its tail and the room's reverb are still the voice, and
            # counting them put a film's score 7-9 dB under its dialogue where
            # the separated stems measure 12.6. A song has no such gaps to wait
            # for -- its vocals come with a band, so it does not read as
            # talking at all.
            self.music_db = (level if self.music_db is None
                             else self.music_db + self._a_music * (level - self.music_db))
        if self._since_speech > self.dialogue_memory:
            self.dialogue_db = None
        self.conversation = self.dialogue_db is not None
        # No dialogue remembered: nothing to compare against, music shows. A
        # conversation but no pause yet to measure the music in: unknown is
        # not comparable, so the glow holds until there is a level to judge.
        comparable = (self.dialogue_db is None
                      or (self.music_db is not None
                          and self.music_db >= self.dialogue_db - self.prominence_db))
        if self.conversation and not comparable:
            # Background music, measured against the conversation it sits
            # under: hold the glow, and let a running animation release as it
            # would on dialogue.
            vote = min(vote, self.OFF - 0.1)
        self._strong_frames = self._strong_frames + 1 if vote >= STRONG else 0
        snapping = self._strong_frames >= self._strong_needed
        self._music.alpha_rise = self._snap if snapping else self._slow_rise
        music = float(self._music.update(vote))

        if self.is_music:
            self.is_music = music >= self.OFF
        else:
            self.is_music = music >= self.ON
        voice = voice_s / energy_s if energy_s > EPS else 0.0
        if self.is_music:
            label = "song" if voice > 0.35 and accompaniment < 0.8 else "music"
        else:
            # A voice's fundamental sits below the band, so even pure speech
            # puts only a fraction of its energy in it.
            label = "dialogue" if voice > 0.1 else "quiet"
        self.content = Content(music=music, label=label, accompaniment=accompaniment,
                               rhythm=rhythm, continuity=continuity, syllables=syllables,
                               held=held, groove=groove, voice=voice,
                               conversation=self.conversation,
                               dialogue_db=self.dialogue_db, music_db=self.music_db)
        return self.content

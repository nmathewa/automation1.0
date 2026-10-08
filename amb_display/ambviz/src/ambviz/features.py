"""What an effect gets to look at, and the onset detector behind it.

Effects used to receive a bare Mel array, which meant every effect could react
to *spectrum* but none could react to a *beat*. That rules out the largest and
most satisfying family of LED effects -- anything that drops a ripple, launches
a ball or fires a burst when the music hits.

Onset detection is not source separation. Spectral flux is the positive
frame-to-frame change in the Mel bands summed across the spectrum, which the
pipeline is already halfway to computing; a running threshold and a refractory
period turn it into beats. It costs microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ambviz.dsp import ExpFilter
from ambviz.scene import Scene
from ambviz.stems import Stems


@dataclass
class StereoImage:
    """What each channel is doing on its own, as low/mid/high thirds, 0-1.

    The pipeline folds stereo to mid and side immediately, because that is what
    the spectrum and the vocal suppression want. Neither says which *speaker* a
    sound is coming out of, and a room rig has a wall on each side of the
    listener -- so a hard-panned guitar should light one wall and not the other,
    which needs the channels kept apart rather than summed.

    Computed only when the rig actually has sides. A plain strip pays nothing.
    """

    left: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right: tuple[float, float, float] = (0.0, 0.0, 0.0)
    available: bool = False

    left_mel: np.ndarray | None = None
    """The left channel's own filterbank, so a side can be *driven* by it
    rather than merely tinted from it."""

    right_mel: np.ndarray | None = None

    left_centroid_hz: float = 0.0
    """Each channel's own spectral centroid, in Hz.

    Swapping only the filterbank is not enough to make a wall look like its own
    channel: several effects take their *colour* from the centroid, so with a
    shared one the two walls came out the same hue and differed only in
    brightness -- which reads as one picture drawn twice."""

    right_centroid_hz: float = 0.0

    side_mel: np.ndarray | None = None
    """The difference signal's own filterbank -- everything *not* centred.

    This is where a normal mix keeps its stereo information. The two channels
    of a modern master carry almost the same magnitude spectrum (measured: band
    imbalance of 0.006-0.012), so comparing them says very little; but L - R
    isolates the reverb, the spread and the wide synths, and its spectral shape
    differs from the mix's by a wide margin -- 0.622 similarity against the
    0.99 the channels themselves manage.

    It is also why a side wall is dark at the bottom: bass is mono in almost
    every mix, so almost none of it survives the subtraction. That is correct,
    not a fault."""

    side_level: float = 0.0
    """How wide the mix is right now, 0-1."""

    @property
    def difference(self) -> float:
        """How differently the two channels are behaving, 0-1.

        Mean absolute difference between the channels' filterbanks, normalised
        by their combined level -- so it answers "are these carrying different
        material", not "is one louder". A mono file scores 0; a hard-panned
        arrangement approaches 1.
        """
        if self.left_mel is None or self.right_mel is None:
            return 0.0
        l, r = np.abs(self.left_mel), np.abs(self.right_mel)
        total = l + r
        energy = float(total.sum())
        if energy < 1e-9:
            return 0.0
        # Weighted by each band's own energy.
        #
        # The unweighted mean counts a near-silent band as loudly as the one
        # carrying the song, and near-silent bands differ between channels by
        # large *fractions* of almost nothing. On real material that read 0.650
        # for a mix whose actual panned share was 0.01 -- so the sides split
        # when there was nothing to split, and showed the front twice.
        return float(np.clip((np.abs(l - r) * total).sum() / (total * total).sum(),
                             0.0, 1.0))

    def level(self, right: bool = False) -> float:
        bands = self.right if right else self.left
        return float(max(bands)) if bands else 0.0

    @property
    def balance(self) -> float:
        """-1 fully left, 0 centred, +1 fully right."""
        l, r = self.level(), self.level(True)
        return 0.0 if l + r < 1e-9 else float((r - l) / (r + l))

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "left": [round(v, 3) for v in self.left],
            "right": [round(v, 3) for v in self.right],
            "balance": round(self.balance, 3),
            "difference": round(self.difference, 3),
            "width": round(self.side_level, 3),
        }


@dataclass
class Features:
    """One frame of analysis, shared by every node in a rig."""

    mel: np.ndarray
    """Mel filterbank levels after gain control, roughly 0-1 per band."""

    volume: float
    """Peak amplitude of the frame, 0-1."""

    onset: float = 0.0
    """Decaying strength of the most recent onset, 1.0 on the beat itself."""

    beat: bool = False
    """True only on the frame an onset fired -- use for one-shot events."""

    flux: float = 0.0
    """Raw spectral flux, before thresholding."""

    t: float = 0.0
    """Seconds since the pipeline started. Effects animating on their own -- a
    wave, a fire -- should use this rather than counting frames, so they run at
    the same speed whatever the frame rate."""

    silent: bool = False

    centroid: float = 0.5
    """Spectral centroid, rescaled to the range it has actually occupied.

    The perceptual "brightness" of the sound, and a far better hue driver than
    band position: it moves when the character of the audio moves, not when its
    shape happens to sit somewhere. Adaptive rescaling is what keeps it using
    the whole range on content that only occupies a slice of it."""

    centroid_hz: float = 0.0
    """The same thing before any rescaling, in Hz.

    A slow consumer wants this: it must smooth first and learn the range of the
    *smoothed* signal. Learning the range of the raw one instead measures how far
    speech jitters frame to frame, which is far wider than how far the mood moves
    across a scene -- and the mood then averages to the middle of it."""

    dialogue: float = 0.0
    """How centre-dominated the speech band is, 0-1.

    Film dialogue is the centre channel, so it cancels in ``L - R``. A high
    value means someone is probably talking, without recognising anything."""

    slow: float = 0.0
    """Level on a scene time-scale rather than a frame one -- the mood signal."""

    spread: float = 0.0
    """Spectral bandwidth in Hz: how far energy is spread around the centroid.

    A voice is narrow, an explosion is wide. Together with level and onset rate
    this is what separates a dialogue scene from a fight."""

    scene: Scene = field(default_factory=Scene)
    """What a classifier thinks the audio *is*, when one is running."""

    image: StereoImage = field(default_factory=StereoImage)
    """Per-channel level, for rigs with a wall on each side of the listener."""

    stems: Stems = field(default_factory=Stems)
    """What the audio is *made of* -- the drums/bass/other/vocals balance,
    when a separator is running. Smoothed and about a second stale, so it
    belongs to the slow layer; nothing frame-timed may read it."""

    onset_rate: float = 0.0
    """Onset density, 0-1 -- how beat-driven this passage is."""

    brightness: float = 0.0
    """Spectral spread rescaled to 0-1: narrow content near 0, wide near 1."""

    percussive: float = 0.5
    """Share of spectral energy that is transient rather than sustained, 0-1.

    From median-filter HPSS, smoothed into a density. Unlike ``energy`` and
    ``brightness`` it is a ratio of two quantities in the same units, so it is
    absolute: 0.25 means the same thing in any song and on any input gain. That
    is why the director switches on it -- an adaptively rescaled feature drifts
    on static material and reads as a scene change that never happened.

    Drums and plucks push it up, pads and strings and speech pull it down.
    0.5 is the neutral value reported when there is no energy to judge."""

    energy: float = 0.0
    """How much is going on, 0-1, adaptively rescaled across the film.

    Loud, wideband and transient-rich reads high; quiet, narrow and centred reads
    low. Effects use it to decide how energetic to be -- the point is not to be
    subtle always, but to be subtle when the content is."""

    drums: "Drums" = field(default_factory=lambda: Drums())
    """Kick, snare and hat, each detected in its own band.

    ``beat`` is derived from these, but an effect that wants to answer the
    snare differently from the kick can read them directly."""

    content: "Content" = field(default_factory=lambda: _content())
    """Music, a song, dialogue or quiet -- what ``movie`` switches on. Slow."""

    tempo: "Tempo" = field(default_factory=lambda: Tempo())
    """The beat grid: period, phase and a ``pulse`` on each predicted beat."""

    @property
    def bands(self) -> int:
        return len(self.mel)

    def thirds(self) -> tuple[float, float, float]:
        """Mean level of the low, mid and high thirds of the spectrum."""
        n = max(1, len(self.mel) // 3)
        return (
            float(np.mean(self.mel[:n])),
            float(np.mean(self.mel[n:2 * n])),
            float(np.mean(self.mel[2 * n:])),
        )


@dataclass
class OnsetDetector:
    """Fires when the spectrum jumps -- a kick, a snare, a chord change.

    Spectral flux rises whenever energy appears that was not there a frame ago.
    A fixed threshold cannot work across quiet and loud passages, so the
    threshold follows a slow average of the flux itself and the detector simply
    asks whether this frame stands out from its neighbours.
    """

    sensitivity: float = 1.4
    """Multiple of the running average the flux must exceed. Lower fires more."""

    refractory: float = 0.12
    """Minimum seconds between onsets. 0.12 s caps at 500 BPM, which is well
    past any real tempo while still allowing fast hi-hat patterns."""

    decay: float = 0.12
    """Seconds for the onset strength to fall back to zero after a hit."""

    min_flux: float = 0.06
    """Minimum flux as a fraction of current spectral energy.

    The adaptive threshold alone is purely relative, so on sustained material --
    a held chord, a drone -- the running floor collapses and ordinary numerical
    wobble clears it. That fired onsets as fast on a swell as on a drum track.
    An absolute floor, scaled by the spectrum's own energy so it survives gain
    changes, is what distinguishes a hit from a steady tone."""

    _prev: np.ndarray | None = field(default=None, repr=False)
    _floor: ExpFilter = field(
        default_factory=lambda: ExpFilter(1e-3, alpha_decay=0.02, alpha_rise=0.08),
        repr=False,
    )
    _last_beat: float = field(default=-1e9, repr=False)
    _strength: float = field(default=0.0, repr=False)

    def update(self, mel: np.ndarray, t: float) -> tuple[float, bool, float]:
        """Return ``(onset, beat, flux)`` for this frame."""
        if self._prev is None or self._prev.shape != mel.shape:
            self._prev = np.copy(mel)
            return 0.0, False, 0.0

        flux = float(np.sum(np.maximum(mel - self._prev, 0.0)))
        self._prev = np.copy(mel)

        # Relative to the spectrum's own energy: a hit adds a large fraction of
        # what is already there, a steady tone does not.
        energy = float(np.sum(mel))
        relative = flux / energy if energy > 1e-9 else 0.0

        floor = float(self._floor.update(flux))
        beat = False
        if (flux > floor * self.sensitivity
                and relative >= self.min_flux
                and t - self._last_beat >= self.refractory):
            beat = True
            self._last_beat = t
            # Scale with how far past the threshold it landed, so a soft hit
            # reads softer than a hard one instead of everything being binary.
            self._strength = float(np.clip(flux / max(floor * self.sensitivity, 1e-9) - 1.0, 0.0, 1.0))

        elapsed = t - self._last_beat
        onset = self._strength * max(0.0, 1.0 - elapsed / self.decay) if self.decay else 0.0
        return float(onset), beat, flux


@dataclass
class Hit:
    """One instrument detector's reading for a single frame."""

    hit: bool = False
    """True only on the frame the hit was detected."""

    strength: float = 0.0
    """1.0-ish on the hit, decaying to zero before the next one could land."""

    flux: float = 0.0
    """The detector's novelty this frame, before thresholding."""

    def to_dict(self) -> dict:
        return {"hit": self.hit, "strength": round(self.strength, 3),
                "flux": round(self.flux, 4)}


class BandOnsetDetector:
    """Onsets inside one frequency band -- a kick, a snare, a hat.

    The full-mix :class:`OnsetDetector` cannot tell instruments apart, so a
    sung syllable fires it as readily as a drum. It also never saw the kick at
    all: the Mel bank starts at ``dsp.min_frequency`` (200 Hz), and a kick lives
    at 40-150 Hz. Measured on a 30 s pop recording it caught 57% of the kicks,
    and 30% of what it fired landed on nothing.

    Each band is judged against *its own* recent history, so a quiet hat line is
    not drowned by the kick, and log compression makes the flux a ratio rather
    than a difference -- a hit is a hit at any playback volume. The threshold is
    mean plus a multiple of the mean absolute deviation, rather than a multiple
    of the mean alone, so a band that is always busy does not fire on its
    ordinary churn.
    """

    def __init__(self, low: float, high: float, sensitivity: float = 2.0,
                 refractory: float = 0.1, decay: float = 0.15,
                 min_flux: float = 0.02, fps: float = 60.0):
        self.low, self.high = float(low), float(high)
        self.sensitivity = float(sensitivity)
        self.refractory = float(refractory)
        self.decay = float(decay)
        self.min_flux = float(min_flux)
        a_stat = float(np.clip(1.0 / max(0.5 * fps, 1.0), 1e-4, 0.5))
        a_ref = float(np.clip(1.0 / max(3.0 * fps, 1.0), 1e-4, 0.5))
        self._mean = ExpFilter(0.0, alpha_decay=a_stat, alpha_rise=a_stat)
        self._dev = ExpFilter(0.0, alpha_decay=a_stat, alpha_rise=a_stat)
        # Slow band level, the reference the log is taken against. Rises
        # faster than it falls, so a track starting loud is not one long hit.
        self._ref = ExpFilter(0.0, alpha_decay=a_ref, alpha_rise=a_ref * 10)
        self._mask: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._last_hit = -1e9
        self._strength = 0.0
        self.hits = 0

    def _band(self, spectrum: np.ndarray, rate: float) -> np.ndarray:
        if self._mask is None or len(self._mask) != len(spectrum):
            freqs = np.fft.rfftfreq(2 * (len(spectrum) - 1), 1.0 / rate)
            self._mask = (freqs >= self.low) & (freqs < min(self.high, rate / 2))
            self._prev = None
        return spectrum[self._mask]

    def update(self, spectrum: np.ndarray, rate: float, t: float) -> Hit:
        band = self._band(np.asarray(spectrum, dtype=np.float64), rate)
        if band.size == 0:
            return Hit()
        ref = float(self._ref.update(float(np.mean(band))))
        level = np.log1p(band / (0.1 * ref + 1e-9))
        if self._prev is None:
            self._prev = level
            return Hit()
        flux = float(np.mean(np.maximum(level - self._prev, 0.0)))
        self._prev = level

        # Decide against the statistics *before* this frame updates them, so a
        # hit does not raise its own bar.
        mean, dev = float(self._mean.value), float(self._dev.value)
        threshold = mean + self.sensitivity * dev
        hit = (flux > threshold and flux >= self.min_flux
               and t - self._last_hit >= self.refractory)
        self._mean.update(flux)
        self._dev.update(abs(flux - mean))
        if hit:
            self._last_hit = t
            self.hits += 1
            # Same scale as OnsetDetector: 0 just over the line, 1 at twice it.
            self._strength = float(np.clip(flux / max(threshold, 1e-9) - 1.0, 0.0, 1.0))
        elapsed = t - self._last_hit
        strength = self._strength * max(0.0, 1.0 - elapsed / self.decay) if self.decay else 0.0
        return Hit(hit=hit, strength=float(strength), flux=flux)


@dataclass
class Drums:
    """Kick, snare and hat, detected separately, plus the beat grid."""

    kick: Hit = field(default_factory=Hit)
    snare: Hit = field(default_factory=Hit)
    hat: Hit = field(default_factory=Hit)

    def to_dict(self) -> dict:
        return {"kick": self.kick.to_dict(), "snare": self.snare.to_dict(),
                "hat": self.hat.to_dict()}


@dataclass
class Tempo:
    """Where the beat is, as the tracker currently believes."""

    bpm: float = 0.0
    period: float = 0.5
    """Seconds per beat."""

    confidence: float = 0.0
    """How periodic the recent novelty is, 0-1. Below about 0.3 there is no
    reliable pulse and nothing should be snapped to the grid."""

    phase: float = 0.0
    """Position inside the current beat, 0 on the beat, rising to 1."""

    pulse: bool = False
    """True on the frame the grid says a beat falls."""

    @property
    def locked(self) -> bool:
        return self.confidence >= TempoTracker.LOCK_CONFIDENCE

    def to_dict(self) -> dict:
        return {"bpm": round(self.bpm, 1), "period": round(self.period, 4),
                "confidence": round(self.confidence, 3), "phase": round(self.phase, 3),
                "pulse": self.pulse, "locked": self.locked}


class TempoTracker:
    """Finds the beat period and phase from a per-frame novelty signal.

    Reacting to every onset is what made the strip flash on syllables and
    report 221 BPM for a song at 160: the old period was a running average of
    the gaps between *any* two onsets. This looks for periodicity instead.

    Period: autocorrelation of the last few seconds of novelty, weighted by a
    log-normal prior around 120 BPM (Ellis 2007) so the octave it settles on is
    the one people tap to, with the second multiple of each lag added in to
    favour periods the music actually repeats at. A new period has to win twice
    in a row before it replaces the current one.

    Phase: a comb over the same buffer picks the offset that best lines up with
    the novelty, and between estimates a phase-locked loop nudges the predicted
    beat toward hits that land near it. Fully causal; nothing waits for audio
    that has not arrived.
    """

    LOCK_CONFIDENCE = 0.3

    #: Once locked, a different period must outscore the current one by this
    #: factor before it can take over.
    SWITCH_MARGIN = 1.25

    def __init__(self, fps: float = 60.0, window: float = 6.0, min_bpm: float = 70.0,
                 max_bpm: float = 190.0, prior_bpm: float = 120.0,
                 interval: float = 0.5):
        self.fps = float(fps)
        self.n = max(8, int(window * fps))
        self.buf = np.zeros(self.n)
        self.min_lag = max(1, int(round(fps * 60.0 / max_bpm)))
        self.max_lag = min(self.n // 2, int(round(fps * 60.0 / min_bpm)))
        self.prior_bpm = float(prior_bpm)
        self.every = max(1, int(interval * fps))
        self._frames = 0
        self.period = 0.5
        self.confidence = 0.0
        self._candidate: float | None = None
        self._next_beat: float | None = None
        self._estimates = 0

    def _estimate(self) -> tuple[float, float, float]:
        """Return ``(period, confidence, advantage)``.

        ``advantage`` is how much the best period outscores the current one;
        1.0 when they are the same lag."""
        x = self.buf - self.buf.mean()
        energy = float(np.dot(x, x))
        if energy <= 1e-12:
            return self.period, 0.0, 1.0
        spec = np.fft.rfft(x, 2 * self.n)
        ac = np.fft.irfft(spec * np.conj(spec))[:self.n] / energy
        lags = np.arange(self.min_lag, self.max_lag + 1)
        doubled = np.where(2 * lags < self.n, ac[np.minimum(2 * lags, self.n - 1)], 0.0)
        bpm = 60.0 * self.fps / lags
        prior = np.exp(-0.5 * (np.log2(bpm / self.prior_bpm) / 0.9) ** 2)
        score = (ac[lags] + 0.5 * doubled) * prior
        i = int(np.argmax(score))
        here = int(np.clip(round(self.period * self.fps) - self.min_lag, 0, len(lags) - 1))
        advantage = float(score[i] / score[here]) if score[here] > 1e-12 else float("inf")
        lag = float(lags[i])
        # Parabolic refinement: frame resolution alone is 2-3% of a period at
        # these tempos, which is a whole beat of drift every forty.
        if 0 < i < len(lags) - 1:
            a, b, c = score[i - 1], score[i], score[i + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                lag += float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        return lag / self.fps, float(np.clip(ac[lags[i]], 0.0, 1.0)), advantage

    def _anchor(self, t: float) -> None:
        """Re-derive the phase from the buffer with a comb at the current period."""
        p = self.period * self.fps
        k = np.arange(int((self.n - 1) / p) + 1)
        best, best_off = -1.0, 0
        for off in range(int(p)):
            idx = (self.n - 1 - off - k * p).astype(int)
            idx = idx[idx >= 0]
            s = float(np.sum(self.buf[idx] * np.exp(-0.15 * np.arange(len(idx)))))
            if s > best:
                best, best_off = s, off
        last = t - best_off / self.fps
        self._next_beat = last + self.period
        while self._next_beat <= t:
            self._next_beat += self.period

    def update(self, novelty: float, t: float, hit: bool = False) -> Tempo:
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = max(0.0, float(novelty))
        self._frames += 1

        if self._frames % self.every == 0 and self._frames >= self.n // 2:
            period, confidence, advantage = self._estimate()
            # The first estimate is taken as it stands; smoothing it up from
            # zero would hold off a lock for seconds after every new track.
            first = self._estimates == 0
            self._estimates += 1
            if first:
                # Nothing to defend yet: the starting period is a placeholder,
                # and making it win twice would lock onto it at full confidence.
                self.confidence, self.period = confidence, period
                self._next_beat = None
            elif abs(period - self.period) / self.period < 0.06:
                self.period += 0.25 * (period - self.period)
                self._candidate = None
            elif self.confidence >= self.LOCK_CONFIDENCE and advantage < self.SWITCH_MARGIN:
                # Locked, and the challenger is barely better. A double-time
                # or half-time reading of the same music often scores within a
                # few percent of the true one, and flipping between them halves
                # the strip's pace for no audible reason.
                self._candidate = None
            elif self._candidate is not None and abs(period - self._candidate) / period < 0.06:
                self.period = period
                self._candidate = None
                self._next_beat = None          # re-anchor at the new period
            else:
                self._candidate = period
            if not first:
                self.confidence += 0.3 * (confidence - self.confidence)
            if self._next_beat is None and self.confidence > 0.0:
                self._anchor(t)

        pulse = False
        if self._next_beat is not None:
            # Pull the grid toward hits that land near a predicted beat.
            if hit:
                err = t - self._next_beat
                if err < -0.5 * self.period:
                    err += self.period          # just after the previous beat
                if abs(err) < 0.2 * self.period:
                    self._next_beat += 0.35 * err
            if t >= self._next_beat:
                pulse = True
                while self._next_beat <= t:
                    self._next_beat += self.period
            phase = 1.0 - (self._next_beat - t) / self.period
        else:
            phase = 0.0
        return Tempo(bpm=60.0 / self.period, period=self.period,
                     confidence=self.confidence, phase=float(np.clip(phase, 0.0, 1.0)),
                     pulse=pulse)

    def distance(self, t: float) -> float:
        """Seconds from ``t`` to the nearest grid beat, or inf if unanchored."""
        if self._next_beat is None:
            return float("inf")
        after = self._next_beat - t
        return min(abs(after), abs(self.period - after))


def _content():
    # Imported late: content.py imports dsp, which features also imports, and
    # keeping Features constructible without it avoids a cycle.
    from ambviz.content import Content
    return Content()

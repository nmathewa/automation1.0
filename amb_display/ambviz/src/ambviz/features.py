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

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

        floor = float(self._floor.update(flux))
        beat = False
        if flux > floor * self.sensitivity and t - self._last_beat >= self.refractory:
            beat = True
            self._last_beat = t
            # Scale with how far past the threshold it landed, so a soft hit
            # reads softer than a hard one instead of everything being binary.
            self._strength = float(np.clip(flux / max(floor * self.sensitivity, 1e-9) - 1.0, 0.0, 1.0))

        elapsed = t - self._last_beat
        onset = self._strength * max(0.0, 1.0 - elapsed / self.decay) if self.decay else 0.0
        return float(onset), beat, flux

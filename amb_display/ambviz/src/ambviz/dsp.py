"""Signal-processing primitives: smoothing, the Mel filterbank, and HPSS."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.ndimage import median_filter

from ambviz import melbank
from ambviz.settings import Settings

EPS = 1e-12


class ExpFilter:
    """Exponential smoothing with separate rise and decay rates.

    A small alpha means heavy smoothing. Splitting rise from decay is what lets
    the visualization snap up on a transient but fall away slowly.
    """

    def __init__(
        self,
        val: float | np.ndarray = 0.0,
        alpha_decay: float = 0.5,
        alpha_rise: float = 0.5,
    ):
        if not 0.0 < alpha_decay < 1.0:
            raise ValueError(f"alpha_decay must be in (0, 1), got {alpha_decay}")
        if not 0.0 < alpha_rise < 1.0:
            raise ValueError(f"alpha_rise must be in (0, 1), got {alpha_rise}")
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val

    @classmethod
    def from_alpha(cls, val: float | np.ndarray, alpha: tuple[float, float]) -> "ExpFilter":
        decay, rise = alpha
        return cls(val, alpha_decay=decay, alpha_rise=rise)

    def update(self, value: float | np.ndarray) -> float | np.ndarray:
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = value - self.value
            alpha[alpha > 0.0] = self.alpha_rise
            alpha[alpha <= 0.0] = self.alpha_decay
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class MelBank:
    """Mel filterbank sized from the audio and DSP settings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.rebuild()

    def rebuild(self) -> None:
        s = self.settings
        # The filterbank's frequency axis must match the FFT it will be applied
        # to. The analysis window is zero-padded to the next power of two, and
        # rfft then returns nfft//2 + 1 bins spanning 0 to Nyquist.
        #
        # The original code sized the bank to rate*history/(2*fps) instead --
        # the unpadded half-window -- and then sliced the spectrum to match. The
        # two axes disagreed by the padding ratio, about 1.4x at the defaults,
        # so every band sat well above the frequency it claimed: a 900 Hz tone
        # landed in the band labelled 1256 Hz.
        window = s.audio.samples_per_frame * s.audio.rolling_history
        self.n_fft = 1 << (window - 1).bit_length()
        self.n_fft_bands = self.n_fft // 2 + 1
        self.matrix, (centers_mel, self.frequencies) = melbank.compute_melmat(
            num_mel_bands=s.dsp.fft_bins,
            freq_min=s.dsp.min_frequency,
            freq_max=s.dsp.max_frequency,
            num_fft_bands=self.n_fft_bands,
            sample_rate=s.audio.rate,
        )
        # compute_melmat returns band centres in MEL, not Hz. Anything that
        # labels a band for a human wants Hz, so convert once here.
        self.center_frequencies_mel = centers_mel
        self.center_frequencies = melbank.mel_to_hertz(centers_mel)

    def apply(self, spectrum: np.ndarray) -> np.ndarray:
        return np.sum(np.atleast_2d(spectrum).T * self.matrix.T, axis=0)


def interpolate(y: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample ``y`` to ``length`` samples."""
    if len(y) == length:
        return y
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(y)), y)


class AdaptiveRange:
    """Rescales a feature to the range it has actually been occupying.

    A fixed mapping assumes the input uses its whole nominal range. Real content
    rarely does -- a film's spectral centroid may never leave a narrow slice --
    and everything then maps to the same colour. Tracking the observed range and
    stretching it back out is what stops that.

    Percentiles rather than min and max: a single loud transient would otherwise
    set the ceiling and flatten the mapping for as long as the window is deep.
    """

    def __init__(self, seconds: float = 45.0, fps: float = 60.0,
                 low_pct: float = 5.0, high_pct: float = 95.0,
                 min_span: float = 1e-6, relative_span: float = 0.02):
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.min_span = min_span
        # The floor has to scale with the feature, not be an absolute number.
        # A centroid in Hz drifting by half a hertz is noise, but half a hertz
        # comfortably exceeds any fixed epsilon -- and rescaling it would stretch
        # that noise across the whole output range.
        self.relative_span = relative_span
        # One sample per frame would be needlessly dense for a 45 s window, and
        # percentiles over 2700 values are not free at 60 fps.
        self.stride = max(1, int(fps / 8))
        self._history: deque[float] = deque(maxlen=max(8, int(seconds * fps / self.stride)))
        self._seen = 0
        self.low = 0.0
        self.high = 1.0

    def update(self, value: float) -> float:
        """Record ``value`` and return it rescaled to 0-1 over the seen range."""
        value = float(value)
        self._seen += 1
        if self._seen % self.stride == 0 or not self._history:
            self._history.append(value)
            if len(self._history) >= 4:
                data = np.fromiter(self._history, dtype=float)
                self.low = float(np.percentile(data, self.low_pct))
                self.high = float(np.percentile(data, self.high_pct))
        span = self.high - self.low
        floor = max(self.min_span, self.relative_span * abs(self.low + self.high) / 2.0)
        if span < floor:
            return 0.5          # not actually varying; sit in the middle
        return float(np.clip((value - self.low) / span, 0.0, 1.0))

    @property
    def span(self) -> float:
        return self.high - self.low


class RateLimiter:
    """Caps how fast a value may move, with a deadband.

    :class:`ExpFilter` smooths but places no ceiling on rate: a large step still
    produces a large first move. For a mood layer the rate *is* the point --
    "gentle over several seconds" is a statement about maximum speed. The
    deadband stops sub-threshold wobble from producing any movement at all,
    which is what separates a calm light from a nervous one.
    """

    def __init__(self, per_second: float, value: float = 0.0,
                 deadband: float = 0.0, wrap: bool = False,
                 fall_per_second: float | None = None):
        self.per_second = per_second
        # Perceived dynamics are asymmetric. A drum hit should arrive at once
        # and fade slowly; capping both directions equally makes a loud moment
        # take as long to appear as it takes to die away, which reads as
        # sluggish however well tuned the number is.
        self.fall_per_second = per_second if fall_per_second is None else fall_per_second
        self.deadband = deadband
        self.wrap = wrap
        self.value = value

    def update(self, target: float, dt: float) -> float:
        delta = target - self.value
        if self.wrap:
            # Hue is circular: 0.95 -> 0.05 is a short step forward, not a long
            # one backward.
            delta = (delta + 0.5) % 1.0 - 0.5
        if abs(delta) <= self.deadband:
            return self.value
        rate = self.per_second if delta > 0 else self.fall_per_second
        step = rate * max(dt, 0.0)
        self.value += float(np.clip(delta, -step, step))
        if self.wrap:
            self.value %= 1.0
        return self.value


class HarmonicPercussive:
    """Splits a spectrum into sustained and transient energy by median filtering.

    Fitzgerald's trick (2010), which needs no model and no training. On a
    spectrogram a held note is a *horizontal* ridge -- the same bin, frame after
    frame -- and a drum hit is a *vertical* one, every bin at the same instant.
    A median along time therefore survives the note and erases the hit, and a
    median along frequency does the reverse. Comparing the two estimates per bin
    gives a soft mask, and the masked sums are how much of this frame is
    sustained and how much is percussive.

    This exists for the director rather than for the effects. The character
    vector it switches on was three quarters unusable: ``energy`` was pinned near
    a constant by the classifier bias, and ``brightness`` is an
    :class:`AdaptiveRange` output, which by construction stretches whatever it
    is fed to fill 0-1 and so drifts on *any* material. The percussive fraction
    below is neither -- it is a ratio of two energies in the same units, so it is
    absolute, needs no adaptation, and means the same thing between songs.

    Deliberately **causal**: the time median runs over a trailing window, never a
    centred one, so nothing here waits on audio that has not arrived. The honest
    cost is that the harmonic estimate lags by about half a window (~75 ms at the
    defaults), which smears a little transient energy into the harmonic side. For
    a signal consumed by a selector with an eight second dwell that is
    irrelevant; it would not be for an effect drawn per frame.

    Measured at **130 us per frame** on a 1025-bin spectrum with the defaults,
    0.78% of the 16.7 ms budget at 60 fps. ``np.partition`` rather than
    ``np.median`` is worth 3x of that on its own and is exact for an odd window,
    since only the middle order statistic is ever needed.
    """

    def __init__(self, bins: int, frames: int = 9, kernel: int = 17,
                 power: float = 2.0):
        # An even window has no single middle element, and taking either
        # neighbour biases the estimate; round up rather than silently pick one.
        self.frames = int(frames) | 1
        self.kernel = max(1, int(kernel))
        self.power = float(power)
        self._mid = self.frames // 2
        self._buf = np.zeros((self.frames, int(bins)), dtype=np.float32)
        self._primed = False

    def update(self, spectrum: np.ndarray) -> tuple[float, float]:
        """Return ``(harmonic, percussive)`` energy for this frame.

        Both are sums over the masked spectrum, in the spectrum's own units, so
        only their ratio is meaningful across different material.
        """
        s = np.asarray(spectrum, dtype=np.float32)
        if s.shape != self._buf.shape[1:]:
            self._buf = np.zeros((self.frames, s.shape[0]), dtype=np.float32)
            self._primed = False
        # Fill the history with the first real frame instead of ramping up from
        # zeros. Starting empty makes the time median near zero for the first
        # nine frames, which drives the mask fully percussive and fires a switch
        # at startup on any material -- the same trap the director's own anchor
        # seeding was written to avoid.
        if not self._primed:
            self._buf[:] = s
            self._primed = True
        else:
            self._buf[:-1] = self._buf[1:]
            self._buf[-1] = s

        # Exact for an odd window and cheaper than a full sort: partition only
        # guarantees the k-th element is in place, which is all a median is.
        harmonic = np.partition(self._buf, self._mid, axis=0)[self._mid]
        percussive = median_filter(s, size=self.kernel, mode="nearest")

        # Wiener-style soft mask rather than a hard one. A binary mask assigns
        # every bin wholly to one side, and real audio is a mixture in almost
        # every bin; squaring sharpens the split without making it brittle.
        h = harmonic ** self.power
        p = percussive ** self.power
        mask = p / (p + h + EPS)
        return float(np.sum(s * (1.0 - mask))), float(np.sum(s * mask))

    @staticmethod
    def ratio(harmonic: float, percussive: float) -> float:
        """Percussive share of the two, 0-1, or 0.5 when there is no energy."""
        total = harmonic + percussive
        if total <= EPS:
            return 0.5
        return float(np.clip(percussive / total, 0.0, 1.0))

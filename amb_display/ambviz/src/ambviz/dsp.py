"""Signal-processing primitives: smoothing and the Mel filterbank."""

from __future__ import annotations

import numpy as np

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
        self.n_fft_bands = int(s.audio.rate * s.audio.rolling_history / (2.0 * s.audio.fps))
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

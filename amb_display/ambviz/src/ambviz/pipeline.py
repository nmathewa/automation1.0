"""The audio -> pixels pipeline."""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ambviz.dsp import EPS, ExpFilter, MelBank
from ambviz.effects import EFFECTS
from ambviz.settings import Settings

# Which settings force which object to be rebuilt when changed at runtime.
_REBUILDS_MEL_BANK = {"dsp.min_frequency", "dsp.max_frequency", "dsp.fft_bins"}
_REBUILDS_MEL_FILTERS = {"dsp.fft_bins", "smoothing.mel_gain", "smoothing.mel_smoothing"}
_REBUILDS_EFFECT = {
    "effect.name", "effect.mirror", "dsp.fft_bins",
    "smoothing.red", "smoothing.green", "smoothing.blue",
    "smoothing.common_mode", "smoothing.pixel", "smoothing.gain",
}


class Visualizer:
    """Turns raw audio frames into strip pixels.

    Call :meth:`process` once per audio frame. It returns a ``(3, n_pixels)``
    uint8-ranged array, or a blank strip when the input is below the volume
    threshold.

    :meth:`snapshot` publishes what the pipeline is currently seeing, and
    :meth:`apply` changes settings between frames. Both are meant to be driven
    from :mod:`ambviz.api` via :class:`~ambviz.control.CommandQueue` -- call
    :meth:`apply` on the same thread as :meth:`process`, never concurrently.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.n_pixels = settings.output.pixels
        self.mirror = settings.effect.mirror
        # A mirrored odd-length strip shares its centre pixel between both halves.
        self.width = (self.n_pixels + 1) // 2 if self.mirror else self.n_pixels

        self.mel_bank = MelBank(settings)
        self.effect = EFFECTS[settings.effect.name](settings, self.width)

        sm = settings.smoothing
        bins = settings.dsp.fft_bins
        self.mel_gain = ExpFilter.from_alpha(np.tile(1e-1, bins), sm.mel_gain)
        self.mel_smoothing = ExpFilter.from_alpha(np.tile(1e-1, bins), sm.mel_smoothing)

        self.samples_per_frame = settings.audio.samples_per_frame
        self._window = np.hamming(self.samples_per_frame * settings.audio.rolling_history)
        self._roll = np.zeros((settings.audio.rolling_history, self.samples_per_frame))
        self.mel: np.ndarray = np.zeros(bins)
        self.volume = 0.0
        self.silent = True
        self.frames = 0
        self._fps = ExpFilter(float(settings.audio.fps), alpha_decay=0.2, alpha_rise=0.2)
        self._last_frame: float | None = None

    # ── runtime reconfiguration ──────────────────────────────────────────────
    def set_effect(self, name: str) -> None:
        if name not in EFFECTS:
            raise ValueError(f"unknown effect {name!r}; expected one of {sorted(EFFECTS)}")
        self.apply({"effect": {"name": name}})

    def apply(self, patch: dict) -> None:
        """Apply a validated settings patch, rebuilding only what it touches.

        Call this from the audio thread between frames. Patches normally arrive
        pre-validated from :class:`~ambviz.control.CommandQueue`; validating
        again here is cheap and keeps direct callers honest.
        """
        touched = {
            f"{section}.{key}"
            for section, values in patch.items()
            if isinstance(values, dict)
            for key in values
        }
        self.settings._apply(patch)
        self.settings.validate()

        if touched & _REBUILDS_MEL_BANK:
            self.mel_bank.rebuild()
        if touched & _REBUILDS_MEL_FILTERS:
            self._build_mel_filters()
        if touched & _REBUILDS_EFFECT:
            self._build_effect()

    def _build_mel_filters(self) -> None:
        sm = self.settings.smoothing
        bins = self.settings.dsp.fft_bins
        self.mel_gain = ExpFilter.from_alpha(np.tile(1e-1, bins), sm.mel_gain)
        self.mel_smoothing = ExpFilter.from_alpha(np.tile(1e-1, bins), sm.mel_smoothing)
        self.mel = np.zeros(bins)

    def _build_effect(self) -> None:
        self.mirror = self.settings.effect.mirror
        self.width = (self.n_pixels + 1) // 2 if self.mirror else self.n_pixels
        self.effect = EFFECTS[self.settings.effect.name](self.settings, self.width)

    # ── telemetry ────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """What the pipeline is currently seeing.

        Safe to call from another thread: every value is copied out, and
        attribute rebinding is atomic in CPython, so no lock is needed.
        """
        s = self.settings
        mel_gain = self.mel_gain.value
        return {
            "fps": round(float(self._fps.value), 1),
            "frames": self.frames,
            "volume": round(self.volume, 6),
            "silent": self.silent,
            "effect": s.effect.name,
            "brightness": s.effect.brightness,
            "mirror": self.mirror,
            "pixels": self.n_pixels,
            "width": self.width,
            "bins": s.dsp.fft_bins,
            "source": s.audio.source,
            "mel": [round(float(v), 4) for v in self.mel],
            "mel_gain": round(float(np.max(mel_gain)), 6),
            "center_frequencies": [round(float(f), 1) for f in self.mel_bank.center_frequencies],
            "min_frequency": s.dsp.min_frequency,
            "max_frequency": s.dsp.max_frequency,
        }

    # ── the loop ─────────────────────────────────────────────────────────────
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Consume one frame of int16-scaled audio and return strip pixels."""
        self.frames += 1
        now = time.monotonic()
        if self._last_frame is not None and now > self._last_frame:
            self._fps.update(1.0 / (now - self._last_frame))
        self._last_frame = now

        y = samples / 2.0 ** 15
        self._roll[:-1] = self._roll[1:]
        self._roll[-1, :] = y[:self.samples_per_frame]
        data = np.concatenate(self._roll, axis=0).astype(np.float32)

        self.volume = float(np.max(np.abs(data)))
        self.silent = self.volume < self.settings.audio.min_volume
        if self.silent:
            self.mel = np.zeros_like(self.mel)
            return np.zeros((3, self.n_pixels))

        n = len(data)
        padded = np.pad(data * self._window, (0, 2 ** int(np.ceil(np.log2(n))) - n))
        spectrum = np.abs(np.fft.rfft(padded)[:n // 2])

        mel = self.mel_bank.apply(spectrum) ** self.settings.dsp.mel_exponent
        self.mel_gain.update(np.max(gaussian_filter1d(mel, sigma=self.settings.dsp.gain_sigma)))
        mel = mel / np.maximum(self.mel_gain.value, EPS)
        self.mel = self.mel_smoothing.update(mel)

        return self._to_strip(self.effect.render(self.mel))

    def _to_strip(self, half: np.ndarray) -> np.ndarray:
        if self.mirror:
            # Drop the duplicated centre column when the strip length is odd.
            tail = half[:, 1:] if self.n_pixels % 2 else half
            pixels = np.concatenate((half[:, ::-1], tail), axis=1)
        else:
            pixels = half
        pixels = pixels * self.settings.effect.brightness
        return np.clip(pixels, 0, 255)

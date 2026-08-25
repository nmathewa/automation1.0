"""The audio -> pixels pipeline."""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ambviz.dsp import EPS, AdaptiveRange, ExpFilter, MelBank
from ambviz.effects import EFFECTS
from ambviz.features import Features, OnsetDetector
from ambviz.scene import Scene, try_create
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
        # Side channel, kept alongside mid so centre-panned content can be
        # cancelled at analysis time. Stays zero for mono sources.
        self._roll_side = np.zeros_like(self._roll)
        self.stereo = False
        self._band_mask: np.ndarray | None = None
        self.mel: np.ndarray = np.zeros(bins)
        self.volume = 0.0
        self.silent = True
        self.frames = 0
        self._fps = ExpFilter(float(settings.audio.fps), alpha_decay=0.2, alpha_rise=0.2)
        self._last_frame: float | None = None

        fps = float(settings.audio.fps)
        self.centroid_range = AdaptiveRange(seconds=settings.mood.range_seconds, fps=fps)
        # Each component gets its own range: they have different units and very
        # different dynamics, so one shared normaliser would let the loudest
        # dominate.
        self.level_range = AdaptiveRange(seconds=settings.mood.range_seconds, fps=fps)
        self.spread_range = AdaptiveRange(seconds=settings.mood.range_seconds, fps=fps)
        self.onset_range = AdaptiveRange(seconds=settings.mood.range_seconds, fps=fps)
        self.onset_rate = ExpFilter(0.0, alpha_decay=0.02, alpha_rise=0.15)
        # Optional and best-effort: returns None if the runtime, the model or
        # the network is missing, and everything downstream copes.
        self.classifier = (
            try_create(settings.audio.rate, interval=settings.mood.scene_interval)
            if settings.mood.scene_weight > 0.0 else None
        )
        # One frame over the response time: a level envelope that moves over a
        # scene rather than a beat.
        alpha = float(np.clip(1.0 / max(settings.mood.response_seconds * fps, 1.0), 1e-4, 0.5))
        self.slow_level = ExpFilter(0.0, alpha_decay=alpha, alpha_rise=alpha * 3)
        self._speech_mask: np.ndarray | None = None

        self.onsets = OnsetDetector(
            sensitivity=settings.dsp.onset_sensitivity,
            refractory=settings.dsp.onset_refractory,
        )
        self.features = Features(mel=np.zeros(bins), volume=0.0, silent=True)
        # Static per process, so computed once: it lets a client build its
        # effect list from the state stream instead of a second request.
        self.available_effects = sorted(EFFECTS)
        self.beats = 0
        self._started = time.monotonic()

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

        if touched & {"dsp.vocal_band"}:
            self._band_mask = None
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
            "effects": self.available_effects,
            "brightness": s.effect.brightness,
            "mirror": self.mirror,
            "pixels": self.n_pixels,
            "width": self.width,
            "bins": s.dsp.fft_bins,
            "source": s.audio.source,
            "stereo": self.stereo,
            "vocal_suppression": s.dsp.vocal_suppression,
            "vocal_band": list(s.dsp.vocal_band),
            "centroid": round(float(self.features.centroid), 3),
            "dialogue": round(float(self.features.dialogue), 3),
            "slow": round(float(self.features.slow), 4),
            "spread": round(float(self.features.spread), 1),
            "energy": round(float(self.features.energy), 3),
            "scene": self.features.scene.to_dict(),
            "mood": self._mood_snapshot(),
            "director": (self.effect.state()
                         if hasattr(self.effect, "state") else None),
            "mel": [round(float(v), 4) for v in self.mel],
            "onset": round(float(self.features.onset), 3),
            "beat": self.features.beat,
            "beats": self.beats,
            "flux": round(float(self.features.flux), 4),
            "mel_gain": round(float(np.max(mel_gain)), 6),
            "center_frequencies": [round(float(f), 1) for f in self.mel_bank.center_frequencies],
            "min_frequency": s.dsp.min_frequency,
            "max_frequency": s.dsp.max_frequency,
        }

    # ── the loop ─────────────────────────────────────────────────────────────
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Consume one frame of audio and return strip pixels.

        Accepts mono ``(n,)`` or stereo ``(n, 2)``. Stereo is split into mid and
        side; only the side channel makes vocal suppression possible.
        """
        self.frames += 1
        now = time.monotonic()
        if self._last_frame is not None and now > self._last_frame:
            self._fps.update(1.0 / (now - self._last_frame))
        self._last_frame = now

        samples = np.asarray(samples)
        self.stereo = samples.ndim == 2 and samples.shape[1] == 2
        if self.stereo:
            left, right = samples[:, 0] / 2.0 ** 15, samples[:, 1] / 2.0 ** 15
            y, y_side = (left + right) / 2.0, (left - right) / 2.0
        else:
            y = samples / 2.0 ** 15
            y_side = np.zeros_like(y)

        self._roll[:-1] = self._roll[1:]
        self._roll[-1, :] = y[:self.samples_per_frame]
        self._roll_side[:-1] = self._roll_side[1:]
        self._roll_side[-1, :] = y_side[:self.samples_per_frame]
        if self.classifier is not None:
            self.classifier.push(y[:self.samples_per_frame], self.settings.audio.rate)
        data = np.concatenate(self._roll, axis=0).astype(np.float32)

        # Audio time, not wall-clock. Anything downstream that integrates over
        # time -- the mood's rate limiter, the onset refractory -- must advance
        # with the audio it is describing. Wall-clock makes offline processing
        # run the mood far too fast and makes a dropped frame skew it, and it
        # makes results irreproducible between runs.
        elapsed = self.frames * self.samples_per_frame / float(self.settings.audio.rate)
        self.volume = float(np.max(np.abs(data)))
        self.silent = self.volume < self.settings.audio.min_volume
        if self.silent:
            self.mel = np.zeros_like(self.mel)
            self.features = Features(mel=self.mel, volume=self.volume, t=elapsed, silent=True)
            return np.zeros((3, self.n_pixels))

        n = len(data)
        pad = self.mel_bank.n_fft - n
        # The whole rfft, not a slice: the filterbank is built for exactly these
        # bins, and truncating here is what put the two frequency axes out of
        # step in the first place.
        spectrum = np.abs(np.fft.rfft(np.pad(data * self._window, (0, pad))))

        # Computed whenever the input is stereo, not only when suppression is
        # on: the dialogue indicator needs the side channel either way.
        side_spectrum = None
        if self.stereo:
            side = np.concatenate(self._roll_side, axis=0).astype(np.float32)
            side_spectrum = np.abs(np.fft.rfft(np.pad(side * self._window, (0, pad))))

        centred = self._centred(spectrum, side_spectrum)

        # Vocal suppression applies to *colour only*, never to rhythm.
        #
        # L - R removes everything centred, and in a real mix that is the kick,
        # the snare and usually the lead as well as the voice. Suppressing the
        # whole analysis therefore does not remove the singer, it removes the
        # song: onsets fell from 93 to 21 on test material at full strength.
        #
        # What actually annoys is the *colour* chasing the melody. So the full
        # mix drives level, onsets and energy, and a separate suppressed
        # spectrum drives the hue. The strip keeps the beat and stops following
        # the singer.
        mel = self.mel_bank.apply(spectrum) ** self.settings.dsp.mel_exponent

        suppression = self.settings.dsp.vocal_suppression
        if suppression > 0.0 and side_spectrum is not None:
            tonal = self._suppress_centre(spectrum, side_spectrum, suppression)
            mel_tonal = self.mel_bank.apply(tonal) ** self.settings.dsp.mel_exponent
        else:
            mel_tonal = mel
        self.mel_gain.update(np.max(gaussian_filter1d(mel, sigma=self.settings.dsp.gain_sigma)))
        mel = mel / np.maximum(self.mel_gain.value, EPS)
        self.mel = self.mel_smoothing.update(mel)

        onset, beat, flux = self.onsets.update(self.mel, elapsed)
        if beat:
            self.beats += 1

        # Hue follows the arrangement rather than the vocal line.
        centroid_hz = self._centroid(self._normalise(mel_tonal))
        centroid = self.centroid_range.update(centroid_hz)
        slow = float(self.slow_level.update(float(np.max(self.mel))))
        spread = self._spread(self.mel, centroid_hz)
        # Centred alone is not speech: a fight scene has centred bass and brass
        # too, and reading that as dialogue damps exactly the scene that should
        # be most energetic. A voice is centred *and* narrow.
        narrow = 1.0 - self.spread_range.update(spread)
        dialogue = float(np.clip(centred * narrow, 0.0, 1.0))

        scene = self.classifier.scene if self.classifier is not None else Scene()

        # A fight scene is loud, wide and full of transients; a dialogue scene is
        # none of those. Averaging the three normalised components is enough to
        # tell them apart without recognising anything.
        rate_norm = self.onset_range.update(float(self.onset_rate.update(1.0 if beat else 0.0)))
        energy = float(np.clip(
            0.5 * self.level_range.update(slow)
            + 0.3 * (1.0 - narrow)
            + 0.2 * rate_norm,
            0.0, 1.0))
        if scene.available:
            # What is playing says more about how energetic to be than the
            # running statistics do. Percussion and electronic music want a
            # spectrum; an orchestra wants a wash, however loud it gets.
            w = self.settings.mood.scene_weight
            driven = max(scene.get("percussion"), scene.get("electronic"),
                         scene.get("loud"))
            sustained = max(scene.get("orchestral"), scene.get("acoustic"))
            bias = float(np.clip(0.5 + 0.5 * (driven - sustained), 0.0, 1.0))
            energy = float(np.clip((1 - w) * energy + w * bias, 0.0, 1.0))

        self.features = Features(
            mel=self.mel, volume=self.volume, onset=onset, beat=beat,
            flux=flux, t=elapsed, silent=False,
            centroid=centroid, centroid_hz=centroid_hz, dialogue=dialogue, slow=slow,
            spread=spread, energy=energy, scene=scene,
            onset_rate=rate_norm, brightness=float(1.0 - narrow),
        )
        return self._to_strip(self.effect.render(self.features))

    def _mood_snapshot(self) -> dict | None:
        """Whatever slow state the running effect exposes, or None.

        The mood is the thing being tuned, so it has to be observable -- a hue
        that will not move is impossible to diagnose from the spectrum alone.
        """
        mood = getattr(self.effect, "mood", None)
        if mood is None:
            return None
        source = getattr(self.effect, "mood_source", None)
        out = {
            "hue": round(float(mood.hue), 4),
            "level": round(float(mood.level), 4),
            "accent": round(float(mood.accent), 4),
            "detail": round(float(mood.detail), 4),
        }
        if source is not None:
            out["target"] = round(float(source._last_target), 4)
            out["smoothed_hz"] = round(float(source._smooth.value), 1)
            out["range_lo"] = round(float(source._range.low), 1)
            out["range_hi"] = round(float(source._range.high), 1)
        return out

    def _normalise(self, mel: np.ndarray) -> np.ndarray:
        """Same gain treatment as the main path, so the two are comparable."""
        peak = float(np.max(mel))
        return mel / peak if peak > EPS else mel

    def _centroid(self, mel: np.ndarray) -> float:
        """Energy-weighted mean frequency of the filterbank, in Hz."""
        total = float(np.sum(mel))
        if total <= EPS:
            return 0.0
        return float(np.dot(mel, self.mel_bank.center_frequencies) / total)

    def _spread(self, mel: np.ndarray, centroid_hz: float) -> float:
        """Spectral bandwidth in Hz -- narrow for a voice, wide for an explosion."""
        total = float(np.sum(mel))
        if total <= EPS:
            return 0.0
        deviation = self.mel_bank.center_frequencies - centroid_hz
        return float(np.sqrt(np.dot(mel, deviation ** 2) / total))

    def _centred(self, mid: np.ndarray, side: np.ndarray | None) -> float:
        """How centre-dominated the speech band is, 0-1.

        Centre-panned content cancels in ``L - R``, so a large mid relative to
        side means the band is dominated by something sitting dead centre --
        which film dialogue always is. Mono input cannot tell, so it returns 0
        rather than claiming everything is speech.
        """
        if side is None:
            return 0.0
        if self._speech_mask is None or len(self._speech_mask) != len(mid):
            freqs = np.fft.rfftfreq(2 * (len(mid) - 1), 1.0 / self.settings.audio.rate)
            self._speech_mask = ((freqs >= 300.0) & (freqs <= 3400.0)).astype(np.float32)
        mid_e = float(np.sum((mid * self._speech_mask) ** 2))
        side_e = float(np.sum((side * self._speech_mask) ** 2))
        if mid_e + side_e <= EPS:
            return 0.0                      # silence says nothing either way
        return float(mid_e / (mid_e + side_e))

    def _suppress_centre(self, mid: np.ndarray, side: np.ndarray, amount: float) -> np.ndarray:
        """Crossfade from mid toward side inside the vocal band.

        Centre-panned content cancels in ``L - R``, and vocals are centred in
        almost every mix. Applying it across the whole spectrum would also
        cancel the kick and bass, which are centred too -- so outside the band
        the mid channel passes through untouched.
        """
        if self._band_mask is None or len(self._band_mask) != len(mid):
            # Derive the transform size from the spectrum itself: a real FFT of
            # nfft samples yields nfft//2 + 1 bins, so nfft = 2 * (bins - 1).
            # Taking it from the spectrum rather than from settings keeps this
            # correct for any window size.
            freqs = np.fft.rfftfreq(2 * (len(mid) - 1), 1.0 / self.settings.audio.rate)
            low, high = self.settings.dsp.vocal_band
            self._band_mask = ((freqs >= low) & (freqs <= high)).astype(np.float32)
        k = self._band_mask * amount
        return mid * (1.0 - k) + side * k

    def _to_strip(self, half: np.ndarray) -> np.ndarray:
        if self.mirror:
            # Drop the duplicated centre column when the strip length is odd.
            tail = half[:, 1:] if self.n_pixels % 2 else half
            pixels = np.concatenate((half[:, ::-1], tail), axis=1)
        else:
            pixels = half
        pixels = pixels * self.settings.effect.brightness
        return np.clip(pixels, 0, 255)

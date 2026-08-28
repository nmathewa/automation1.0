"""The audio -> pixels pipeline."""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ambviz.dsp import (EPS, AdaptiveRange, ExpFilter, HarmonicPercussive,
                        MelBank, interpolate)
from ambviz.effects import EFFECTS, hsv_to_rgb, rescale_clocks
from dataclasses import replace

from ambviz.features import Features, OnsetDetector, StereoImage
from ambviz.scene import Scene, try_create
from ambviz.stems import HUES, Stems, try_create as try_create_stems
from ambviz.settings import Settings

# Which settings force which object to be rebuilt when changed at runtime.
_REBUILDS_MEL_BANK = {"dsp.min_frequency", "dsp.max_frequency", "dsp.fft_bins"}
_REBUILDS_MEL_FILTERS = {"dsp.fft_bins", "smoothing.mel_gain", "smoothing.mel_smoothing"}
_REBUILDS_HPSS = {"dsp.hpss_frames", "dsp.hpss_kernel"}
_REBUILDS_EFFECT = {
    "effect.name", "effect.mirror", "dsp.fft_bins",
    "smoothing.red", "smoothing.green", "smoothing.blue",
    "smoothing.common_mode", "smoothing.pixel", "smoothing.gain",
}


#: Onset-density envelope value treated as "as busy as music gets". The
#: envelope is an ExpFilter of a 0/1 beat flag, so it saturates well below 1.
ONSET_RATE_FULL_SCALE = 0.5


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
        # A room is described by its sides; a plain strip is one side.
        self.segments = tuple(settings.output.segments) or (self.n_pixels,)
        # The widest side is the front and carries the animation; it renders
        # exactly as a single strip of that length would, so an existing rig
        # keeps behaving as it did. Mirroring folds about the *front's* centre,
        # never the room's -- the same fold taken across the whole chain would
        # put the left wall's reflection on the right one.
        self.front = int(np.argmax(self.segments))
        self.render_pixels = max(self.segments)
        # A mirrored odd-length strip shares its centre pixel between both halves.
        self.width = (self.render_pixels + 1) // 2 if self.mirror else self.render_pixels

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
        # Per-channel history, only for rigs that have a side to put it on.
        # A plain strip never allocates these and never pays the two extra
        # transforms they cost.
        self._sided = len(tuple(settings.output.segments)) > 1
        # The walls are ambient, so their level moves on a scene time-scale
        # rather than a frame one -- an ambient light that flickers is just a
        # small bright effect in the corner of your eye.
        a_side = float(np.clip(1.0 / max(0.8 * settings.audio.fps, 1.0), 1e-4, 0.5))
        self._side_level = [ExpFilter(0.0, alpha_decay=a_side, alpha_rise=a_side * 3)
                            for _ in range(2)]
        self._side_hue = ExpFilter(0.5, alpha_decay=a_side * 0.25, alpha_rise=a_side * 0.25)
        # One effect instance per side, at the side's own width. This is what
        # makes a side *centred on itself*: a mirrored effect folds about the
        # middle of whatever width it was built for, so building it at 30 puts
        # its centre in the middle of a 30-pixel wall instead of inheriting the
        # front's.
        self._side_effects: dict[tuple[int, bool, str], object] = {}
        self._burst_effects: dict[tuple[int, bool], object] = {}
        # Asymmetric, and the rise is the important half. A symmetric response
        # slower than the hold it has to fill can never reach full: at a 1.2 s
        # response and a two-beat hold the uplift peaked at 0.21 and was never
        # seen. It swells in quickly and eases out over the full response, which
        # is also the shape that reads as the music arriving and then letting go.
        a_uni = float(np.clip(
            1.0 / max(settings.output.unison_response * settings.audio.fps, 1.0),
            1e-4, 0.5))
        self._unison = ExpFilter(0.0, alpha_decay=a_uni,
                                 alpha_rise=min(0.5, a_uni * 4.0))
        # Level in dB, and the quiet it most recently passed through. The floor
        # drops to a new low at once and climbs back slowly, so a breakdown is
        # still remembered when the drop lands on it.
        self._level_db = ExpFilter(-90.0, alpha_decay=0.25, alpha_rise=0.5)
        self._db_floor = -90.0
        self._db_primed = False
        self.surge_db = 0.0
        self.track_started = 0.0
        self.tracks = 1
        self._silence_since: float | None = None
        self._unison_until = -1e9
        self._unison_armed = True
        self.unison_events = 0
        # Which wall answers the next event, and how lit each one currently is.
        self._accent = [0.0, 0.0]
        self._accent_target = [0.0, 0.0]
        self._accent_next = 0
        self._accent_at = -1e9
        self._accent_drifted = False
        self._accent_stems: float | None = None
        self.accent_events = 0
        # Median-ish spacing between onsets, so the accent can be paced in
        # beats instead of seconds. Seeded at a 120 BPM half-note, which is
        # what it converges to on most material anyway.
        self.beat_period = 0.5
        self._last_beat_t: float | None = None
        self._roll_lr = (np.zeros((2, *self._roll.shape)) if self._sided else None)
        self.image = StereoImage()
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
        # One range per channel: each wall's colour should use the full circle
        # for what *that* side is doing, rather than be squeezed into whatever
        # part of the range the mixed signal happens to occupy.
        self.channel_range = [
            AdaptiveRange(seconds=settings.mood.range_seconds, fps=fps) for _ in range(2)]
        # Per-channel and difference-signal bands are built by
        # _build_mel_filters, which the live path also calls -- describing them
        # in two places is how the front once came back 120 pixels wide against
        # a 120-pixel output expecting 60 of them.
        self._build_mel_filters()
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
        # Same contract as the classifier: optional, best-effort, and the
        # visualizer is unaffected if torch or the weights are missing.
        self.separator = (
            try_create_stems(settings.audio.rate, window=settings.mood.stem_window,
                             interval=settings.mood.stem_interval)
            if settings.mood.stem_weight > 0.0 else None
        )
        # The published balance is already about a second stale; unsmoothed it
        # correlates 0.62 with the truth and smoothed over two seconds, 0.85.
        alpha_s = float(np.clip(1.0 / max(settings.mood.stem_smoothing * fps, 1.0),
                                1e-4, 0.5))
        self._stem_filter: dict[str, ExpFilter] = {}
        self._stem_alpha = alpha_s
        self.stems = Stems()
        # One frame over the response time: a level envelope that moves over a
        # scene rather than a beat.
        alpha = float(np.clip(1.0 / max(settings.mood.response_seconds * fps, 1.0), 1e-4, 0.5))
        self.slow_level = ExpFilter(0.0, alpha_decay=alpha, alpha_rise=alpha * 3)
        self._speech_mask: np.ndarray | None = None

        # Sized to the padded rfft, which is what it is fed. Restart-only
        # settings decide that size, so it never changes under a live patch --
        # but update() reshapes on a mismatch anyway rather than trusting that.
        self.hpss = HarmonicPercussive(
            self.mel_bank.n_fft_bands,
            frames=settings.dsp.hpss_frames,
            kernel=settings.dsp.hpss_kernel,
        )
        # Same treatment as onset_rate: the per-frame ratio is near-binary, so
        # what the director wants is its density over a passage, not its value
        # on one frame.
        alpha_p = float(np.clip(1.0 / max(settings.dsp.percussive_smoothing * fps, 1.0),
                                1e-4, 0.5))
        self.percussive = ExpFilter(0.5, alpha_decay=alpha_p, alpha_rise=alpha_p)

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
            self._side_effects.clear()
            self._burst_effects.clear()
        if "mood.animations" in touched:
            # Not a full effect rebuild: that would discard every cached
            # animation's state and show a seam on a change that need not have
            # one. Directors retune in place; anything else ignores the list.
            retune = getattr(self.effect, "retune", None)
            if retune is not None:
                retune(tuple(self.settings.mood.animations))
        if "mood.stem_weight" in touched:
            self._ensure_separator()
        if touched & _REBUILDS_HPSS:
            self.hpss = HarmonicPercussive(
                self.mel_bank.n_fft_bands,
                frames=self.settings.dsp.hpss_frames,
                kernel=self.settings.dsp.hpss_kernel,
            )
        if touched & {"dsp.percussive_smoothing"}:
            a = float(np.clip(
                1.0 / max(self.settings.dsp.percussive_smoothing * self.settings.audio.fps, 1.0),
                1e-4, 0.5))
            # Keep the current value: rebuilding at 0.5 would inject a step into
            # the character vector and read as a scene change.
            self.percussive = ExpFilter(float(self.percussive.value),
                                        alpha_decay=a, alpha_rise=a)

    def _build_mel_filters(self) -> None:
        sm = self.settings.smoothing
        bins = self.settings.dsp.fft_bins
        self.mel_gain = ExpFilter.from_alpha(np.tile(1e-1, bins), sm.mel_gain)
        self.mel_smoothing = ExpFilter.from_alpha(np.tile(1e-1, bins), sm.mel_smoothing)
        self.mel = np.zeros(bins)
        # Sized to the band count too, and dsp.fft_bins is live.
        #
        # These are smoothed on the time-scale effects actually work on.
        # Comparing raw frames instead measures decorrelation -- the fast,
        # noise-like phase difference between two channels carrying the same
        # arrangement -- and every effect in the library smooths precisely that
        # away. It read 0.50 on a mix whose visible panned share was 0.01, so
        # the sides split and then showed the front twice.
        fps = float(self.settings.audio.fps)
        a_ch = float(np.clip(1.0 / max(0.6 * fps, 1.0), 1e-4, 0.5))
        self._channel_mel = [
            ExpFilter(np.tile(1e-2, bins), alpha_decay=a_ch, alpha_rise=a_ch)
            for _ in range(2)]
        self._side_mel = ExpFilter(np.tile(1e-2, bins),
                                   alpha_decay=a_ch * 2, alpha_rise=a_ch * 4)

    def _build_effect(self) -> None:
        # Sized from the *front*, not the whole chain.
        #
        # This is the rebuild path, and it was left sizing from n_pixels after
        # the room was added while __init__ moved to render_pixels. On a
        # segmented rig the two disagree, so the first live effect change
        # rendered a front as wide as the entire room and the frame came out
        # (3, 180) against an output expecting (3, 120), which killed the
        # process outright. Deriving both from the same number is the fix; the
        # duplication was the bug.
        self.mirror = self.settings.effect.mirror
        self.width = (self.render_pixels + 1) // 2 if self.mirror else self.render_pixels
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
            "segments": list(self.segments),
            "accents": self.accent_events,
            "beat_period": round(self.beat_period, 3),
            "accent_length": round(self.accent_length, 3),
            "fullness": round(self.fullness, 3),
            "surge_db": round(self.surge_db, 1),
            "tracks": self.tracks,
            "unison": round(float(self._unison.value), 3),
            "unisons": self.unison_events,
            "accent": [round(v, 3) for v in self._accent],
            "source": s.audio.source,
            "stereo": self.stereo,
            "vocal_suppression": s.dsp.vocal_suppression,
            "vocal_band": list(s.dsp.vocal_band),
            "centroid": round(float(self.features.centroid), 3),
            "dialogue": round(float(self.features.dialogue), 3),
            "slow": round(float(self.features.slow), 4),
            "spread": round(float(self.features.spread), 1),
            "energy": round(float(self.features.energy), 3),
            "percussive": round(float(self.features.percussive), 3),
            "scene": self.features.scene.to_dict(),
            "stems": self.features.stems.to_dict(),
            "image": self.features.image.to_dict(),
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
        if self._roll_lr is not None:
            for ch, sig in enumerate((left, right) if self.stereo else (y, y)):
                self._roll_lr[ch, :-1] = self._roll_lr[ch, 1:]
                self._roll_lr[ch, -1, :] = sig[:self.samples_per_frame]
        if self.classifier is not None:
            self.classifier.push(y[:self.samples_per_frame], self.settings.audio.rate)
        if self.separator is not None:
            # Stereo where possible: Demucs leans on the stereo image, so
            # folding to mono first throws away part of what makes it work.
            frame = (samples[:self.samples_per_frame].T / 2.0 ** 15
                     if self.stereo else y[:self.samples_per_frame])
            self.separator.push(frame, self.settings.audio.rate)
        data = np.concatenate(self._roll, axis=0).astype(np.float32)

        # Audio time, not wall-clock. Anything downstream that integrates over
        # time -- the mood's rate limiter, the onset refractory -- must advance
        # with the audio it is describing. Wall-clock makes offline processing
        # run the mood far too fast and makes a dropped frame skew it, and it
        # makes results irreproducible between runs.
        elapsed = self.frames * self.samples_per_frame / float(self.settings.audio.rate)
        self.volume = float(np.max(np.abs(data)))
        self.silent = self.volume < self.settings.audio.min_volume
        # Here rather than under the render, because process() returns early on
        # silence -- so anything downstream never runs during the very gap the
        # watcher exists to notice.
        self._watch_for_a_new_track(elapsed, self.silent)
        if self.silent:
            self.mel = np.zeros_like(self.mel)
            self.features = Features(mel=self.mel, volume=self.volume, t=elapsed,
                                     silent=True, percussive=float(self.percussive.value))
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
            if self._last_beat_t is not None:
                gap = elapsed - self._last_beat_t
                # Ignore the extremes: a double-triggered hit and a gap across
                # a silence say nothing about the tempo.
                if 0.08 <= gap <= 2.0:
                    self.beat_period += 0.12 * (gap - self.beat_period)
            self._last_beat_t = elapsed

        # HPSS on the *linear* spectrum, not the mel one. A mel band is already
        # a weighted average over many FFT bins, which smears the narrow ridge a
        # harmonic makes into something as wide as a drum -- exactly the
        # distinction being measured. Running it here costs 130 us against a
        # 16.7 ms frame, so the cheaper axis is not worth the blunter answer.
        harmonic, percussive = self.hpss.update(spectrum)
        percussive_now = HarmonicPercussive.ratio(harmonic, percussive)
        percussive_rate = float(self.percussive.update(percussive_now))

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

        self.image = self._stereo_image(pad, side_spectrum)
        scene = self.classifier.scene if self.classifier is not None else Scene()
        self.stems = self._smooth_stems()

        # A fight scene is loud, wide and full of transients; a dialogue scene is
        # none of those. Averaging the three normalised components is enough to
        # tell them apart without recognising anything.
        # Two values, deliberately. ``rate_norm`` is range-adapted and feeds the
        # ``energy`` composite below, where "relative to this film" is the right
        # question. ``rate_raw`` is the absolute onset density and is what the
        # director scores on: AdaptiveRange rescales whatever it is fed to fill
        # 0-1, measured at 2.4x on real audio, and candidates are compared
        # against each other so they need a scale that does not move.
        rate_raw = float(self.onset_rate.update(1.0 if beat else 0.0))
        rate_norm = self.onset_range.update(rate_raw)
        # Fixed full scale, not an adaptive one. The envelope saturates near
        # 0.5 on continuously percussive material (measured max 0.505 over 43 s
        # of real audio), so half the 0-1 range would otherwise never be used
        # and the onset-scored candidates could not compete. The point of the
        # change was to stop the mapping moving with recent history, not to
        # leave headroom unused: the same audio now always gives the same
        # number.
        rate_abs = float(np.clip(rate_raw / ONSET_RATE_FULL_SCALE, 0.0, 1.0))
        energy = float(np.clip(
            0.5 * self.level_range.update(slow)
            + 0.3 * (1.0 - narrow)
            + 0.2 * rate_norm,
            0.0, 1.0))
        if scene.available:
            # What is playing says more about how energetic to be than the
            # running statistics do. Percussion and electronic music want a
            # spectrum; an orchestra wants a wash, however loud it gets.
            driven = max(scene.get("percussion"), scene.get("electronic"),
                         scene.get("loud"))
            sustained = max(scene.get("orchestral"), scene.get("acoustic"))
            bias = float(np.clip(0.5 + 0.5 * (driven - sustained), 0.0, 1.0))
            # Weighted by how strong an opinion the classifier actually holds.
            #
            # ``bias`` is built from the *difference* of two group scores, so
            # when neither group fires it is not a judgement of 0.5, it is a
            # shrug that happens to be worth 0.5. Blending a flat 0.7 of that
            # in pinned energy near the middle whatever the audio did: measured
            # live, YAMNet reported ``music`` 0.5 with every other group at 0.0
            # while energy read 0.454 against a volume of 0.009. That constant
            # is a quarter of the vector the director switches on, which is a
            # large part of why it switched on the clock instead of on the
            # music. Scaling by the strongest musical group means an absent or
            # undecided classifier now changes nothing at all.
            opinion = max(driven, sustained)
            w = self.settings.mood.scene_weight * opinion
            energy = float(np.clip((1 - w) * energy + w * bias, 0.0, 1.0))

        self.features = Features(
            mel=self.mel, volume=self.volume, onset=onset, beat=beat,
            flux=flux, t=elapsed, silent=False,
            centroid=centroid, centroid_hz=centroid_hz, dialogue=dialogue, slow=slow,
            spread=spread, energy=energy, scene=scene,
            onset_rate=rate_abs, brightness=float(1.0 - narrow),
            percussive=percussive_rate, stems=self.stems, image=self.image,
        )
        return self._to_strip(self.effect.render(self.features))

    def _ensure_separator(self) -> None:
        """Start the separator the first time a non-zero weight asks for it.

        Built once and then left running even if the weight returns to zero:
        loading the weights takes seconds, and a slider the user is sweeping
        must not reload a model on every step. Idle it costs one memcpy per
        frame in :meth:`push`.

        Construction happens on its own thread because it can block for a long
        time -- the weights are a few hundred MB and are downloaded on first
        use -- and :meth:`apply` runs on the audio thread, which must never
        stall. Until it finishes, ``stems`` simply reports nothing, which is
        the same state as torch being absent.
        """
        if self.separator is not None or self.settings.mood.stem_weight <= 0.0:
            return
        s = self.settings

        def build() -> None:
            made = try_create_stems(s.audio.rate, window=s.mood.stem_window,
                                    interval=s.mood.stem_interval)
            if made is not None:
                self.separator = made

        threading.Thread(target=build, daemon=True, name="ambviz-stems-init").start()

    def _stereo_image(self, pad: int, side_spectrum: np.ndarray | None) -> StereoImage:
        """Low/mid/high energy for each channel on its own.

        Two more transforms than the mono path needs, which is why it is gated
        on the rig having sides: measured at about 60 us together, against a
        16.7 ms frame.
        """
        if self._roll_lr is None:
            return StereoImage()
        bands = []
        for ch in range(2):  # noqa: B007 - ch is used inside the loop body
            data = np.concatenate(self._roll_lr[ch], axis=0).astype(np.float32)
            spec = np.abs(np.fft.rfft(np.pad(data * self._window, (0, pad))))
            mel = self.mel_bank.apply(spec) ** self.settings.dsp.mel_exponent
            mel = mel / np.maximum(self.mel_gain.value, EPS)
            n = max(1, len(mel) // 3)
            bands.append((tuple(float(np.clip(np.mean(mel[a:b]), 0.0, 1.0))
                                for a, b in ((0, n), (n, 2 * n), (2 * n, len(mel)))),
                          np.clip(self._channel_mel[ch].update(mel), 0.0, 1.5),
                          self._centroid(self._normalise(mel))))
        side_mel = None
        side_level = 0.0
        if side_spectrum is not None:
            sm = self.mel_bank.apply(side_spectrum) ** self.settings.dsp.mel_exponent
            side_mel = np.clip(self._side_mel.update(
                sm / np.maximum(self.mel_gain.value, EPS)), 0.0, 1.5)
            side_level = float(np.clip(np.max(side_mel), 0.0, 1.0))
        return StereoImage(side_mel=side_mel, side_level=side_level,
                           left=bands[0][0], right=bands[1][0],
                           left_mel=bands[0][1], right_mel=bands[1][1],
                           left_centroid_hz=bands[0][2], right_centroid_hz=bands[1][2],
                           available=self.stereo)

    def _smooth_stems(self) -> Stems:
        """Smooth the published shares, then renormalise.

        Filtering each share independently lets them drift off summing to one,
        and a "share" that does not is a number nobody can reason about.
        """
        if self.separator is None:
            return Stems()
        raw = self.separator.stems
        if not raw.available or not raw.shares:
            return raw
        out = {}
        for name, value in raw.shares.items():
            f = self._stem_filter.get(name)
            if f is None:
                # Seed on the first real reading rather than ramping from zero,
                # which would otherwise spend the smoothing window claiming the
                # music has no drums in it.
                f = ExpFilter(value, alpha_decay=self._stem_alpha,
                              alpha_rise=self._stem_alpha)
                self._stem_filter[name] = f
            out[name] = float(f.update(value))
        total = sum(out.values())
        if total > 1e-9:
            out = {k: v / total for k, v in out.items()}
        # `change` is carried, not recomputed. It describes the step the
        # separator just took, and smoothing the shares says nothing about it
        # -- dropping it here left the whole arrangement trigger reading 0.0
        # for ever, so it never once fired.
        return Stems(shares=out, available=True, device=raw.device,
                     inference_ms=raw.inference_ms, change=raw.change)

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
            tail = half[:, 1:] if self.render_pixels % 2 else half
            pixels = np.concatenate((half[:, ::-1], tail), axis=1)
        else:
            pixels = half
        pixels = self._to_room(pixels)
        pixels = pixels * self.settings.effect.brightness
        return np.clip(pixels, 0, 255)

    def _to_room(self, frame: np.ndarray) -> np.ndarray:
        """Lay the rendered front out, and wash the other sides from it.

        The front is the picture. The sides are the light coming off it, which
        is a different job: they carry colour and level, not structure. Copying
        the animation onto them gives a room with three competing focal points
        and makes position mean frequency on a wall the listener is beside
        rather than facing.
        """
        if len(self.segments) == 1:
            return frame if frame.shape[1] == self.n_pixels else np.stack(
                [interpolate(c, self.n_pixels) for c in frame])
        self._update_accents()
        out = []
        for i, n in enumerate(self.segments):
            if i == self.front:
                out.append(frame)
            else:
                # Sides before the front take the left end, sides after it the
                # right, so each wall is lit by the corner it actually meets.
                out.append(self._side(frame, n, left=i < self.front))
        return np.concatenate(out, axis=1)

    def _side(self, frame: np.ndarray, n: int, left: bool) -> np.ndarray:
        """One side wall, by a strict order of preference.

        The sides run a real instance of the front's animation at their *own*
        width, rather than a gradient drawn to fit. That is what centres them:
        a mirrored effect folds about the middle of whatever width it was built
        for, so one built at 30 is centred on a 30-pixel wall instead of
        inheriting the front's centre. It also scales the travel speed by the
        length ratio -- 16 px/s crosses a 60-pixel front in 3.75 s and a
        30-pixel side in half that, so an uncorrected clone races.

        Three sources, best first:

        1. **The channel itself.** When the two channels are carrying different
           material, each wall runs the animation driven purely by its own
           channel. A guitar panned hard right then plays out on the right wall
           and nowhere else, which is the one thing a strip in front of you
           cannot say.
        2. **The runner-up stem.** When the channels agree there is no stereo
           story to tell, so the walls show what the mix contains that the
           front is not already drawing -- the second most prominent source,
           in its own colour. Not the loudest: that is most of what the front
           *is*.
        3. **The front's own analysis.** Nothing else to go on, so run the same
           animation the front is running, sized and paced for a short wall.
        """
        return self._finish_side(frame, self._ambient_side(frame, n, left), n, left)

    def _finish_side(self, frame: np.ndarray, out: np.ndarray,
                     n: int, left: bool) -> np.ndarray:
        """Hold a wall behind the front, then let an event cut through it.

        The cap and the accent are opposites and have to be applied in that
        order. The cap keeps the wall's *ambience* below the centre so the room
        has a focus; the accent is the moment the music does something and a
        wall answers, and it is meant to out-shine the room briefly. Capped
        along with the ambience it peaked at 68 against the front's 114 and read
        as a slight brightening rather than as a hit.

        The cap is measured on the *clipped* front, because that is the front
        the strip will actually show: an effect that overshoots 255 loses the
        excess a moment later, and a cap taken before the clip is set against a
        brightness the front never reaches.
        """
        cfg = self.settings.output
        front_mean = float(np.mean(np.clip(frame, 0.0, 255.0)))
        mean = float(np.mean(out))
        if front_mean > EPS and mean > cfg.side_brightness * front_mean:
            out = out * (cfg.side_brightness * front_mean / mean)

        level = self._accent[0 if left else 1]
        if cfg.accent_strength > 0.0 and level > 0.0:
            burst = self._accent_burst(n, left)
            if burst is None:
                # Brightest at the corner it shares with the front, so a hit
                # reads as arriving from the music rather than switching on.
                burst = 255.0 * np.linspace(1.0, 0.35, n)[None, :] * np.ones((3, 1))
            # Crossfaded, never added.
            #
            # There is no headroom to add into: the strip already runs at full
            # brightness, so an additive accent simply clips and a wall that is
            # already lit cannot get brighter. What makes the event visible is
            # that the wall's *ambience* is deliberately held at
            # side_brightness of the front, which leaves the top of the range
            # free for the accent to climb into -- and that the burst is a
            # travelling animation, so it reads as motion rather than as level.
            mix = cfg.accent_strength * level
            out = out * (1.0 - mix) + burst * mix

        # At the song's biggest moments the room stops being a focus and its
        # surroundings, and becomes one surface. Applied last, so it overrides
        # both the wall's own content and any accent running on it -- during a
        # drop the walls agreeing with the centre is the whole effect, and a
        # burst crossing one of them would only break it up.
        unison = self._unison_amount()
        if unison > 0.0:
            mirrored = (frame if frame.shape[1] == n
                        else np.stack([interpolate(c, n) for c in frame]))
            out = out * (1.0 - unison) + mirrored * unison
        # Pixel 0 of the left run is the far end of that wall, so it is wired
        # away from the corner -- flip it to match.
        return out[:, ::-1] if left else out

    def _accent_burst(self, n: int, left: bool) -> np.ndarray | None:
        """One frame of the burst animation, sized and paced for this wall.

        Paced to cross the wall once per ``accent_decay``, whatever the wall's
        length and whatever the library's own travel speed is. A burst that has
        not arrived by the time it fades has not happened -- at the shared
        16-24 px/s a scroll needs two and a half seconds to cross a side, and
        the whole event lasts well under one.
        """
        cfg = self.settings.output
        if not cfg.accent_animation:
            return None
        half = (n + 1) // 2 if self.mirror else n
        key = (half, left)
        effect = self._burst_effects.get(key)
        if effect is None:
            effect = EFFECTS[cfg.accent_animation](self.settings, half)
            self._burst_effects[key] = effect
        # Cross the wall once per accent, so the burst arrives exactly as the
        # envelope peaks and is gone as it fades.
        want = half / max(self.accent_length, 1e-3)
        rescale_clocks(effect, want / max(self.settings.effect.travel_pixels_per_second, 1e-6))
        out = effect.render(self.features)
        if self.mirror:
            tail = out[:, 1:] if n % 2 else out
            out = np.concatenate((out[:, ::-1], tail), axis=1)
        if out.shape[1] != n:
            out = np.stack([interpolate(c, n) for c in out])
        # Normalised to full scale. The burst replaces the wall's ambience
        # rather than adding to it, so at the ambience's own brightness the
        # event stopped reading at all -- a wall clearly out-shone the front on
        # 21% of frames with a plain brightness lift and on 1% with a burst
        # drawn at its natural level. What makes it an event is that it arrives
        # loud; the envelope decides for how long.
        peak = float(np.max(out))
        if peak > EPS:
            out = out * (255.0 / peak)
        return out

    def _ambient_side(self, frame: np.ndarray, n: int, left: bool) -> np.ndarray:
        """What the wall shows between events."""
        cfg = self.settings.output
        if cfg.side_mode != "stereo":
            return self._wash(frame, n, left)

        image = self.features.image
        # Strictly greater, so the ends of the range mean what they say: at 0.0
        # a mono source (difference exactly 0) still must not split, and at 1.0
        # nothing can, since difference is clipped to 1.
        if image.available and image.difference > cfg.stereo_threshold:
            if (image.left_mel if left else image.right_mel) is not None:
                return self._render_side(n, left, self._channel_features(left))

        width = self._width_features(left)
        if width is not None:
            return self._render_side(n, left, width)

        stems = self.features.stems
        name, share = stems.secondary() if stems.available else ("", 0.0)
        if name:
            # The separator publishes how much of each stem is present, not its
            # isolated audio, so the stem sets the wall's colour and how strongly
            # it reads while the front's own analysis still supplies the motion.
            out = self._render_side(n, left, self.features)
            return self._tint(out, HUES.get(name, 0.5), 0.55 + 0.45 * share)

        return self._render_side(n, left, self.features)

    #: Width below which the difference signal is not worth showing.
    MIN_WIDTH = 0.02

    #: How far above ``unison_threshold`` the music must get for the room to be
    #: fully in unison. Ramping over the whole remaining range instead meant a
    #: peak passage reached 0.59 and the walls never actually met the centre --
    #: the effect existed and was never seen.
    UNISON_RAMP = 0.12

    #: How far the music must fall back before another uplift can fire. Deeper
    #: than the ramp on purpose: re-arming just under the trigger meant the
    #: next uplift fired before the last had faded, and they overlapped into a
    #: continuous 73% -- the sustained state the event form exists to avoid.
    UNISON_RELEASE = 0.22

    @property
    def fullness(self) -> float:
        """How much is going on at once, 0-1.

        Energy and onset density say how *hard* a passage is working. How many
        instruments are playing at the same time says how *full* it is, and a
        spectrum cannot see that -- a wall of one loud synth and a full band
        look much alike to it. That part comes from the separator, where a
        second of lag costs nothing because this is a passage, not an event.
        """
        f = self.features
        full = 0.5 * f.energy + 0.5 * f.onset_rate
        stems = f.stems
        if stems.available and stems.shares:
            shares = list(stems.shares.values())
            # Present, not merely non-zero: a stem the model is unsure about
            # sits near a few percent whatever is playing.
            active = sum(1 for v in shares if v > 0.12) / max(len(shares), 1)
            full = 0.6 * full + 0.4 * min(1.0, active * 1.8)
        return float(np.clip(full, 0.0, 1.0))

    def _watch_for_a_new_track(self, t: float, silent: bool) -> None:
        """Re-baseline when a gap says the song changed.

        Playback does not stop between songs, so everything learned about how
        loud *this* one gets would otherwise carry into the next: a quiet track
        followed by a loud one produces an enormous surge in its first bar,
        measured against a floor belonging to a different piece of music.

        A gap is the only boundary visible without metadata, and it is not
        always there -- a gapless album runs on as one track. That is wrong and
        harmless: the floor still adapts within seconds, it simply does not get
        a fresh warm-up.
        """
        cfg = self.settings.output
        if cfg.track_gap <= 0:
            return
        if silent:
            if self._silence_since is None:
                self._silence_since = t
            return
        gap = t - self._silence_since if self._silence_since is not None else 0.0
        self._silence_since = None
        if gap < cfg.track_gap:
            return
        self.tracks += 1
        self.track_started = t
        # Everything the surge is judged against belonged to the last song.
        self._db_primed = False
        self._db_floor = -90.0
        self.surge_db = 0.0
        self._unison_armed = True
        self._unison_until = -1e9

    def _track_surge(self) -> float:
        """How far above its recent quiet the music has just jumped, in dB.

        The most reliable thing about a drop is what comes before it: songs
        build, strip back to almost nothing, then arrive all at once. Measuring
        the rise against how quiet it just was finds that moment far more
        dependably than any measure of how much is playing -- fullness is high
        through an entire chorus and says nothing about where the chorus began.
        """
        cfg = self.settings.output
        db = 20.0 * np.log10(max(float(self.features.volume), 1e-5))
        # Seeded on the first real level rather than ramping up from silence.
        # Starting the floor at -90 dB makes the opening of any track read as
        # an 80 dB surge and fires an uplift before a note has been heard.
        if not self._db_primed:
            if self.features.silent:
                return 0.0
            self._level_db.value = db
            self._db_floor = db
            self._db_primed = True
        db = float(self._level_db.update(db))
        rise = cfg.uplift_floor_recovery / max(self.settings.audio.fps, 1)
        self._db_floor = min(db, self._db_floor + rise)
        self.surge_db = float(max(0.0, db - self._db_floor))
        return self.surge_db

    def _unison_amount(self) -> float:
        """How far the room should currently act as one surface, 0-1.

        Fired on the *rise* into a full passage and held for a set number of
        beats, rather than held for as long as the music stays loud. Tracking
        the level meant a chorus put the room in unison for its whole duration,
        and an uplift that does not end is not an uplift -- it is just the room
        with its focus removed for thirty seconds.

        It re-arms only once the music has dropped back under the threshold, so
        a long loud section lifts once at its start instead of pulsing.
        """
        cfg = self.settings.output
        if cfg.unison_threshold >= 1.0:
            return float(self._unison.update(0.0))

        full = self.fullness
        surge = self._track_surge()
        t = self.features.t
        # Not while one is already running. Re-firing pushed `_unison_until`
        # further out on every frame the music stayed loud, so the hold never
        # expired and the room sat in unison 73% of the time -- exactly the
        # sustained state the event form was meant to replace.
        # The surge fires it; fullness only vouches for it.
        #
        # Fullness on its own is a poor trigger: it is high through an entire
        # chorus and says nothing about where the chorus began, so it put the
        # room in unison for 79% of an ordinary verse. The rise out of a quiet
        # passage is the thing that actually marks the moment, and requiring
        # the result to also be full stops a lone loud noise in an empty
        # passage from claiming one.
        #
        # Nothing fires during the warm-up: the surge is measured against how
        # quiet the song has recently been, and at the start there is no
        # "recently" to measure against.
        # Two ways in, and they are not the same bar.
        #
        # The surge finds a song arriving out of a breakdown, which is what
        # most uplifts are; fullness has to vouch for the result so a lone loud
        # noise in an empty passage cannot claim one. Separately, a passage can
        # simply be enormous -- a last chorus with every part playing, arrived
        # at gradually -- and then there is no surge to find because nothing
        # ever got quiet. That needs a far higher bar: at the gate's own level
        # fullness alone lifted the room for 79% of an ordinary verse.
        #
        # Neither fires during the warm-up: the surge is measured against how
        # quiet this track has been, and at its start there is no "recently".
        ready = t - self.track_started >= cfg.uplift_warmup
        by_surge = (cfg.uplift_surge_db > 0 and surge >= cfg.uplift_surge_db
                    and full >= cfg.unison_threshold)
        by_fullness = full >= cfg.unison_full_trigger
        lifted = ready and (by_surge or by_fullness)
        if self._unison_armed and lifted and t >= self._unison_until:
            self._unison_until = t + cfg.unison_beats * self.beat_period
            self._unison_armed = False
            self.unison_events += 1

        # Held for a minimum, then for as long as the music stays lifted.
        #
        # `unison_beats` is a floor rather than a limit: it stops a single loud
        # bar from producing a flicker, and after it the room stays in unison
        # until the song comes back down -- either because it went quiet or
        # because the surge has decayed as the floor climbed back to meet it.
        # Cutting off mid-chorus on a fixed count looked like the effect
        # breaking rather than the music letting go.
        holding = t < self._unison_until or lifted
        if not holding and float(self._unison.value) < 0.1:
            # Re-arm only once it has actually finished, or the next lift
            # begins before the room has visibly returned.
            self._unison_armed = True
        return float(np.clip(self._unison.update(1.0 if holding else 0.0), 0.0, 1.0))

    @property
    def accent_length(self) -> float:
        """How long an accent lasts, in seconds, at the music's current tempo.

        One number, driving both the fade and how fast the burst crosses the
        wall, so the whole event scales with the song. Fixed in seconds it was
        tuned for one tempo and wrong at every other: right on fast material,
        where a beat is about as long as the fade, and a crawl on anything
        slower, where the burst lost its connection to the beat entirely.
        """
        cfg = self.settings.output
        return max(cfg.accent_decay, cfg.accent_length_beats * self.beat_period)

    def _update_accents(self) -> None:
        """Decide which wall answers this frame, and fade the last one.

        The width the walls carry is true to the audio and, alone, barely moves
        -- it is the mix's reverb tail, not its rhythm. This is the half that
        makes a room feel like it is listening: when something actually happens,
        one wall answers, and the *next* event answers on the other side.

        Alternating is the whole trick. Both walls lighting together is just a
        brighter room; taking turns makes the music appear to cross the space,
        which is the one thing a pair of side walls can do that a strip in front
        of you cannot.

        Only events that stand out get a turn -- every hi-hat taking one would
        blur the alternation into a flicker.
        """
        cfg = self.settings.output
        f = self.features
        fps = max(self.settings.audio.fps, 1)
        decay = 1.0 / max(self.accent_length * fps, 1.0)
        rise = 1.0 / max(cfg.accent_attack * fps, 1.0)
        for i in range(2):
            # The target falls away; the visible level chases it. Setting the
            # level directly made every accent a hard edge, and the eye reads
            # the edge rather than the envelope -- it looked like a strobe
            # however long the fade was.
            self._accent_target[i] = max(0.0, self._accent_target[i] - decay)
            gap = self._accent_target[i] - self._accent[i]
            self._accent[i] += min(gap, rise) if gap > 0 else max(gap, -decay)
        if cfg.accent_strength <= 0.0 or f.silent:
            return

        # A beat that stands out, or a change of character big enough that the
        # director would consider switching on it -- the two kinds of "something
        # happened" this system already knows how to detect.
        # A change of character is an *edge*, not a state.
        #
        # Testing `drift > threshold` fires on every frame the audio stays away
        # from its anchor, which on real music is most of them: it swamped the
        # onset path completely and left the walls lit 79% of the time whatever
        # accent_threshold was set to. Only the crossing is an event.
        drift = getattr(self.effect, "drift", 0.0)
        above = drift > self.settings.mood.change_threshold
        changed = above and not self._accent_drifted
        self._accent_drifted = above

        # The musical trigger: the arrangement itself moved. Slower than an
        # onset and far more meaningful -- it is drums dropping out or a guitar
        # arriving, not a hi-hat.
        stems = f.stems
        arranged = False
        if stems.available and cfg.accent_stem_change > 0.0:
            if self._accent_stems is None or stems.change != self._accent_stems:
                arranged = stems.change >= cfg.accent_stem_change
                self._accent_stems = stems.change

        strong = f.beat and f.onset >= cfg.accent_threshold
        if not (strong or changed or arranged):
            return
        # Paced in beats, not seconds.
        #
        # A fixed gap is right at one tempo and wrong at every other: it looked
        # correct on fast material, where a beat is about as long as the fade,
        # and fired far too often on anything slower because the detector still
        # finds the subdivisions between beats and each one took a turn.
        gap = max(self.accent_length, cfg.accent_beats * self.beat_period)
        # An arrangement change is rare enough to be worth interrupting for.
        if not arranged and f.t - self._accent_at < gap:
            return
        self._accent_at = f.t
        side = self._accent_next if cfg.accent_alternate else None
        # Near full, whatever the onset measured.
        #
        # Using the onset's own strength as the envelope height meant a hit that
        # had already cleared accent_threshold still showed at a third of the
        # wall, because onset strength typically lands at 0.3-0.6 -- not one
        # frame in eighteen seconds reached the top of the envelope. The
        # threshold decides *whether* this is an event; once it is one, it is
        # worth seeing. A little headroom is left so a hard hit still reads
        # harder than a soft one.
        strength = float(np.clip(0.8 + 0.2 * max(f.onset, 0.5), 0.0, 1.0))
        if side is None:
            self._accent_target = [strength, strength]
        else:
            self._accent_target[side] = strength
            self._accent_next = 1 - side
        self.accent_events += 1

    def _width_features(self, left: bool) -> Features | None:
        """The frame as the *difference* signal saw it: everything not centred.

        This is what a side wall in a room is for, and it is the thing the
        earlier attempts kept missing. Comparing the two channels says almost
        nothing about a normal mix -- measured band imbalance of 0.006 to
        0.012, because a modern master puts nearly the same magnitude spectrum
        in both speakers and separates them by phase. But ``L - R`` isolates
        exactly what is *not* shared: the reverb, the spread, the wide synths.
        Its spectral shape differs from the mix's by a wide margin (0.678
        similarity, against the 0.99 the channels themselves manage), so a wall
        driven by it finally shows something the front is not already showing.

        It is the same principle a surround decoder uses to feed rear speakers,
        and the walls are exactly where the rears would stand.

        Returns None when the mix is too narrow to have anything to say, so the
        stem takes over rather than the walls displaying silence.
        """
        image = self.features.image
        if image.side_mel is None or image.side_level < self.MIN_WIDTH:
            return None
        mel = image.side_mel

        # What little panning there is decides which wall leans brighter. It is
        # a small correction on most material and the whole story on some.
        mine = image.left_mel if left else image.right_mel
        other = image.right_mel if left else image.left_mel
        if mine is not None and other is not None:
            pan = (mine - other) / np.maximum(mine + other, EPS)
            mel = mel * np.clip(1.0 + self.settings.output.stereo_emphasis * pan,
                                0.0, 2.0)

        # Lifted to the front's scale before side_brightness decides how far
        # behind it sits. The difference signal is a fraction of the mix by
        # construction, so left raw the walls would simply be dark.
        peak = float(np.max(mel))
        if peak > EPS:
            mel = mel * (float(np.max(self.features.mel)) / peak)
        hz = self._centroid(self._normalise(mel))
        idx = 0 if left else 1
        return replace(
            self.features,
            mel=mel,
            centroid_hz=hz,
            centroid=float(self.channel_range[idx].update(hz)),
            energy=float(np.clip(image.side_level * 3.0, 0.0, 1.0)),
            slow=image.side_level,
        )

    def _channel_features(self, left: bool) -> Features:
        """The frame as one channel alone saw it.

        Every field an effect reads for shape *or colour* is replaced, not just
        the filterbank. Swapping the bands alone left both walls taking their
        hue from the mixed centroid, so they came out the same colour differing
        only in brightness -- one picture drawn twice, which is exactly what the
        sides exist not to be. Measured on real audio, that alone had the two
        walls 96.6% identical while the channels themselves differed by 0.65.

        The beat stays shared. An onset is an event in the music, not in a
        channel, and letting the walls disagree about when the track hits would
        pull the room apart.
        """
        image = self.features.image
        idx = 0 if left else 1
        mine = image.left_mel if left else image.right_mel
        other = image.right_mel if left else image.left_mel
        # What this side has that the other does not.
        #
        # Driving a wall from everything its speaker plays is the obvious
        # reading and the wrong one: both channels carry nearly the whole
        # arrangement, so both walls end up redrawing the front. Subtracting the
        # opposite channel leaves only the panned material, which is the part
        # that actually says where the mix is standing.
        k = self.settings.output.stereo_emphasis
        mel = np.clip(mine - k * other, 0.0, None) if k else mine
        # Deliberately not rescaled. A wall with nothing panned to it *should*
        # go dark -- that darkness is the whole message, and normalising it
        # away would light both walls equally for a centred mix, which is the
        # thing this path exists to avoid.
        level = float(np.clip(np.mean(mel) * 3.0, 0.0, 1.0))
        hz = self._centroid(self._normalise(mel)) if k else (
            image.left_centroid_hz if left else image.right_centroid_hz)
        return replace(
            self.features,
            mel=mel,
            centroid_hz=hz,
            centroid=float(self.channel_range[idx].update(hz)),
            energy=float(np.clip(level, 0.0, 1.0)),
            slow=level,
        )

    def _render_side(self, n: int, left: bool, features: Features) -> np.ndarray:
        """Run the front's animation at one side's width, on given features."""
        half = (n + 1) // 2 if self.mirror else n
        # Keyed by side as well as width. Effects carry state -- smoothed band
        # levels, scroll histories, peak markers -- so one instance rendering
        # both walls has its filters fed the left channel and the right channel
        # alternately, and converges on their average. The two walls then show
        # the same picture however different the channels are, which defeats
        # the entire point of having two.
        # A fixed, quiet animation rather than whatever the front is running.
        # Cloning the front onto three surfaces gives a room with three
        # competing focal points and no centre: the front is the thing to look
        # at, the sides are what you see around it. Set side_animation to "" to
        # follow the front instead.
        name = (self.settings.output.side_animation
                or getattr(self.effect, "current", None)
                or self.settings.effect.name)
        key = (half, left, name)
        effect = self._side_effects.get(key)
        if effect is None:
            effect = EFFECTS[name](self.settings, half)
            self._side_effects[key] = effect
        # Paced for the shorter run, so the crossing time matches the front --
        # and re-derived every frame rather than baked in when the effect was
        # built. The rig can be re-described live: change `output.segments` or
        # toggle `effect.mirror` and both widths move, so a ratio captured at
        # construction is wrong from the next patch onward.
        rescale_clocks(effect, max(half, 1) / max(self.width, 1))
        out = effect.render(features)
        if self.mirror:
            tail = out[:, 1:] if n % 2 else out
            out = np.concatenate((out[:, ::-1], tail), axis=1)
        if out.shape[1] != n:
            out = np.stack([interpolate(c, n) for c in out])
        # Left/right orientation is applied once, in _finish_side.
        return out

    @staticmethod
    def _tint(rgb: np.ndarray, hue: float, amount: float) -> np.ndarray:
        """Pull a rendered frame toward one hue, keeping its brightness.

        Recolouring rather than overwriting: the animation's shape is the part
        worth keeping, and its own hues are what say which *instrument* the
        wall is standing in for.
        """
        value = rgb.max(axis=0) / 255.0
        target = hsv_to_rgb(np.full(rgb.shape[1], hue), 0.85, value) * 255.0
        a = float(np.clip(amount, 0.0, 1.0))
        return np.clip(rgb * (1.0 - a) + target * a, 0.0, 255.0)

    def _wash(self, frame: np.ndarray, n: int, left: bool) -> np.ndarray:
        """Blurred light off one end of the front, laid along a side wall."""
        cfg = self.settings.output
        w = frame.shape[1]
        span = max(2, int(round(w * cfg.wash_span)))
        edge = frame[:, :span] if left else frame[:, -span:]
        # Blur hard enough that no band survives as a band. Structure on a wall
        # beside the listener reads as a second, wrong spectrum display.
        sigma = max(0.5, span * cfg.wash_softness)
        edge = gaussian_filter1d(edge, sigma=sigma, axis=1, mode="nearest")
        # Orient so pixel 0 of each run is the end the chain arrives at: the
        # left wall is wired back-to-front, the right wall front-to-back, which
        # is how a single chain actually goes round a room.
        return np.stack([interpolate(c, n) for c in edge])

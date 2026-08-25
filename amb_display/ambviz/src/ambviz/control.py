"""Live control: validated setting changes handed from the API to the run loop.

The API runs on its own thread; the visualizer runs on the audio thread. The API
must never touch :class:`~ambviz.pipeline.Visualizer` directly -- changing the
effect swaps an object, and changing the frequency range reallocates the Mel
matrix, so a write landing mid-frame is a torn read.

Instead the API validates a patch here and enqueues it; the run loop drains the
queue between frames and applies it. All mutation therefore happens on one
thread, and an invalid patch is rejected before it is ever queued, so the run
loop cannot be handed a value that would crash it.

Standard library plus :mod:`ambviz.settings` only, so :mod:`ambviz.api` can
import this without pulling in numpy.
"""

from __future__ import annotations

import copy
import queue
from typing import Any

from ambviz.settings import Settings

CONTROLLABLE: frozenset[str] = frozenset({
    # what the effect looks like
    "effect.name",
    "effect.brightness",
    "effect.mirror",
    "effect.scroll_decay",
    "effect.scroll_sigma",
    "effect.energy_scale",
    "effect.energy_sigma",
    # how the audio is analysed
    "dsp.min_frequency",
    "dsp.max_frequency",
    "dsp.fft_bins",
    "dsp.mel_exponent",
    "dsp.gain_sigma",
    "dsp.onset_sensitivity",
    "dsp.onset_refractory",
    "dsp.vocal_suppression",
    "dsp.vocal_band",
    # the slow layer
    "mood.response_seconds",
    "mood.hue_rate",
    "mood.deadband",
    "mood.floor",
    "mood.dialogue_damping",
    "mood.detail",
    "mood.audio_weight",
    "mood.scene_weight",
    # how hard the smoothing bites
    "smoothing.red",
    "smoothing.green",
    "smoothing.blue",
    "smoothing.common_mode",
    "smoothing.pixel",
    "smoothing.gain",
    "smoothing.mel_gain",
    "smoothing.mel_smoothing",
    # output trim that does not resize anything
    "output.gamma_correction",
    "output.full_refresh_interval",
})
"""Settings a client may change while running.

Everything else -- pixel count, sample rate, FPS, host, port, audio source --
resizes buffers or reopens devices and stays restart-only.
"""


class NotControllable(KeyError):
    """Raised for a setting that exists but cannot be changed at runtime."""


class CommandQueue:
    """Validated, thread-safe hand-off of setting patches to the run loop."""

    def __init__(self, settings: Settings):
        self.settings = settings
        # Patches are validated against, and accumulate on, a private copy --
        # never on `settings`, which the audio thread is reading. Mutating that
        # from here would let e.g. dsp.fft_bins change before the Mel bank is
        # rebuilt, and the next frame would blow up on a shape mismatch.
        # The copy also makes queued patches compose: two changes in flight each
        # see the other.
        self._pending = copy.deepcopy(settings)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()

    @property
    def pending(self) -> Settings:
        """Settings as they will be once the run loop drains the queue."""
        return self._pending

    # ── API thread ───────────────────────────────────────────────────────────
    def submit(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Validate ``patch`` and queue it. Returns the patch as accepted.

        Raises :class:`NotControllable` for a runtime-immutable setting,
        :class:`KeyError` for an unknown one, and :class:`ValueError` if the
        result would be an invalid configuration -- all *before* enqueuing.
        """
        if not isinstance(patch, dict) or not patch:
            raise ValueError("patch must be a non-empty object")

        flat = []
        for section, values in patch.items():
            if not isinstance(values, dict):
                raise NotControllable(
                    f"{section!r} is not a settings section; expected e.g. "
                    f'{{"effect": {{"brightness": 0.5}}}}'
                )
            for key in values:
                flat.append(f"{section}.{key}")

        # Reject immutables by name before anything else, so the message is
        # about the real problem rather than a downstream validation failure.
        for name in flat:
            if name in CONTROLLABLE:
                continue
            section, _, key = name.partition(".")
            candidate = getattr(Settings(), section, None)
            if candidate is None or not hasattr(candidate, key):
                raise KeyError(f"unknown setting: {name}")
            raise NotControllable(
                f"{name} cannot be changed while running; it is set at startup"
            )

        # Dry-run first: if the result would not validate, nothing is queued and
        # the pending state is left untouched.
        trial = copy.deepcopy(self._pending)
        trial._apply(patch)
        trial.validate()

        self._pending = trial
        self._queue.put(copy.deepcopy(patch))
        return patch

    # ── audio thread ─────────────────────────────────────────────────────────
    def drain(self) -> list[dict[str, Any]]:
        """Return every queued patch, oldest first, emptying the queue."""
        patches = []
        while True:
            try:
                patches.append(self._queue.get_nowait())
            except queue.Empty:
                return patches

    def __len__(self) -> int:
        return self._queue.qsize()

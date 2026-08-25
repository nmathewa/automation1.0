"""Effects: filterbank frame in, pixel values out.

Each effect owns its filter state, so two visualizers can run in one process
without corrupting each other -- which the module-level globals in the legacy
``visualization.py`` could not do.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ambviz.dsp import EPS, ExpFilter, interpolate
from ambviz.features import Features
from ambviz.settings import Settings


class Effect:
    """Maps one frame of analysis to ``(3, width)`` pixel values in 0-255.

    ``mirrored`` says how the effect behaves on a rig of several strips. An
    effect that maps frequency onto position wants each node to show the same
    thing; one that animates a travelling wave wants a per-node phase offset so
    the strips read as one system rather than two copies.
    """

    #: False for effects that should be phase-offset per node rather than cloned.
    clone_across_nodes = True

    def __init__(self, settings: Settings, width: int):
        self.settings = settings
        self.width = width

    def render(self, features: Features) -> np.ndarray:
        raise NotImplementedError


class SpectrumEffect(Effect):
    """Maps the filterbank across the strip.

    Red carries spectral contrast (the level minus a slow-moving floor), green
    carries frame-to-frame change, blue carries the smoothed level.
    """

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        sm = settings.smoothing
        seed = np.tile(0.01, width)
        self.r_filt = ExpFilter.from_alpha(np.copy(seed), sm.red)
        self.g_filt = ExpFilter.from_alpha(np.copy(seed), sm.green)
        self.b_filt = ExpFilter.from_alpha(np.copy(seed), sm.blue)
        self.common_mode = ExpFilter.from_alpha(np.copy(seed), sm.common_mode)
        self._prev = np.copy(seed)

    def render(self, features: Features) -> np.ndarray:
        y = interpolate(features.mel, self.width)
        self.common_mode.update(y)
        diff = y - self._prev
        self._prev = np.copy(y)
        r = self.r_filt.update(y - self.common_mode.value)
        g = self.g_filt.update(np.abs(diff))
        b = self.b_filt.update(np.copy(y))
        return np.array([r, g, b]) * 255.0


class EnergyEffect(Effect):
    """Bars growing from the origin, one per frequency third."""

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        sm = settings.smoothing
        self.p = np.tile(1.0, (3, width))
        self.p_filt = ExpFilter.from_alpha(np.tile(1.0, (3, width)), sm.pixel)
        self.gain = ExpFilter.from_alpha(np.tile(0.01, settings.dsp.fft_bins), sm.gain)

    def render(self, features: Features) -> np.ndarray:
        cfg = self.settings.effect
        y = np.copy(features.mel)
        self.gain.update(y)
        y = y / np.maximum(self.gain.value, EPS)
        y *= float(self.width - 1)
        third = len(y) // 3
        lengths = [
            int(np.mean(y[:third] ** cfg.energy_scale)),
            int(np.mean(y[third:2 * third] ** cfg.energy_scale)),
            int(np.mean(y[2 * third:] ** cfg.energy_scale)),
        ]
        for channel, n in enumerate(lengths):
            n = int(np.clip(n, 0, self.width))
            self.p[channel, :n] = 255.0
            self.p[channel, n:] = 0.0
        self.p_filt.update(self.p)
        self.p = np.round(self.p_filt.value)
        for channel in range(3):
            self.p[channel] = gaussian_filter1d(self.p[channel], sigma=cfg.energy_sigma)
        return np.copy(self.p)


class ScrollEffect(Effect):
    """New colour appears at the origin each frame and drifts outward, decaying."""

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.p = np.tile(1.0, (3, width))
        self.gain = ExpFilter.from_alpha(
            np.tile(0.01, settings.dsp.fft_bins), settings.smoothing.gain
        )

    def render(self, features: Features) -> np.ndarray:
        cfg = self.settings.effect
        y = features.mel ** 2.0
        self.gain.update(y)
        y = y / np.maximum(self.gain.value, EPS) * 255.0
        third = len(y) // 3
        head = [
            int(np.max(y[:third])),
            int(np.max(y[third:2 * third])),
            int(np.max(y[2 * third:])),
        ]
        self.p[:, 1:] = self.p[:, :-1]
        self.p *= cfg.scroll_decay
        self.p = gaussian_filter1d(self.p, sigma=cfg.scroll_sigma)
        self.p[:, 0] = head
        return np.copy(self.p)




# ── the library ──────────────────────────────────────────────────────────────
def _hsv_to_rgb(h: np.ndarray, s: float, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV to RGB, hue in turns. Returns ``(3, n)`` in 0-1."""
    i = np.floor(h * 6.0) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    out = np.zeros((3, len(h)))
    for k, (r, g, b) in enumerate(((v, t, p), (q, v, p), (p, v, t),
                                   (p, q, v), (t, p, v), (v, p, q))):
        m = i == k
        out[0][m], out[1][m], out[2][m] = np.broadcast_to(r, h.shape)[m], \
            np.broadcast_to(g, h.shape)[m], np.broadcast_to(b, h.shape)[m]
    return out


class BarsEffect(Effect):
    """One block per Mel band, brightness following that band's level.

    The graphic-equaliser reading of a strip: position is frequency, brightness
    is energy. Hue runs low-to-high across the strip so a band keeps its colour
    as its level moves.
    """

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.levels = ExpFilter(np.tile(0.01, settings.dsp.fft_bins),
                                alpha_decay=0.08, alpha_rise=0.9)

    def render(self, features: Features) -> np.ndarray:
        levels = np.clip(self.levels.update(np.copy(features.mel)), 0.0, 1.5)
        spread = interpolate(levels, self.width)
        hue = np.linspace(0.0, 0.75, self.width)
        return _hsv_to_rgb(hue, 1.0, np.clip(spread, 0.0, 1.0)) * 255.0


class GravcenterEffect(Effect):
    """Level pushes pixels out from the origin; they fall back under gravity.

    The peak is held by a falling marker rather than snapping down with the
    signal, which is what makes it read as physical instead of twitchy.
    """

    clone_across_nodes = True

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.peak = 0.0
        self.peak_velocity = 0.0
        self.level = ExpFilter(0.01, alpha_decay=0.2, alpha_rise=0.9)

    def render(self, features: Features) -> np.ndarray:
        low, mid, high = features.thirds()
        level = float(np.clip(self.level.update(low * 1.6 + mid * 0.8 + high * 0.4), 0, 1))
        height = level * self.width

        # Gravity on the peak marker: rise instantly, fall at 9.8 units/s^2
        # scaled to the strip, so it lags the music the way a real meter does.
        if height >= self.peak:
            self.peak, self.peak_velocity = height, 0.0
        else:
            self.peak_velocity += 9.8 * self.width / 3600.0
            self.peak = max(0.0, self.peak - self.peak_velocity)

        out = np.zeros((3, self.width))
        lit = int(np.clip(height, 0, self.width))
        if lit:
            ramp = np.linspace(0.0, 1.0, lit)
            out[0, :lit] = 255 * ramp
            out[1, :lit] = 255 * (1.0 - ramp) * 0.6
            out[2, :lit] = 255 * (1.0 - ramp)
        marker = int(np.clip(self.peak, 0, self.width - 1))
        out[:, marker] = 255.0
        return out


class WaterfallEffect(Effect):
    """Spectrum written at the origin, scrolling away as history.

    Position becomes time rather than frequency: what you see further along the
    strip is what the music did a moment ago.
    """

    clone_across_nodes = False

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.pixels = np.zeros((3, width))

    def render(self, features: Features) -> np.ndarray:
        self.pixels[:, 1:] = self.pixels[:, :-1] * self.settings.effect.scroll_decay
        low, mid, high = features.thirds()
        peak = max(low, mid, high)
        # Hue by which third dominates: bass red, mids green, highs blue.
        hue = 0.0 if peak == low else (0.33 if peak == mid else 0.66)
        colour = _hsv_to_rgb(np.array([hue]), 1.0, np.array([min(1.0, peak * 1.4)]))
        self.pixels[:, 0] = colour[:, 0] * 255.0
        return np.copy(self.pixels)


class PixelwaveEffect(Effect):
    """Beats launch a wave from the origin that travels outward and fades.

    Nothing happens between hits, which is the point: it follows rhythm rather
    than loudness, so it stays still through sustained passages.
    """

    clone_across_nodes = False

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.pixels = np.zeros((3, width))

    def render(self, features: Features) -> np.ndarray:
        self.pixels[:, 1:] = self.pixels[:, :-1]
        self.pixels *= 0.92
        if features.beat:
            low, mid, high = features.thirds()
            total = low + mid + high + EPS
            strength = 0.35 + 0.65 * features.onset
            self.pixels[:, 0] = np.array([low, mid, high]) / total * 255.0 * strength * 3
        else:
            self.pixels[:, 0] *= 0.7
        return np.clip(self.pixels, 0, 255)


class NoisemeterEffect(Effect):
    """A drifting noise field whose brightness follows the overall level.

    No structure to read -- it is texture, and the gentlest thing in the
    library. The closest the current set gets to an ambient wash.
    """

    clone_across_nodes = False

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.level = ExpFilter(0.01, alpha_decay=0.05, alpha_rise=0.4)

    def render(self, features: Features) -> np.ndarray:
        level = float(np.clip(self.level.update(max(features.thirds())), 0, 1))
        x = np.linspace(0, 4 * np.pi, self.width)
        # Three incommensurate sines never repeat, which reads as noise while
        # staying smooth and cheap -- no Perlin table required.
        field = (np.sin(x + features.t * 0.7)
                 + np.sin(x * 0.61 - features.t * 0.43)
                 + np.sin(x * 1.37 + features.t * 0.29)) / 3.0
        brightness = np.clip((field * 0.5 + 0.5) * level, 0, 1)
        hue = (features.t * 0.02 + np.linspace(0, 0.15, self.width)) % 1.0
        return _hsv_to_rgb(hue, 0.85, brightness) * 255.0


class SolidEffect(Effect):
    """A single colour whose brightness follows the level, hue set by balance.

    A baseline: the least distracting thing the strip can do while still
    responding, and a known-good reference when tuning something else.
    """

    clone_across_nodes = True

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.hue = ExpFilter(0.5, alpha_decay=0.02, alpha_rise=0.05)
        self.level = ExpFilter(0.01, alpha_decay=0.05, alpha_rise=0.5)

    def render(self, features: Features) -> np.ndarray:
        low, mid, high = features.thirds()
        total = low + mid + high + EPS
        # Bass pulls the hue red, treble pulls it blue.
        hue = float(self.hue.update(0.0 * low / total + 0.33 * mid / total + 0.66 * high / total))
        level = float(np.clip(self.level.update(max(low, mid, high)), 0, 1))
        rgb = _hsv_to_rgb(np.array([hue % 1.0]), 0.9, np.array([level]))
        return np.repeat(rgb, self.width, axis=1) * 255.0


EFFECTS: dict[str, type[Effect]] = {
    "spectrum": SpectrumEffect,
    "energy": EnergyEffect,
    "scroll": ScrollEffect,
    "bars": BarsEffect,
    "gravcenter": GravcenterEffect,
    "waterfall": WaterfallEffect,
    "pixelwave": PixelwaveEffect,
    "noisemeter": NoisemeterEffect,
    "solid": SolidEffect,
}

#: Which effects react to beats rather than only to level.
BEAT_DRIVEN = frozenset({"pixelwave"})

"""Effects: filterbank frame in, pixel values out.

Each effect owns its filter state, so two visualizers can run in one process
without corrupting each other -- which the module-level globals in the legacy
``visualization.py`` could not do.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ambviz.dsp import EPS, ExpFilter, interpolate
from ambviz.director import score_candidates
from ambviz.features import Features
from ambviz.mood import AudioMood, blend
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
        # Undo the analysis exponent for display. dsp.mel_exponent squares the
        # filterbank, which sharpens peaks for the feature extractors and is
        # exactly wrong for brightness: mapped straight through, 78% of band
        # values landed under 13/255 and the strip read as black with a couple
        # of lit pixels. Analysis and display want opposite curves here.
        gamma = 1.0 / max(self.settings.dsp.mel_exponent, 1e-6)
        value = np.clip(spread, 0.0, 1.0) ** gamma
        hue = np.linspace(0.0, 0.75, self.width)
        return _hsv_to_rgb(hue, 1.0, value) * 255.0


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


class CinemaEffect(Effect):
    """A wash that becomes a spectrum when the scene earns it.

    The mistake in the first version was treating "subtle" as the goal. It is
    not: a quiet conversation should barely move, and a fight scene should look
    close to ``bars`` -- just smoother. So this is a cross-fade rather than a
    wash with decoration on top.

    At one end, a single slowly drifting colour with a brightness floor. At the
    other, the Mel bands across the strip, smoothed far harder than ``bars``
    smooths them so the result reads as motion rather than flicker. Scene energy
    picks the point between, and dialogue pulls it back toward the wash.
    """

    clone_across_nodes = False

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.mood_source = AudioMood(settings)
        self.mood = None
        # Much heavier smoothing than BarsEffect uses: the same information,
        # moving at a pace that suits a film rather than a dance floor.
        self.bands = ExpFilter(np.tile(0.01, settings.dsp.fft_bins),
                               alpha_decay=0.02, alpha_rise=0.10)
        self._mix = ExpFilter(0.0, alpha_decay=0.01, alpha_rise=0.03)

    def render(self, features: Features) -> np.ndarray:
        cfg = self.settings.mood
        self.mood = blend([self.mood_source.update(features)])
        m = self.mood

        x = np.linspace(0.0, 1.0, self.width)
        # A gentle arch, so even a still strip has some shape to it.
        wash_value = max(m.level, cfg.floor) * (0.75 + 0.25 * np.cos((x - 0.5) * np.pi))
        wash_hue = (m.hue + np.linspace(-0.02, 0.02, self.width)) % 1.0

        # Smoothed spectrum, mapped the way bars maps it.
        levels = np.clip(self.bands.update(np.copy(features.mel)), 0.0, 1.5)
        spectral_value = np.clip(interpolate(levels, self.width), 0.0, 1.0)
        spectral_hue = (m.hue + np.linspace(-0.16, 0.16, self.width)) % 1.0

        # Cross-fade slowly, so the scene changing does not snap the look.
        mix = float(self._mix.update(m.detail))
        value = np.clip(wash_value * (1.0 - mix) + spectral_value * mix, 0.0, 1.0)
        value = np.maximum(value, cfg.floor * (1.0 - mix))

        # The accent is added after the cross-fade rather than folded into the
        # level, so a hit punches through whatever the slow layer is doing --
        # the whole point being that it should not have to wait for it.
        if m.accent > 0.0:
            punch = 0.5 + 0.5 * np.cos((x - 0.5) * np.pi)   # strongest at centre
            value = np.clip(value + m.accent * punch * 0.6, 0.0, 1.0)
        hue = np.where(mix > 0.5, spectral_hue, wash_hue)
        saturation = m.saturation * (1.0 - 0.15 * mix)
        return _hsv_to_rgb(hue, saturation, value) * 255.0


class PacificaEffect(Effect):
    """Layered ocean swells: gentle, wide, and slow.

    Ported from Mark Kriegsman's ``Pacifica`` in FastLED (MIT), which is four
    blue-green waves at different scales and speeds summed together, with
    whitecaps where they happen to coincide. The original is free-running; here
    the swell height follows the level, so a quiet scene barely moves and a
    loud one rolls.

    This is the ambient end of the library. Nothing in it maps frequency onto
    position, which is the point -- it is meant to be looked past rather than
    read.
    """

    clone_across_nodes = False

    #: (cycles across the strip, drift speed, weight) per layer.
    #:
    #: The original expresses scale in radians per *pixel*, which assumes a
    #: strip long enough for that to produce waves. On 60 pixels no layer
    #: completed even one cycle and the whole effect flattened to one teal bar
    #: -- measured spatial contrast 0.089 against 0.35-0.87 for the rest of the
    #: library. Cycles-across-the-strip makes it look the same at any width,
    #: which a rig of several nodes needs anyway.
    LAYERS = ((2.6, 0.34, 0.55), (1.9, -0.29, 0.40),
              (1.3, 0.21, 0.28), (0.8, -0.15, 0.22))

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.level = ExpFilter(0.05, alpha_decay=0.006, alpha_rise=0.02)

    def render(self, features: Features) -> np.ndarray:
        x = np.linspace(0.0, 1.0, self.width)
        swell = np.zeros(self.width)
        for cycles, speed, weight in self.LAYERS:
            swell += weight * (0.5 + 0.5 * np.sin(
                2.0 * np.pi * x * cycles + features.t * speed))
        swell /= sum(w for _, _, w in self.LAYERS)

        # A swell that never quite stills, lifted by the music rather than
        # driven by it.
        level = float(np.clip(self.level.update(features.energy), 0, 1))
        value = np.clip((0.25 + 0.75 * level) * swell, 0.0, 1.0)

        # Deep blue in the troughs, cyan at the crests.
        hue = 0.60 - 0.11 * swell
        out = _hsv_to_rgb(hue, 0.90, value)

        # Whitecaps where the layers coincide, scaled by how loud it is so
        # they stay rare in a quiet scene.
        caps = np.clip((swell - 0.82) / 0.18, 0.0, 1.0) * level
        out = np.clip(out + caps * 0.85, 0.0, 1.0)
        return out * 255.0


class PuddlesEffect(Effect):
    """Onsets drop a blob of colour somewhere on the strip; everything fades.

    Sparse and discrete, where most of the library is a continuous field. Only
    hits put light on the strip, so a sustained passage decays to nothing and a
    drum part reads as separate events rather than as a level.

    Behaviour follows WLED-SR's ``Puddles``; the implementation is our own,
    because WLED is EUPL-1.2 and this package is MIT.
    """

    clone_across_nodes = False

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.pixels = np.zeros((3, width))
        # Seeded so a test sees the same puddles twice. The irregularity that
        # matters visually comes from when onsets land, not from the seed.
        self.rng = np.random.default_rng(0x9E3779B9)

    def render(self, features: Features) -> np.ndarray:
        # Pace the fade off the music. A fixed decay of 0.98 holds a puddle for
        # roughly three seconds whatever is playing, so during a busy passage
        # hits smear into each other and the strip stops reading as separate
        # events -- which is the whole point of this effect. Busy audio now
        # clears the strip in well under a second and a quiet passage lets a
        # puddle linger.
        pace = 0.5 + 4.0 * float(max(features.energy, features.onset_rate))
        self.pixels *= self.settings.effect.scroll_decay ** pace
        if features.beat and not features.silent:
            strength = 0.4 + 0.6 * features.onset
            size = int(np.clip(1 + strength * self.width / 5, 1, self.width))
            start = int(self.rng.integers(0, max(1, self.width - size + 1)))
            # Hue from where the hit sat in the spectrum, so a kick and a hat
            # land in different colours.
            hue = np.full(size, float(np.clip(features.centroid, 0.0, 1.0)) * 0.8)
            blob = _hsv_to_rgb(hue, 1.0, np.full(size, strength)) * 255.0
            seg = self.pixels[:, start:start + size]
            self.pixels[:, start:start + size] = np.maximum(seg, blob)
        return np.clip(self.pixels, 0, 255)


class FreqwaveEffect(Effect):
    """Hue from the dominant frequency, across the whole strip at once.

    Every other effect in the library derives hue from *position*. This one
    derives it from pitch, so a bassline and a cymbal are different colours
    everywhere rather than in different places.

    The first version drew that hue over a travelling sine of fixed speed. It
    looked plain, and measurement said why: with the audio frozen -- the same
    frame fed repeatedly -- it still produced 56% of its normal motion, where
    every other effect produces exactly zero. The only thing the music
    controlled was overall brightness. The shape now comes from the spectrum,
    so the pitch-hue idea survives but nothing moves unless the audio does.
    """

    clone_across_nodes = True

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        # Hue is the whole picture here, so it is smoothed harder than a level
        # would be -- an unsteady centroid would swing the entire strip.
        self.hue = ExpFilter(0.5, alpha_decay=0.05, alpha_rise=0.05)
        self.bands = ExpFilter(np.tile(0.01, settings.dsp.fft_bins),
                               alpha_decay=0.10, alpha_rise=0.65)

    def render(self, features: Features) -> np.ndarray:
        dsp = self.settings.dsp
        # Log frequency: an octave is the same hue distance anywhere on the
        # range, which is how pitch is actually heard.
        lo, hi = max(dsp.min_frequency, 1.0), max(dsp.max_frequency, 2.0)
        pos = (np.log2(max(features.centroid_hz, lo)) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
        hue = float(self.hue.update(float(np.clip(pos, 0.0, 1.0))))

        levels = np.clip(self.bands.update(np.copy(features.mel)), 0.0, 1.5)
        value = np.clip(interpolate(levels, self.width), 0.0, 1.0)
        # A hit lifts the whole strip, since there is no position here for it
        # to land on.
        if features.onset > 0.0:
            value = np.clip(value + 0.35 * features.onset, 0.0, 1.0)
        return _hsv_to_rgb(np.full(self.width, hue * 0.85), 0.95, value) * 255.0


class FireEffect(Effect):
    """Fire, with the draught coming from the music.

    Ported from Mark Kriegsman's ``Fire2012`` in FastLED (MIT): cells cool a
    little each frame, heat drifts along the strip, and sparks are injected at
    the origin. The two knobs the original exposes as constants are what get
    coupled here -- onsets throw sparks, and a loud passage cools more slowly,
    so the flame stands up under energy and gutters when it drops.
    """

    clone_across_nodes = False

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self.heat = np.zeros(width)
        self.rng = np.random.default_rng(0x85EBCA6B)

    def render(self, features: Features) -> np.ndarray:
        n = self.width
        # 1. Cool every cell a little. Less cooling when it is loud.
        cooling = (1.0 - 0.55 * features.energy) * (55.0 * 10.0 / n + 2.0) / 255.0
        self.heat = np.maximum(0.0, self.heat - self.rng.random(n) * cooling)

        # 2. Heat drifts away from the origin and diffuses.
        if n >= 3:
            self.heat[2:] = (self.heat[1:-1] + 2.0 * self.heat[:-2]) / 3.0

        # 3. Sparks at the origin, thrown by onsets rather than at random.
        spark = 0.12 + 0.50 * features.onset + (0.30 if features.beat else 0.0)
        if self.rng.random() < spark:
            at = int(self.rng.integers(0, min(3, n)))
            self.heat[at] = min(1.0, self.heat[at] + 0.6 + 0.4 * self.rng.random())

        # 4. Heat to colour: black -> red -> yellow -> white.
        h = np.clip(self.heat, 0.0, 1.0)
        out = np.array([np.clip(h * 3.0, 0, 1),
                        np.clip(h * 3.0 - 1.0, 0, 1),
                        np.clip(h * 3.0 - 2.0, 0, 1)])
        return out * 255.0


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
    "cinema": CinemaEffect,
    "pacifica": PacificaEffect,
    "puddles": PuddlesEffect,
    "freqwave": FreqwaveEffect,
    "fire": FireEffect,
}


class Director(Effect):
    """Runs the best-suited animation, fading when it changes."""

    clone_across_nodes = False

    def __init__(self, settings: Settings, width: int):
        super().__init__(settings, width)
        self._cache: dict[str, Effect] = {}
        self.allowed = tuple(settings.mood.animations)
        self.current = self.allowed[0]
        self.previous: str | None = None
        self.fade = 1.0
        """1.0 once the current animation is fully faded in."""
        self.last_switch = 0.0
        self.switches = 0
        self.scores: dict[str, float] = {}
        # One filter per candidate. Comparing raw per-frame scores made the
        # lead change roughly twice a second, so dwell and margin were
        # arbitrating noise.
        seconds = max(settings.mood.score_smoothing, 1e-3)
        alpha = min(0.999, 1.0 / (seconds * max(settings.audio.fps, 1)))
        self._smooth = {n: ExpFilter(0.0, alpha_decay=alpha, alpha_rise=alpha)
                        for n in self.allowed}
        self._char = ExpFilter(np.zeros(4), alpha_decay=alpha, alpha_rise=alpha)
        self.anchor: np.ndarray | None = None
        self.drift = 0.0
        """Distance the audio's character has moved since the last switch."""

    def _effect(self, name: str) -> Effect:
        # Built on first use and kept: recreating one per switch would discard
        # its state and reintroduce the seam the fade exists to hide.
        if name not in self._cache:
            self._cache[name] = EFFECTS[name](self.settings, self.width)
        return self._cache[name]

    def _character(self, f: Features) -> np.ndarray:
        """The audio's character as a point: what "the scene changed" measures."""
        raw = np.array([f.energy, f.onset_rate, f.brightness, f.dialogue])
        # Seed on first sight rather than smoothing up from zero. Warming up
        # from zeros meant the anchor was captured mid-warm-up and the drift
        # measured the filter converging, not the audio moving -- the director
        # switched once at startup on any material.
        if self.anchor is None:
            self._char.value = np.copy(raw)
        return self._char.update(raw)

    def choose(self, f: Features) -> str:
        """Switch when the audio changes character; scores only rank what
        comes next.

        The previous rule -- argmax of suitability scores behind margin and
        dwell -- was undependable by construction: all candidates score within
        about 0.2 of each other while any one score moves more than that with
        the material, so every weighting starved some animation and locked the
        winner to the song. Whether to switch and what to switch to are now
        separate questions. A switch happens when the character vector moves
        ``change_threshold`` from where it was at the last switch (or
        ``max_dwell`` expires, the rotation guarantee), and it always goes to
        the best-ranked candidate *other than* the incumbent, so variety is a
        property of the rule rather than of the weights.
        """
        cfg = self.settings.mood
        raw = score_candidates(f, self.allowed)
        if raw:
            self.scores = {n: float(self._smooth[n].update(v)) for n, v in raw.items()}
        vec = self._character(f)
        if self.anchor is None:
            self.anchor = np.copy(vec)
        self.drift = float(np.linalg.norm(vec - self.anchor))
        if len(self.allowed) < 2 or not self.scores:
            return self.current
        if f.t - self.last_switch < cfg.switch_dwell:
            return self.current
        moved = self.drift > cfg.change_threshold
        expired = f.t - self.last_switch > cfg.max_dwell
        if not (moved or expired):
            return self.current
        self.anchor = np.copy(vec)
        others = {n: v for n, v in self.scores.items() if n != self.current}
        return max(others, key=lambda k: others[k])

    def render(self, f: Features) -> np.ndarray:
        cfg = self.settings.mood
        want = self.choose(f)
        if want != self.current:
            self.previous, self.current = self.current, want
            self.fade = 0.0
            self.last_switch = f.t
            self.switches += 1

        out = self._effect(self.current).render(f)
        if self.fade < 1.0 and self.previous is not None:
            # Both keep running during the fade, so neither restarts mid-scene.
            old = self._effect(self.previous).render(f)
            out = old * (1.0 - self.fade) + out * self.fade
            self.fade = min(1.0, self.fade + 1.0 / max(cfg.crossfade * self.settings.audio.fps, 1.0))
            if self.fade >= 1.0:
                self.previous = None
        return out

    @property
    def mood(self):
        """Whatever the running animation exposes, so telemetry keeps working."""
        return getattr(self._effect(self.current), "mood", None)

    def state(self) -> dict:
        return {
            "current": self.current,
            "previous": self.previous,
            "fade": round(self.fade, 3),
            "switches": self.switches,
            "drift": round(self.drift, 3),
            "scores": {k: round(v, 3) for k, v in sorted(self.scores.items())},
        }


EFFECTS["auto"] = Director

#: Which effects react to beats rather than only to level.
BEAT_DRIVEN = frozenset({"pixelwave", "puddles", "fire"})

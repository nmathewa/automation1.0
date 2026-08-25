"""Deciding which animation suits the scene.

Only the *policy* lives here. The :class:`~ambviz.effects.Director` that acts on
it is in :mod:`ambviz.effects`, because a director is itself an effect and
keeping the class here made the two modules import each other. The split is
useful anyway: scoring depends on nothing but :class:`Features`, so it can be
replaced -- by a model, by a learned policy, by anything -- without touching how
switching or fading work.

The wash in :class:`~ambviz.effects.CinemaEffect` is right for a quiet scene and
wrong for a fight. Making the wash itself more energetic was the wrong fix --
it ends up throbbing through dialogue and still not looking like a spectrum when
the music wants one. The library already contains animations built for each of
those; the missing piece was deciding between them.

Three things make that work rather than flap:

* **Hysteresis.** A candidate must beat the incumbent by a margin, so a tie
  keeps what is already on screen.
* **Dwell time.** No switch may happen for several seconds after the last one,
  however strong the case, because a selector that changes its mind every
  second looks far worse than one mediocre animation held steady.
* **Cross-fading.** Effects carry internal state -- scroll histories, filter
  values, peak markers -- so swapping instantly shows a visible seam. Both
  animations render during a fade and are mixed.

The rules below are deliberately simple and hand-written. There is no ground
truth for "the correct animation for this scene", so there is nothing to train
against; this is a starting point to tune by eye, and the scoring is separated
from the switching so it can be replaced without touching the rest.
"""

from __future__ import annotations

import numpy as np

from ambviz.features import Features

#: The wash. Everything falls back here when nothing else makes a case.
DEFAULT = "cinema"


def score_candidates(f: Features) -> dict[str, float]:
    """How well each animation suits this moment, 0-1 each.

    Reads only from :class:`~ambviz.features.Features`, so it stays testable
    without audio and replaceable without touching the director.
    """
    scene = f.scene
    percussive = scene.get("percussion") if scene.available else 0.0
    electronic = scene.get("electronic") if scene.available else 0.0
    driven = scene.get("loud") if scene.available else 0.0
    sustained = max(scene.get("orchestral"), scene.get("acoustic")) if scene.available else 0.0
    voice = scene.get("voice") if scene.available else 0.0

    # Scored from the DSP features first, because those work on any material.
    # The classifier only tips a decision -- leaning on it made every candidate
    # score zero whenever it was quiet or absent, so the wash won by default and
    # nothing ever switched.
    # Singing counts toward calm: a voice is something to sit behind, not
    # something to chase.
    calm = max(f.dialogue, sustained, voice, 1.0 - f.energy)

    return {
        # A spectrum when there is width worth showing.
        "bars": float(np.clip(0.45 * f.energy + 0.35 * f.brightness
                              + 0.20 * max(percussive, electronic), 0, 1)),
        # Beats, when they are the point rather than a texture.
        "pixelwave": float(np.clip(0.65 * f.onset_rate + 0.20 * percussive
                                   + 0.15 * f.energy, 0, 1)),
        # Loud and weighty, without a strong pulse.
        "gravcenter": float(np.clip(0.55 * f.energy * (1.0 - f.onset_rate)
                                    + 0.25 * driven + 0.20 * (1.0 - f.brightness), 0, 1)),
        # Slow harmonic movement. Deliberately not scored on voice: singing
        # used to select this, which meant detecting a vocal made the strip
        # react to it rather than settle down.
        "noisemeter": float(np.clip(0.55 * (1.0 - f.energy) + 0.45 * sustained, 0, 1)),
        # The floor. Deliberately modest: it should win when nothing else makes
        # a case, not outscore candidates that do.
        DEFAULT: float(np.clip(0.55 * calm, 0, 1)),
    }

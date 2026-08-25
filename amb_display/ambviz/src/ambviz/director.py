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

import numpy as np

from ambviz.features import Features

def score_candidates(f: Features, allowed: tuple[str, ...] | None = None) -> dict[str, float]:
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

    # Singing counts toward calm: a voice is something to sit behind, not chase.
    calm = max(f.dialogue, sustained, voice, 1.0 - f.energy)

    # Scored from the DSP features first, because those work on any material.
    # Leaning on the classifier made every candidate score zero whenever it was
    # quiet or absent, and then nothing ever switched.
    scores = {
        # Per-band blocks: wants width worth showing.
        "bars": 0.45 * f.energy + 0.35 * f.brightness
                + 0.20 * max(percussive, electronic),
        # Bars growing from the centre: loud and weighty, without a strong pulse.
        "energy": 0.50 * f.energy * (1.0 - f.onset_rate)
                  + 0.25 * driven + 0.25 * (1.0 - f.brightness),
        # Colour injected at the centre and pushed outward: suits a pulse.
        "scroll": 0.55 * f.onset_rate + 0.25 * percussive + 0.20 * f.energy,
        # Mirrored filterbank, the calmest of the spectral displays.
        "spectrum": 0.45 * calm + 0.35 * f.energy + 0.20 * sustained,
        # Position becomes time: rewards material that changes rather than one
        # that is merely loud.
        "waterfall": 0.45 * f.brightness + 0.30 * (1.0 - f.dialogue)
                     + 0.25 * f.onset_rate,
        # The wash, when it is in the shortlist.
        "cinema": 0.55 * calm,
        # Remaining library members, scored so a custom shortlist still works.
        "gravcenter": 0.55 * f.energy * (1.0 - f.onset_rate) + 0.45 * driven,
        "pixelwave": 0.65 * f.onset_rate + 0.20 * percussive + 0.15 * f.energy,
        "noisemeter": 0.55 * (1.0 - f.energy) + 0.45 * sustained,
        "solid": 0.40 * calm,
    }
    if allowed:
        scores = {k: v for k, v in scores.items() if k in allowed}
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in scores.items()}

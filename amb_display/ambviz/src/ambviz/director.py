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

def score_candidates(f: Features, allowed: tuple[str, ...] | None = None) -> dict[str, float]:
    """How well each animation suits this moment, 0-1 each.

    Reads only from :class:`~ambviz.features.Features`, so it stays testable
    without audio and replaceable without touching the director.
    """
    scene = f.scene
    # Measured, not asserted: YAMNet's ``percussion`` group read 0.000 through a
    # whole run of ordinary music while the strip was plainly reacting to drums.
    # A term that is almost always zero is not a weak signal, it is a constant,
    # and constants cannot rank candidates apart. ``f.percussive`` measures the
    # same property from the spectrum itself, is always available, and needs no
    # model; the classifier group is kept only as an upward vote when it does
    # fire.
    percussive = max(f.percussive, scene.get("percussion") if scene.available else 0.0)
    electronic = scene.get("electronic") if scene.available else 0.0
    # Every classifier term gets a DSP floor, for the reason above. A group
    # that reads zero is not a weak vote, it is a constant, and a constant
    # inside a weighted sum is a permanent handicap: the candidate can never
    # reach the top of a field whose scores sit within about 0.2 of each other.
    #
    # ``loud`` is the clearest case. It covers rock and metal and nothing else,
    # so it reads 0.000 on the overwhelming majority of music -- and it carried
    # 45% of ``gravcenter``, 30% of ``fire`` and 25% of ``energy``. Gravcenter
    # was capped near 0.3 in practice and never once took the strip. Weight
    # without brightness is what those three actually want, and the spectrum
    # can say that on its own.
    driven = max(f.energy * (1.0 - f.brightness),
                 scene.get("loud") if scene.available else 0.0)
    # Sustained is the harmonic half of the HPSS split: strings and pads hold
    # their partials, drums do not. Same failure otherwise -- ``orchestral``
    # and ``acoustic`` are as narrow as ``loud``.
    sustained = max(1.0 - f.percussive,
                    max(scene.get("orchestral"), scene.get("acoustic"))
                    if scene.available else 0.0)
    voice = scene.get("voice") if scene.available else 0.0

    # Singing counts toward calm: a voice is something to sit behind, not chase.
    #
    # ``sustained`` is deliberately *not* in here. With a DSP floor it is high
    # on anything harmonic, including a loud wall of synth, and folding that
    # into calm would send the wash to the top of a scene that wants a
    # spectrum. Calm means "little is happening", which is what the other three
    # measure; being sustained is a separate question and is scored separately.
    calm = max(f.dialogue, voice, 1.0 - f.energy)

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
        # The wash. Sustained material suits it for the same reason calm does:
        # there is nothing to read, so there should be nothing asking to be
        # read. Without this term the DSP floor on ``sustained`` handed
        # ``spectrum`` a free 0.10 on exactly the quiet scenes the wash exists
        # for, and the wash lost them.
        "cinema": 0.55 * calm + 0.20 * sustained,
        # Sparse hits: the only candidate that goes dark between onsets, so it
        # needs a real rhythm rather than just energy.
        "puddles": 0.60 * f.onset_rate + 0.25 * percussive + 0.15 * f.energy,
        # Hue from pitch rather than position: earns its place when the
        # spectrum moves, which is what brightness tracks.
        #
        # Scored from DSP only. With 0.30 of its weight on a classifier term
        # this sat permanently a third short whenever YAMNet was quiet or
        # absent, and took the strip for 0% of 43 s -- the trap this module's
        # own docstring warns about, walked straight into.
        "freqwave": 0.55 * f.brightness + 0.45 * f.energy,
        # Warm and dark, so it suits weight at the bottom of the spectrum.
        "fire": 0.45 * f.energy + 0.30 * driven + 0.25 * (1.0 - f.brightness),
        # Remaining library members, scored so a custom shortlist still works.
        "gravcenter": 0.55 * f.energy * (1.0 - f.onset_rate) + 0.45 * driven,
        "pixelwave": 0.65 * f.onset_rate + 0.20 * percussive + 0.15 * f.energy,
        "noisemeter": 0.55 * (1.0 - f.energy) + 0.45 * sustained,
        "solid": 0.40 * calm,
    }
    if allowed:
        scores = {k: v for k, v in scores.items() if k in allowed}
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in scores.items()}

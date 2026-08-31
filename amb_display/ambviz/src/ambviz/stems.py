"""Instrument balance from Demucs, on a background thread.

:mod:`ambviz.scene` asks *what kind of sound is this*. This asks *what is it
made of* -- how much of the music is drums, bass, vocals or everything else --
by actually separating the mix rather than classifying it.

Why this feeds the slow layer only
----------------------------------

The tempting design is to publish a spectral mask and apply it to the live
frame, which costs 5 us and would give stem envelopes at no added latency.
**Measured on 33 s of real music, it does not work.** A mask taken from the
current frame recovers per-stem energy almost perfectly (6.1% error, 0.993
correlation), so masking is a fine representation -- but the routing decays
almost immediately:

===========  ==========  ============
mask lag     error       mean corr
===========  ==========  ============
0 ms         6.1%        0.993
17 ms        17.9%       0.873
50 ms        43.1%       0.465
1000 ms      51.5%       0.327
===========  ==========  ============

``|mask(t) - mask(t-0.25s)|`` is 0.148 against 0.178 at two seconds: the
routing is already decorrelated by a quarter of a second, so it is not a slow
signal that can be computed late and applied now. Anything needing frame
timing must keep using onsets and :class:`~ambviz.dsp.HarmonicPercussive`.

What *does* survive the lag is the balance over a passage, which is what the
director consumes -- it smooths over seconds and switches every 8-90 s. With a
1 s window at 1 s lag and 2 s of smoothing, correlation against ground truth is
**drums 0.98, vocals 0.96, other 0.79, bass 0.65**.

Bass is the weak stem throughout and is published but should not be leaned on:
the kick and the bassline share the bottom of the spectrum, and no per-bin
scalar can separate them.

Cost on an RTX 4050 (laptop, 6 GB): **19.3 ms** for a one-second window at
403 MiB, about 4% duty cycle at the default interval. Entirely optional --
without ``torch`` everything here reports nothing and the visualizer is
unaffected, exactly as with the classifier.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

MODEL_RATE = 44100
"""Demucs is trained at 44.1 kHz stereo."""

SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")

#: Where each stem sits on the colour circle, in turns. Taken from the
#: dashboard's own stem colours so a wall lit by the guitar is the same green
#: as the guitar's segment in the balance bar -- one colour language, not two.
HUES: dict[str, float] = {
    "drums": 0.07,     # amber
    "bass": 0.61,      # blue
    "other": 0.42,     # green
    "vocals": 0.90,    # magenta
    "guitar": 0.50,    # cyan
    "piano": 0.75,     # violet
}

#: An even split across four stems. Shares are reported raw, so consumers that
#: want "is this drum-dominated" should compare against this rather than 0.5.
EVEN_SHARE = 1.0 / len(SOURCES)


@dataclass
class Stems:
    """Share of the music's energy belonging to each stem, summing to 1."""

    shares: dict[str, float] = field(default_factory=dict)
    available: bool = False
    device: str = ""
    inference_ms: float = 0.0
    """How long the last separation took, for the dashboard."""

    change: float = 0.0
    """How far the instrument balance just moved, 0-1.

    An arrangement event -- drums dropping out, a guitar entering, a chorus
    arriving with everything at once. Slower and far more meaningful than a
    spectral onset, which fires on every hi-hat, and it is the thing a listener
    would actually call "the music changed"."""

    def get(self, name: str) -> float:
        return float(self.shares.get(name, 0.0))

    def prominence(self, name: str) -> float:
        """Share rescaled so an even four-way split reads 0.5, clipped to 0-1.

        Raw shares sit around 0.25 by construction, which would make every
        stem look weak inside a weighted sum built for 0-1 features.
        """
        return float(np.clip(self.get(name) / (2.0 * EVEN_SHARE), 0.0, 1.0))

    def ranked(self) -> list[tuple[str, float]]:
        """Stems by share, loudest first."""
        return sorted(self.shares.items(), key=lambda kv: -kv[1])

    def secondary(self, exclude: tuple[str, ...] = ("bass",)) -> tuple[str, float]:
        """The second most prominent stem, and its share.

        The *second* rather than the first on purpose: the loudest source is
        already what the front wall is mostly showing, so putting it on the
        sides too says nothing new. The runner-up is the thing the mix has that
        the front is not telling you about.

        ``bass`` is excluded by default because it is the one stem the model is
        unreliable about -- 0.63 correlation against ground truth, since the
        kick and the bassline share the bottom of the spectrum.
        """
        rank = [(n, v) for n, v in self.ranked() if n not in exclude]
        if len(rank) < 2:
            return ("", 0.0)
        return rank[1]

    def to_dict(self) -> dict:
        second, share = self.secondary()
        return {
            "available": self.available,
            "device": self.device,
            "change": round(self.change, 3),
            "secondary": second,
            "secondary_share": round(share, 3),
            "inference_ms": round(self.inference_ms, 1),
            **{k: round(v, 3) for k, v in self.shares.items()},
        }


class StemSeparator:
    """Runs Demucs on a rolling window on its own thread.

    :meth:`push` is called from the audio thread and only copies into a ring
    buffer, so a slow separation can never stall rendering -- the same contract
    :class:`~ambviz.scene.SceneClassifier` works under.
    """

    def __init__(self, rate: int, window: float = 1.0, interval: float = 0.5,
                 device: str | None = None):
        import torch                                    # optional dependency
        import torchaudio

        self.torch = torch
        self.rate = rate
        self.interval = interval
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Six sources where they are available. `guitar` is the reason: it
        # correlates 0.89 against ground truth and carried 24% of the energy on
        # real material, which makes it the most useful thing the four-source
        # model was folding invisibly into `other`.
        #
        # htdemucs_6s is a bag of models and refuses a direct call, so it goes
        # through demucs' own `apply_model`. Without the demucs package the
        # four-source bundle inside torchaudio still works.
        self._apply = None
        try:
            from demucs.apply import apply_model
            from demucs.pretrained import get_model

            self.model = get_model("htdemucs_6s").to(device).eval()
            self.model_rate = self.model.samplerate
            self._apply = apply_model
        except Exception:
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            self.model = bundle.get_model().to(device).eval()
            self.model_rate = bundle.sample_rate
        self.sources = tuple(self.model.sources)

        self.window = int(window * self.model_rate)
        self._prev_shares: np.ndarray | None = None
        self._buf = np.zeros((2, self.window), dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.stems = Stems(available=True, device=device)
        self._thread: threading.Thread | None = None

    # ── audio thread ─────────────────────────────────────────────────────────
    def push(self, samples: np.ndarray, rate: int) -> None:
        """Add audio. Cheap by design; called once per frame.

        Takes stereo where it can get it: Demucs is trained on stereo and its
        separation leans on where things sit in the image, so folding to mono
        first would throw away part of what makes it work.
        """
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x])
        elif x.shape[0] != 2:
            x = x.T
        if x.shape[0] != 2:
            return
        if rate != self.model_rate:
            from scipy.signal import resample_poly

            g = np.gcd(self.model_rate, rate)
            x = resample_poly(x, self.model_rate // g, rate // g, axis=1)
        n = min(x.shape[1], self.window)
        if n <= 0:
            return
        with self._lock:
            self._buf = np.concatenate([self._buf[:, n:], x[:, -n:]], axis=1)

    # ── separator thread ─────────────────────────────────────────────────────
    def start(self) -> "StemSeparator":
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ambviz-stems")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import time

        torch = self.torch
        while not self._stop.wait(self.interval):
            with self._lock:
                window = np.copy(self._buf)
            if not np.any(window):
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                x = torch.from_numpy(window).unsqueeze(0).to(self.device)
                out = (self._apply(self.model, x, split=False, overlap=0,
                                   progress=False)[0]
                       if self._apply is not None else self.model(x)[0])
                # Energy per stem, summed over time and channels. The ratio is
                # what is wanted, so no normalisation by window length.
                energy = out.pow(2).sum(dim=(1, 2)).sqrt().cpu().numpy()
            total = float(energy.sum())
            if total <= 1e-9:
                continue
            shares = energy / total
            # Half the L1 distance, so a complete swap of the balance reads 1
            # and a steady passage reads near 0 however loud it is.
            change = 0.0
            if self._prev_shares is not None:
                change = float(np.clip(
                    np.abs(shares - self._prev_shares).sum() / 2.0, 0.0, 1.0))
            self._prev_shares = shares
            self.stems = Stems(
                change=change,
                shares={n: float(v) for n, v in zip(self.sources, shares)},
                available=True,
                device=self.device,
                inference_ms=(time.perf_counter() - t0) * 1000.0,
            )


def try_create(rate: int, window: float = 1.0, interval: float = 0.5,
               device: str | None = None) -> StemSeparator | None:
    """Build a separator, or return None if it cannot run.

    Missing torch, no GPU, no weights cached and no network -- all are fine.
    The visualizer works without it; this only ever adds information.
    """
    try:
        return StemSeparator(rate, window=window, interval=interval,
                             device=device).start()
    except Exception:
        return None

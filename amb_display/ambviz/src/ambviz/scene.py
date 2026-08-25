"""Semantic scene labels from YAMNet, on a background thread.

The hand-rolled features in :mod:`ambviz.pipeline` describe how audio *behaves*
-- how loud, how wide, how centred. That is enough to separate a conversation
from an explosion most of the time, but it is a proxy, and proxies can be fooled:
a tone with tremolo is centred and narrow, so those features call it dialogue,
while YAMNet correctly calls it a synthesizer.

This adds what the audio *is*. YAMNet is MobileNet-v1 over AudioSet's 521
classes, costs under a millisecond per inference, and runs here at 2 Hz -- about
0.2% of one core. It is entirely optional: without ``ai-edge-litert`` installed
everything below reports nothing and the visualizer is unaffected.

It runs on its own thread because it works on ~1 s windows while the pipeline
works on 17 ms frames. That is not a mismatch to paper over -- a scene label
*should* move slowly, and it feeds the slow layer, never the fast one.
"""

from __future__ import annotations

import threading
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_URL = ("https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1"
             "?lite-format=tflite")
DEFAULT_MODEL = Path.home() / ".cache" / "ambviz-models" / "yamnet.tflite"

MODEL_RATE = 16000
"""YAMNet is fixed at 16 kHz mono."""

# AudioSet has 521 classes, most of which a light strip cannot use -- birds,
# traffic, doors. What matters here is *musical* content: what is playing and
# what it is made of, since that is what the strip is reacting to.
#
# Listed by exact name rather than pattern: "Speech synthesizer" is a keyboard
# however its name reads, and "Bass drum" belongs to percussion rather than to
# bass instruments. "Double bass" is deliberately in two groups -- it really is
# both -- which is harmless because groups score by max, not by sum.
GROUPS: dict[str, tuple[str, ...]] = {
    "music": (
        "Music", "Musical instrument", "Background music", "Theme music",
        "Soundtrack music", "Song",
    ),
    "percussion": (
        "Percussion", "Drum kit", "Drum machine", "Drum", "Snare drum",
        "Bass drum", "Timpani", "Tabla", "Cymbal", "Hi-hat", "Drum roll",
        "Tambourine", "Rattle (instrument)", "Wood block", "Marimba, xylophone",
        "Glockenspiel", "Vibraphone", "Steelpan", "Maraca", "Gong",
    ),
    "bass": ("Bass guitar", "Double bass"),
    "guitar": (
        "Guitar", "Electric guitar", "Acoustic guitar", "Plucked string instrument",
        "Steel guitar, slide guitar", "Banjo", "Mandolin", "Ukulele", "Sitar",
        "Strum", "Tapping (guitar technique)",
    ),
    "keys": (
        "Keyboard (musical)", "Piano", "Electric piano", "Organ",
        "Electronic organ", "Hammond organ", "Synthesizer", "Sampler",
        "Harpsichord", "Speech synthesizer",
    ),
    "orchestral": (
        "Orchestra", "Classical music", "String section", "Violin, fiddle",
        "Cello", "Double bass", "Bowed string instrument", "Harp",
        "Brass instrument", "Trumpet", "Trombone", "French horn",
        "Wind instrument, woodwind instrument", "Flute", "Clarinet",
        "Saxophone",
    ),
    "voice": (
        "Singing", "Choir", "Vocal music", "A capella", "Chant", "Mantra",
        "Child singing", "Synthetic singing",
        "Rapping", "Humming", "Yodeling", "Whistling",
    ),
    "electronic": (
        "Electronic music", "House music", "Techno", "Dubstep",
        "Drum and bass", "Electronica", "Electronic dance music",
        "Trance music", "Ambient music",
    ),
    "acoustic": (
        "Folk music", "Jazz", "Blues", "Country", "Bluegrass",
        "Rhythm and blues", "Soul music", "Reggae", "Swing music",
    ),
    "loud": (
        "Rock music", "Heavy metal", "Punk rock", "Grunge",
        "Progressive rock", "Rock and roll", "Psychedelic rock",
    ),
}

@dataclass
class Scene:
    """Grouped scores, 0-1. All zero when no classifier is running."""

    scores: dict[str, float] = field(default_factory=dict)
    top: str = ""
    """Highest-scoring raw AudioSet class, for diagnostics."""
    top_score: float = 0.0
    available: bool = False

    novelty: float = 0.0
    """How far the classification just moved, 0-1.

    The model changing its mind sharply is itself informative, regardless of
    what it changed its mind to."""

    unusual: float = 0.0
    """Confidence in something outside the musical vocabulary, 0-1.

    This is the useful half of a wrong answer. A production effect that YAMNet
    labels "Sonar" is not sonar -- but it fired confidently on a class that has
    no business in a piece of music, and that is precisely the surprise element
    worth reacting to. The literal label is wrong; the *event* is real."""

    def get(self, group: str) -> float:
        return float(self.scores.get(group, 0.0))

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "top": self.top,
            "top_score": round(self.top_score, 3),
            "novelty": round(self.novelty, 3),
            "unusual": round(self.unusual, 3),
            **{k: round(v, 3) for k, v in self.scores.items()},
        }


def ensure_model(path: Path | None = None) -> Path:
    """Return the model path, downloading it once if needed."""
    path = Path(path) if path else DEFAULT_MODEL
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=120) as r:
            path.write_bytes(r.read())
    return path


class SceneClassifier:
    """Runs YAMNet on a background thread over a rolling window.

    :meth:`push` is called from the audio thread and must stay cheap -- it only
    copies samples into a ring buffer. Inference happens elsewhere, so a slow
    prediction can never stall rendering.
    """

    def __init__(self, rate: int, interval: float = 0.5, model: Path | None = None):
        from ai_edge_litert.interpreter import Interpreter   # optional dependency

        self.rate = rate
        self.interval = interval
        path = ensure_model(model)
        self.labels = zipfile.ZipFile(path).read("yamnet_label_list.txt").decode().splitlines()
        self._index = {name: i for i, name in enumerate(self.labels)}
        self._groups = {
            group: np.array([self._index[n] for n in names if n in self._index], dtype=int)
            for group, names in GROUPS.items()
        }

        self._interp = Interpreter(model_path=str(path))
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self.window = int(self._in["shape"][0])

        # Everything not in a musical group: firing here means something
        # unexpected is happening.
        musical = {i for idx in self._groups.values() for i in idx}
        self._other = np.array(sorted(set(range(len(self.labels))) - musical), dtype=int)
        self._prev_scores: np.ndarray | None = None

        self._buf = np.zeros(self.window, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.scene = Scene(available=True)
        self._thread: threading.Thread | None = None

    # ── audio thread ─────────────────────────────────────────────────────────
    def push(self, mono: np.ndarray, rate: int) -> None:
        """Add audio. Cheap by design; called once per frame."""
        if rate != MODEL_RATE:
            from scipy.signal import resample_poly

            # 44100 -> 16000 is 160/441; resample_poly keeps that exact.
            g = np.gcd(MODEL_RATE, rate)
            mono = resample_poly(mono, MODEL_RATE // g, rate // g)
        if not len(mono):
            return
        with self._lock:
            n = min(len(mono), self.window)
            self._buf = np.concatenate([self._buf[n:], mono[-n:].astype(np.float32)])

    # ── classifier thread ────────────────────────────────────────────────────
    def start(self) -> "SceneClassifier":
        self._thread = threading.Thread(target=self._run, daemon=True, name="ambviz-scene")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                window = np.copy(self._buf)
            if not np.any(window):
                continue
            self._interp.set_tensor(self._in["index"], window)
            self._interp.invoke()
            scores = self._interp.get_tensor(self._out["index"])[0]
            best = int(np.argmax(scores))

            # Cosine distance from the previous prediction. A steady passage
            # scores near zero however loud it is; a sudden change of character
            # spikes, which is what a surprise actually is.
            novelty = 0.0
            if self._prev_scores is not None:
                a, b = scores, self._prev_scores
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denom > 1e-9:
                    novelty = float(np.clip(1.0 - np.dot(a, b) / denom, 0.0, 1.0))
            self._prev_scores = scores.copy()

            # Max rather than sum across a group: five quiet music classes should
            # not out-vote one confident hit.
            self.scene = Scene(
                scores={g: float(np.max(scores[idx])) if len(idx) else 0.0
                        for g, idx in self._groups.items()},
                top=self.labels[best],
                top_score=float(scores[best]),
                novelty=novelty,
                unusual=float(np.max(scores[self._other])) if len(self._other) else 0.0,
                available=True,
            )


def try_create(rate: int, interval: float = 0.5,
               model: Path | None = None) -> SceneClassifier | None:
    """Build a classifier, or return None if it cannot run.

    Missing runtime, missing model, no network -- all are fine. The visualizer
    works without it; this only ever adds information.
    """
    try:
        return SceneClassifier(rate, interval=interval, model=model).start()
    except Exception:
        return None

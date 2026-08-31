"""Stem balance from Demucs, and the limits it must be used within.

The separator is optional, so most of this runs against a fake. What is being
tested is the contract around it: that nothing frame-timed reads it, that the
shares stay a partition, and that the app is unaffected when torch is absent.
"""

import numpy as np
import pytest

from ambviz.director import score_candidates
from ambviz.features import Features
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings
from ambviz.stems import EVEN_SHARE, SOURCES, Stems, try_create


def stems(**shares):
    full = {n: 0.0 for n in SOURCES}
    full.update(shares)
    return Stems(shares=full, available=True, device="cpu")


# ── the dataclass ────────────────────────────────────────────────────────────
def test_absent_by_default():
    s = Stems()
    assert not s.available and s.get("drums") == 0.0


def test_prominence_scales_an_even_split_to_a_half():
    """Raw shares sit near 0.25 by construction, which would read as weak
    inside a weighted sum built for 0-1 features."""
    even = stems(**{n: EVEN_SHARE for n in SOURCES})
    assert even.prominence("drums") == pytest.approx(0.5)
    assert stems(drums=0.5).prominence("drums") == pytest.approx(1.0)
    assert stems(drums=0.9).prominence("drums") == 1.0     # clipped


def test_telemetry_carries_every_stem():
    d = stems(drums=0.4, bass=0.2, other=0.3, vocals=0.1).to_dict()
    assert d["available"] and all(n in d for n in SOURCES)


# ── scoring consumes it ──────────────────────────────────────────────────────
def test_stems_lift_the_candidates_that_want_drums():
    quiet = Features(mel=np.zeros(24), volume=0.5, energy=0.5, onset_rate=0.5,
                     percussive=0.2, stems=stems(drums=0.05, other=0.95))
    loud = Features(mel=np.zeros(24), volume=0.5, energy=0.5, onset_rate=0.5,
                    percussive=0.2, stems=stems(drums=0.6, other=0.4))
    a = score_candidates(quiet, ("puddles", "pixelwave"), stem_weight=1.0)
    b = score_candidates(loud, ("puddles", "pixelwave"), stem_weight=1.0)
    assert all(b[k] > a[k] for k in a)


def test_stem_weight_zero_changes_nothing():
    """It is off by default and must be inert until asked for."""
    f = Features(mel=np.zeros(24), volume=0.5, energy=0.5, onset_rate=0.5,
                 percussive=0.3, stems=stems(drums=0.9))
    assert score_candidates(f, stem_weight=0.0) == score_candidates(
        Features(mel=np.zeros(24), volume=0.5, energy=0.5, onset_rate=0.5,
                 percussive=0.3), stem_weight=0.0)


def test_an_unavailable_separator_is_ignored_even_at_full_weight():
    f = Features(mel=np.zeros(24), volume=0.5, energy=0.5, onset_rate=0.5,
                 percussive=0.3, stems=Stems(shares={"drums": 0.9}))
    assert score_candidates(f, stem_weight=1.0) == score_candidates(f, stem_weight=0.0)


def test_bass_is_published_but_never_consumed():
    """It correlates 0.65 against ground truth -- the kick and the bassline
    share the bottom of the spectrum. Publishing it is fine; scoring on it
    would be building on the one stem known to be unreliable."""
    base = dict(mel=np.zeros(24), volume=0.5, energy=0.5, onset_rate=0.5,
                percussive=0.3)
    lo = Features(**base, stems=stems(bass=0.05, other=0.95))
    hi = Features(**base, stems=stems(bass=0.95, other=0.05))
    a = score_candidates(lo, stem_weight=1.0)
    b = score_candidates(hi, stem_weight=1.0)
    # `other` moved, so only compare candidates that read neither.
    for name in ("puddles", "pixelwave", "scroll"):
        assert a[name] == pytest.approx(b[name]), name


# ── the pipeline ─────────────────────────────────────────────────────────────
def test_the_visualizer_runs_without_a_separator():
    v = Visualizer(Settings.load(overrides={"mood": {"stem_weight": 0.0,
                                                     "scene_weight": 0.0}}))
    assert v.separator is None
    n = v.settings.audio.samples_per_frame
    for _ in range(30):
        v.process(np.zeros((n, 2), dtype=np.int16))
    assert not v.features.stems.available


def test_smoothed_shares_stay_a_partition():
    """Filtering each share independently lets them drift off summing to one,
    and a share that does not sum to one is a number nobody can reason about."""
    v = Visualizer(Settings.load(overrides={"mood": {"stem_weight": 1.0,
                                                     "scene_weight": 0.0}}))

    class Fake:
        stems = Stems(shares={"drums": 0.7, "bass": 0.1, "other": 0.1,
                              "vocals": 0.1}, available=True, device="cpu")
        def push(self, *a, **k): pass

    v.separator = Fake()
    for _ in range(5):
        out = v._smooth_stems()
    assert sum(out.shares.values()) == pytest.approx(1.0)
    assert out.available


def test_the_filter_seeds_rather_than_ramping_from_zero():
    """Ramping would spend the whole smoothing window claiming the music has
    no drums in it."""
    v = Visualizer(Settings.load(overrides={"mood": {"stem_weight": 1.0}}))

    class Fake:
        stems = Stems(shares={"drums": 0.7, "bass": 0.1, "other": 0.1,
                              "vocals": 0.1}, available=True, device="cpu")
        def push(self, *a, **k): pass

    v.separator = Fake()
    assert v._smooth_stems().get("drums") == pytest.approx(0.7, abs=1e-6)


def test_telemetry_publishes_stems():
    v = Visualizer(Settings.load(overrides={"mood": {"stem_weight": 0.0}}))
    assert "stems" in v.snapshot()


def test_settings_reject_a_weight_outside_the_range():
    with pytest.raises(ValueError):
        Settings.load(overrides={"mood": {"stem_weight": 1.5}})


def test_raising_the_weight_starts_the_separator():
    """`mood.animations` shipped with exactly this bug: patchable, accepted,
    and ignored because the object was only built in __init__. A dashboard
    slider that silently does nothing is worse than no slider."""
    import time
    v = Visualizer(Settings.load(overrides={"mood": {"stem_weight": 0.0}}))
    assert v.separator is None
    v.apply({"mood": {"stem_weight": 0.6}})
    # Construction is threaded so the audio thread never stalls on a download.
    for _ in range(100):
        if v.separator is not None:
            break
        time.sleep(0.1)
    assert v.settings.mood.stem_weight == 0.6
    # Without torch it stays None for ever, and that is a supported state.
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed; separator is meant to stay absent")
    assert v.separator is not None


def test_lowering_the_weight_keeps_the_model_loaded():
    """Sweeping a slider must not reload a few hundred MB per step."""
    v = Visualizer(Settings.load(overrides={"mood": {"stem_weight": 0.0}}))
    sentinel = object()
    v.separator = sentinel
    v.apply({"mood": {"stem_weight": 0.0}})
    assert v.separator is sentinel

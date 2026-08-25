"""The optional scene classifier, and the parts that work without it."""

import numpy as np
import pytest

from ambviz.scene import GROUPS, Scene, try_create
from ambviz.pipeline import Visualizer
from ambviz.settings import Settings

litert = pytest.importorskip("ai_edge_litert", reason="classifier is optional")


def test_absent_scene_reports_nothing():
    s = Scene()
    assert not s.available
    assert s.get("percussion") == 0.0
    assert s.get("anything at all") == 0.0


def test_groups_use_real_labels():
    """A typo in a group name silently makes it dead weight."""
    classifier = try_create(44100)
    if classifier is None:
        pytest.skip("no model available")
    known = set(classifier.labels)
    for group, names in GROUPS.items():
        matched = [n for n in names if n in known]
        assert len(matched) >= len(names) * 0.8, \
            f"{group}: only {len(matched)}/{len(names)} names exist in AudioSet"
    classifier.stop()


def test_groups_are_musical():
    """The taxonomy is deliberately about music, not about the environment."""
    assert "percussion" in GROUPS and "orchestral" in GROUPS
    flat = {n for names in GROUPS.values() for n in names}
    for unwanted in ("Bird", "Traffic noise, roadway noise", "Speech", "Explosion"):
        assert unwanted not in flat


def test_disabling_it_costs_nothing():
    v = Visualizer(Settings.load(overrides={"mood": {"scene_weight": 0.0}}))
    assert v.classifier is None
    out = v.process(np.zeros((v.samples_per_frame, 2)))
    assert out.shape == (3, v.settings.output.pixels)
    assert not v.features.scene.available


def test_pipeline_runs_with_the_classifier_attached():
    v = Visualizer(Settings.load(overrides={"mood": {"scene_weight": 0.7}}))
    if v.classifier is None:
        pytest.skip("no model available")
    rng = np.random.default_rng(0)
    for _ in range(30):
        v.process(rng.normal(0, 3000, (v.samples_per_frame, 2)))
    assert "scene" in v.snapshot()


def test_pushing_audio_never_blocks_on_inference():
    """push() runs on the audio thread, so it must only copy."""
    import time

    c = try_create(44100)
    if c is None:
        pytest.skip("no model available")
    try:
        block = np.zeros(735, dtype=np.float32)
        start = time.perf_counter()
        for _ in range(200):
            c.push(block, 44100)
        per_call_ms = (time.perf_counter() - start) / 200 * 1000
        assert per_call_ms < 2.0, f"push cost {per_call_ms:.2f} ms per frame"
    finally:
        c.stop()


def test_novelty_is_zero_on_the_first_prediction_and_bounded_after():
    c = try_create(44100)
    if c is None:
        pytest.skip("no model available")
    try:
        assert c.scene.novelty == 0.0
        rng = np.random.default_rng(0)
        for _ in range(60):
            c.push(rng.normal(0, 0.2, 735).astype(np.float32), 44100)
        import time
        time.sleep(1.5)
        assert 0.0 <= c.scene.novelty <= 1.0
        assert 0.0 <= c.scene.unusual <= 1.0
    finally:
        c.stop()


def test_unusual_covers_everything_outside_the_musical_groups():
    """The useful half of a wrong answer.

    A production effect labelled "Sonar" is not sonar, but it is a confident
    hit on a class with no business in music -- which is exactly the surprise
    worth reacting to.
    """
    c = try_create(44100)
    if c is None:
        pytest.skip("no model available")
    try:
        musical = {i for idx in c._groups.values() for i in idx}
        assert len(c._other) + len(musical) == len(c.labels)
        assert c._index["Sonar"] in set(c._other.tolist())
    finally:
        c.stop()

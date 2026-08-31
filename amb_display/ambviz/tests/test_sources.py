"""Mic capture, and the stereo it must not throw away.

The bindings are stubbed rather than driven: PortAudio needs a device, CI has
none, and the thing under test is the channel decision, not PortAudio.
"""

import numpy as np
import pytest

from ambviz import sources
from ambviz.settings import Settings


class _FakeStream:
    """Just enough of a sounddevice InputStream to read frames from."""

    def __init__(self, channels: int, frame_size: int):
        self.channels = channels
        self.frame_size = frame_size
        self.started = False
        self.read_available = 0

    def start(self) -> None:
        self.started = True

    def read(self, n: int):
        return np.zeros((n, self.channels), dtype=np.int16), False

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeSD:
    """A sounddevice stand-in for one device of a given width.

    ``refuses`` models the case the fallback exists for: a device that
    advertises two inputs and then will not open them.
    """

    def __init__(self, reported: int, refuses: bool = False):
        self.reported = reported
        self.refuses = refuses
        self.attempts: list[int] = []

    def query_devices(self, device=None, kind=None):
        return {"max_input_channels": self.reported, "name": "fake"}

    def InputStream(self, *, channels, blocksize, **kw):  # noqa: N802 - sd's name
        self.attempts.append(channels)
        if self.refuses and channels > 1:
            raise RuntimeError("device cannot open 2 channels at this rate")
        return _FakeStream(channels, blocksize)


def _mic(monkeypatch, sd) -> sources._SoundDeviceMic:
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", sd)
    monkeypatch.setattr(sources, "resolve_input_device", lambda d: None)
    return sources._SoundDeviceMic(Settings())


def test_stereo_device_is_captured_in_stereo(monkeypatch):
    """The whole point: a virtual cable carrying playback arrives as stereo."""
    mic = _mic(monkeypatch, _FakeSD(reported=2))
    assert mic.channels == 2
    frame = next(mic.frames())
    assert frame.ndim == 2 and frame.shape[1] == 2, \
        "downmixing here would cost vocal suppression, the stereo image and the walls"


def test_mono_device_still_yields_flat_frames(monkeypatch):
    """A real microphone has one input and must not be reshaped into two."""
    mic = _mic(monkeypatch, _FakeSD(reported=1))
    assert mic.channels == 1
    assert next(mic.frames()).ndim == 1


def test_falls_back_to_mono_rather_than_failing_to_start(monkeypatch):
    """A device may advertise inputs it will not open. Mono beats not starting."""
    sd = _FakeSD(reported=2, refuses=True)
    mic = _mic(monkeypatch, sd)
    assert sd.attempts == [2, 1], "stereo must be tried first, and only once"
    assert mic.channels == 1
    assert next(mic.frames()).ndim == 1


def test_a_device_that_will_not_say_is_treated_as_mono(monkeypatch):
    """Probing is best-effort: an exception is not a reason to refuse to run."""
    class _Silent(_FakeSD):
        def query_devices(self, device=None, kind=None):
            raise RuntimeError("no such device")

    assert _mic(monkeypatch, _Silent(reported=2)).channels == 1


def test_a_real_failure_is_not_swallowed(monkeypatch):
    """Falling back to mono must not turn a broken device into silence."""
    class _Dead(_FakeSD):
        def InputStream(self, **kw):  # noqa: N802 - sd's name
            raise RuntimeError("device is gone")

    with pytest.raises(RuntimeError, match="device is gone"):
        _mic(monkeypatch, _Dead(reported=1))


@pytest.mark.parametrize("reported,expected", [(0, (1,)), (1, (1,)), (2, (2, 1)), (8, (2, 1))])
def test_never_asks_for_more_than_two(reported, expected):
    """An 8-in interface is not eight walls; the pipeline reads mid and side."""
    assert sources._wanted_channels(reported) == expected


def test_stereo_frames_reach_the_pipeline_as_stereo(monkeypatch):
    """The shape the mic yields is the shape `process()` branches on."""
    from ambviz.pipeline import Visualizer

    mic = _mic(monkeypatch, _FakeSD(reported=2))
    settings = Settings()
    viz = Visualizer(settings)
    frame = next(mic.frames())
    # Silence returns early, so give it something to hear.
    frame = frame + np.array([2000.0, -2000.0], dtype=np.float32)
    viz.process(frame)
    assert viz.stereo is True

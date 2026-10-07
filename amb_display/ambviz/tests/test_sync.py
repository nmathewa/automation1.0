"""Per-device light delay: resolving it, and holding frames back by it."""

import tomllib

import numpy as np
import pytest

from ambviz.control import CommandQueue
from ambviz.outputs import FrameDelay
from ambviz.settings import Settings
from ambviz.sync import resolve

BT = "bluez_output.B8_84_11_62_FE_51.1"
USB = "alsa_output.usb-Generic_USB_audio_FRONT-00.HiFi__hw_FRONT__sink"


def settings(**output):
    return Settings.load(overrides={"output": output} if output else None)


def test_a_wired_output_gets_no_delay():
    sync = resolve(settings(), USB)
    assert (sync.delay_ms, sync.basis) == (0.0, "none")


def test_bluetooth_starts_from_the_typical_figure():
    """The sound server reports 0 latency for Bluetooth, so it cannot be read."""
    sync = resolve(settings(), BT)
    assert sync.delay_ms == Settings().output.bluetooth_delay_ms
    assert sync.basis == "bluetooth-default"


def test_a_calibrated_entry_beats_the_default():
    sync = resolve(settings(device_delays={BT: 165}), BT)
    assert (sync.delay_ms, sync.basis) == (165.0, "configured")


def test_entries_can_be_patterns():
    s = settings(device_delays={"bluez_output.*": 140, "alsa_output.usb-*": 12})
    assert resolve(s, BT).delay_ms == 140.0
    assert resolve(s, USB).delay_ms == 12.0


def test_an_exact_name_beats_a_pattern():
    s = settings(device_delays={"bluez_output.*": 140, BT: 190})
    assert resolve(s, BT).delay_ms == 190.0


def test_the_rig_delay_adds_to_every_device():
    s = settings(delay_ms=15, device_delays={BT: 165})
    assert resolve(s, BT).delay_ms == 180.0
    assert resolve(s, USB).delay_ms == 15.0
    assert resolve(s, "").delay_ms == 15.0


def test_a_generated_source_has_no_device_part():
    assert resolve(settings(), "").basis == "none"


@pytest.mark.parametrize("bad", [
    {"delay_ms": -1}, {"delay_ms": 5000}, {"bluetooth_delay_ms": -5},
    {"device_delays": {BT: -10}}, {"device_delays": {BT: "slow"}},
    {"device_delays": {"": 10}}, {"device_delays": {BT: True}},
])
def test_nonsense_is_rejected(bad):
    with pytest.raises(ValueError, match="delay"):
        settings(**bad)


def test_per_device_delays_survive_a_toml_round_trip():
    s = settings(device_delays={BT: 165, "alsa_output.usb-*": 0})
    back = tomllib.loads(s.to_toml())["output"]["device_delays"]
    assert back == {BT: 165, "alsa_output.usb-*": 0}


def test_delays_can_be_changed_while_running():
    q = CommandQueue(Settings.load())
    q.submit({"output": {"device_delays": {BT: 180}, "delay_ms": 10}})
    assert q.pending.output.device_delays == {BT: 180}


def test_no_delay_passes_frames_straight_through():
    d = FrameDelay(0.0)
    frame = np.ones((3, 4))
    assert d.push(frame, 0.0) is frame
    assert len(d) == 0


def test_frames_come_out_the_delay_later():
    d, fps, out = FrameDelay(200.0), 60.0, []
    for i in range(60):
        due = d.push(np.full((3, 1), i), i / fps)
        out.append(None if due is None else int(due[0, 0]))
    first = next(i for i, v in enumerate(out) if v is not None)
    assert first == pytest.approx(0.2 * fps, abs=1)
    assert out[first] == 0
    # Steady state: one frame in, one frame out, always the same distance behind.
    assert all(out[i] == i - first for i in range(first, 60))


def test_shortening_the_delay_jumps_rather_than_fast_forwards():
    d, fps = FrameDelay(300.0), 60.0
    for i in range(60):
        d.push(np.full((3, 1), i), i / fps)
    d.set(100.0)
    due = d.push(np.full((3, 1), 60), 60 / fps)
    assert int(due[0, 0]) == 60 - 6           # the newest overdue frame, one send
    assert len(d) <= 6

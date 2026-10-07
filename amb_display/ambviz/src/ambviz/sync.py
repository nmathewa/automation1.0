"""Holding the lights back to meet sound that reaches the room late.

Loopback capture taps the mixer *before* the audio is sent to the speaker. For
a wired output that is the same moment it is heard, give or take a few
milliseconds. For Bluetooth it is not: the audio is encoded, sent over the air
and buffered in the speaker, which puts it 150-250 ms behind the tap on a
typical A2DP/SBC link -- so the strip runs *ahead* of what the listener hears.

That latency belongs to the output device, not to the music or the rig, so it
is configured per device and looked up whenever the default output changes:

1. ``output.device_delays`` -- an entry naming this device (exact name or a
   shell-style pattern such as ``bluez_output.*``). A calibrated value always
   wins.
2. ``output.bluetooth_delay_ms`` -- any Bluetooth sink without an entry. The
   sound server reports 0 latency for these, so the number cannot be read; it
   is a typical value to start calibrating from.
3. Nothing, for any other device.

``output.delay_ms`` is added on top in every case, for latency that belongs to
the rig rather than to the output device.

Standard library only, so the API can describe the delay without numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

from ambviz.settings import Settings


@dataclass(frozen=True)
class Sync:
    device: str
    """What the audio is passing through: a sink for loopback, an input for a
    microphone, empty for a generated or file source."""

    delay_ms: float
    """Total delay applied to the lights."""

    basis: str
    """Where the device part came from: ``configured``, ``bluetooth-default``
    or ``none``."""

    def to_dict(self) -> dict:
        return {"device": self.device, "delay_ms": round(self.delay_ms, 1),
                "basis": self.basis}


def is_bluetooth(device: str) -> bool:
    return device.startswith(("bluez_output.", "bluez_sink."))


def resolve(settings: Settings, device: str) -> Sync:
    out = settings.output
    device_ms, basis = 0.0, "none"
    if device:
        if device in out.device_delays:
            device_ms, basis = float(out.device_delays[device]), "configured"
        else:
            for pattern, ms in out.device_delays.items():
                if fnmatchcase(device, pattern):
                    device_ms, basis = float(ms), "configured"
                    break
            else:
                if is_bluetooth(device):
                    device_ms, basis = float(out.bluetooth_delay_ms), "bluetooth-default"
    return Sync(device=device, delay_ms=max(0.0, device_ms + out.delay_ms), basis=basis)

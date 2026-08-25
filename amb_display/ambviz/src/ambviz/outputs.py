"""LED output backends.

The wire format is the one the ESP8266 firmware speaks — a flat sequence of
4-byte ``|index|r|g|b|`` records over UDP. Because the sender only transmits
pixels whose value *changed*, a lost datagram leaves a pixel stale until it next
moves; :attr:`Output.full_refresh_interval` bounds that.

Backends are chosen by ``settings.output.device``. The virtual strip
(:mod:`ambviz.strip`) is not a separate backend on purpose: it listens on the
same UDP port and speaks the same protocol as the firmware, so pointing
``output.host`` at localhost exercises the real packet path rather than a mock.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np

from ambviz.settings import Settings


class Output:
    """Base class. Subclasses implement :meth:`_transmit`."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.n_pixels = settings.output.pixels
        self._prev = np.full((3, self.n_pixels), -1, dtype=int)
        self._last_full = 0.0
        self._gamma: np.ndarray | None = None
        if settings.output.gamma_correction:
            path: Path = settings.output.gamma_table_path()
            if not path.exists():
                raise FileNotFoundError(f"gamma table not found: {path}")
            self._gamma = np.load(path)
        self.packets = 0
        self.bytes = 0
        self.pixels_sent = 0

    # ── public API ───────────────────────────────────────────────────────────
    def send(self, pixels: np.ndarray) -> None:
        """Push a ``(3, n_pixels)`` array of 0-255 values to the strip."""
        if pixels.shape != (3, self.n_pixels):
            raise ValueError(
                f"expected pixel array of shape (3, {self.n_pixels}), got {pixels.shape}"
            )
        # Round rather than truncate. ``astype`` truncates toward zero, which
        # costs up to a full step everywhere and turns a pixel at 0.9 fully
        # dark. Invisible at the top of the range; at the luminance 1-3 an
        # effect sits at during a quiet passage, a step is a third of the
        # brightness.
        p = np.rint(np.clip(pixels, 0, 255)).astype(int)
        if self._gamma is not None:
            p = self._gamma[p]

        interval = self.settings.output.full_refresh_interval
        now = time.monotonic()
        force = bool(interval) and (now - self._last_full) >= interval
        if force:
            self._last_full = now
            changed = np.arange(self.n_pixels)
        else:
            changed = np.flatnonzero(np.any(p != self._prev, axis=0))

        if len(changed):
            self._transmit(p, changed)
            self.pixels_sent += len(changed)
        self._prev = p

    def reset(self) -> None:
        """Forget the cached strip state so the next frame is sent in full."""
        self._prev = np.full((3, self.n_pixels), -1, dtype=int)

    def blank(self) -> None:
        self.send(np.zeros((3, self.n_pixels), dtype=int))

    def close(self) -> None:
        pass

    def __enter__(self) -> "Output":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── subclass hook ────────────────────────────────────────────────────────
    def _transmit(self, pixels: np.ndarray, indices: np.ndarray) -> None:
        raise NotImplementedError


class UdpOutput(Output):
    """Sends ``|i|r|g|b|`` datagrams to an ESP8266 (or the simulator)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.address = (settings.output.host, settings.output.port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._chunk = max(1, settings.output.max_pixels_per_packet)

    def _transmit(self, pixels: np.ndarray, indices: np.ndarray) -> None:
        for start in range(0, len(indices), self._chunk):
            batch = indices[start:start + self._chunk]
            payload = bytes(
                np.stack(
                    [batch, pixels[0][batch], pixels[1][batch], pixels[2][batch]], axis=1
                )
                .astype(np.uint8)
                .ravel()
            )
            self._sock.sendto(payload, self.address)
            self.packets += 1
            self.bytes += len(payload)

    def close(self) -> None:
        self._sock.close()

    def __repr__(self) -> str:
        return f"UdpOutput({self.address[0]}:{self.address[1]}, {self.n_pixels} px)"


class NullOutput(Output):
    """Discards frames. For benchmarking the DSP chain without a strip."""

    def _transmit(self, pixels: np.ndarray, indices: np.ndarray) -> None:
        self.packets += 1
        self.bytes += len(indices) * 4

    def __repr__(self) -> str:
        return f"NullOutput({self.n_pixels} px)"


def make_output(settings: Settings) -> Output:
    backends = {"udp": UdpOutput, "none": NullOutput}
    try:
        return backends[settings.output.device](settings)
    except KeyError:
        raise ValueError(
            f"unknown output.device {settings.output.device!r}; "
            f"expected one of {sorted(backends)}"
        ) from None

"""A virtual LED strip.

Decodes the same 4-byte ``|index|r|g|b|`` records over UDP as the ESP8266
firmware in ``esp_tests/esp_pro_audio/test/main_wifi_audio.cpp``, so pointing a
visualizer at localhost exercises the real wire protocol -- packet splitting,
diff-only updates, index bounds -- rather than a mock of it.

Standard library only: importable on a machine with no numpy.
"""

from __future__ import annotations

import base64
import socket
import threading
import time
from collections import deque

IDLE_AFTER = 1.5      # seconds without a packet before the strip reads as idle
RATE_WINDOW = 1.0     # seconds of history behind the per-second rates
HISTORY_LEN = 60      # samples retained for the dashboard's sparklines
RECORD = 4            # bytes per pixel record on the wire


class VirtualStrip:
    """Decoded strip state plus traffic statistics. Thread-safe."""

    def __init__(self, pixels: int = 60, grow: bool = True):
        self._lock = threading.Lock()
        self.grow = grow
        self.pixels = pixels
        self.data = bytearray(pixels * 3)
        self.seq = 0
        self.packets = 0
        self.bytes = 0
        self.updates = 0
        self.malformed = 0
        self.out_of_range = 0
        self.last_packet = 0.0
        self.peer: str | None = None
        self._lit = 0  # tracked incrementally; snapshot() runs at 30 Hz per client
        self._events: deque[tuple[float, int, int]] = deque()
        self.history: deque[dict[str, float]] = deque(maxlen=HISTORY_LEN)
        self._last_sample = time.monotonic()
        self._sample_base = (0, 0)

    # ── ingest ───────────────────────────────────────────────────────────────
    def ingest(self, payload: bytes, peer: tuple[str, int] | None = None) -> None:
        """Decode one datagram.

        Trailing bytes that do not form a whole record are counted as malformed
        rather than silently dropped; indices past the end are either grown into
        or counted, depending on ``grow``.
        """
        now = time.monotonic()
        with self._lock:
            whole = len(payload) // RECORD
            if len(payload) % RECORD:
                self.malformed += 1
            for r in range(whole):
                index, red, green, blue = payload[r * RECORD: r * RECORD + RECORD]
                if index >= self.pixels:
                    if not self.grow:
                        self.out_of_range += 1
                        continue
                    self._resize(index + 1)
                at = index * 3
                was_lit = any(self.data[at: at + 3])
                self.data[at: at + 3] = bytes((red, green, blue))
                is_lit = red or green or blue
                if was_lit != bool(is_lit):
                    self._lit += 1 if is_lit else -1
            self.packets += 1
            self.bytes += len(payload)
            self.updates += whole
            self.seq += 1
            self.last_packet = now
            if peer is not None:
                self.peer = f"{peer[0]}:{peer[1]}"
            self._events.append((now, len(payload), whole))
            self._trim(now)

    def _resize(self, pixels: int) -> None:
        self.data.extend(bytearray((pixels - self.pixels) * 3))
        self.pixels = pixels

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] > RATE_WINDOW:
            self._events.popleft()

    # ── read ─────────────────────────────────────────────────────────────────
    def pixel(self, index: int) -> tuple[int, int, int]:
        with self._lock:
            return tuple(self.data[index * 3: index * 3 + 3])  # type: ignore[return-value]

    def snapshot(self) -> dict:
        """Everything a client needs for one frame."""
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            idle = self.last_packet == 0.0 or (now - self.last_packet) > IDLE_AFTER
            lit = self._lit
            return {
                "seq": self.seq,
                "pixels": self.pixels,
                "px": base64.b64encode(bytes(self.data)).decode("ascii"),
                "stats": {
                    "packet_rate": round(len(self._events) / RATE_WINDOW, 1),
                    "update_rate": round(sum(e[2] for e in self._events) / RATE_WINDOW, 1),
                    "byte_rate": round(sum(e[1] for e in self._events) / RATE_WINDOW, 1),
                    "coverage": round(100.0 * lit / self.pixels, 1) if self.pixels else 0.0,
                    "packets": self.packets,
                    "bytes": self.bytes,
                    "updates": self.updates,
                    "malformed": self.malformed,
                    "out_of_range": self.out_of_range,
                    "peer": self.peer,
                    "state": "idle" if idle else "live",
                    "since": round(now - self.last_packet, 1) if self.last_packet else None,
                },
                "history": list(self.history),
            }

    def sample_history(self) -> None:
        """Append one point per second, for client-side sparklines."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_sample
            if elapsed < 1.0:
                return
            packets, byts = self._sample_base
            self.history.append(
                {
                    "pps": round((self.packets - packets) / elapsed, 1),
                    "bps": round((self.bytes - byts) / elapsed, 1),
                }
            )
            self._sample_base = (self.packets, self.bytes)
            self._last_sample = now


class UdpReceiver(threading.Thread):
    """Feeds a :class:`VirtualStrip` from a UDP socket. This is the part
    standing in for the ESP8266."""

    daemon = True

    def __init__(self, strip: VirtualStrip, host: str = "0.0.0.0", port: int = 7777):
        super().__init__(name="ambviz-udp")
        self.strip = strip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.address = self.sock.getsockname()
        self._stop = threading.Event()

    def run(self) -> None:
        self.sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                payload, peer = self.sock.recvfrom(2048)
            except TimeoutError:
                continue
            except OSError:
                break
            self.strip.ingest(payload, peer)

    def stop(self) -> None:
        self._stop.set()
        self.sock.close()


class HistorySampler(threading.Thread):
    """Ticks the strip's history once a second, independent of any client."""

    daemon = True

    def __init__(self, strip: VirtualStrip):
        super().__init__(name="ambviz-history")
        self.strip = strip
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(0.5):
            self.strip.sample_history()

    def stop(self) -> None:
        self._stop.set()

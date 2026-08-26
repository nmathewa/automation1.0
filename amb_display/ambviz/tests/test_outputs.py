import socket

import numpy as np
import pytest

from ambviz.outputs import NullOutput, UdpOutput, make_output
from ambviz.settings import Settings
from ambviz.strip import VirtualStrip


@pytest.fixture
def wire():
    """A bound UDP socket plus settings pointing at it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)
    port = sock.getsockname()[1]

    def settings(**output):
        return Settings.load(overrides={"output": {
            "device": "udp", "host": "127.0.0.1", "port": port,
            "full_refresh_interval": 0, **output}})

    yield sock, settings
    sock.close()


def drain(sock, timeout=0.25):
    sock.settimeout(timeout)
    out = []
    while True:
        try:
            out.append(sock.recv(4096))
        except (TimeoutError, socket.timeout):
            return out


def test_first_frame_sends_every_pixel(wire):
    sock, settings = wire
    out = make_output(settings(pixels=8))
    out.send(np.zeros((3, 8)))
    assert sum(len(p) for p in drain(sock)) == 8 * 4


def test_unchanged_frame_sends_nothing(wire):
    sock, settings = wire
    out = make_output(settings(pixels=8))
    frame = np.full((3, 8), 5)
    out.send(frame)
    drain(sock)
    out.send(frame)
    assert drain(sock) == []


def test_single_pixel_change_sends_one_record(wire):
    sock, settings = wire
    out = make_output(settings(pixels=8))
    frame = np.zeros((3, 8))
    out.send(frame)
    drain(sock)
    frame[1, 5] = 99
    out.send(frame)
    packets = drain(sock)
    assert packets == [bytes((5, 0, 99, 0))]


def test_packets_are_split_at_the_chunk_size(wire):
    sock, settings = wire
    out = make_output(settings(pixels=120, max_pixels_per_packet=50))
    out.send(np.full((3, 120), 200))
    sizes = [len(p) for p in drain(sock)]
    assert sizes == [200, 200, 80]
    assert sum(sizes) // 4 == 120


def test_reset_forces_a_full_resend(wire):
    sock, settings = wire
    out = make_output(settings(pixels=8))
    frame = np.full((3, 8), 3)
    out.send(frame)
    drain(sock)
    out.reset()
    out.send(frame)
    assert sum(len(p) for p in drain(sock)) == 8 * 4


def test_full_refresh_interval_resends(wire):
    sock, settings = wire
    out = make_output(settings(pixels=4, full_refresh_interval=0.05))
    frame = np.full((3, 4), 7)
    out.send(frame)
    drain(sock, 0.1)
    out.send(frame)  # unchanged, but past the refresh interval
    assert sum(len(p) for p in drain(sock)) == 4 * 4


def test_wrong_shape_is_rejected(wire):
    _, settings = wire
    out = make_output(settings(pixels=8))
    with pytest.raises(ValueError, match=r"\(3, 8\)"):
        out.send(np.zeros((3, 4)))


def test_values_are_clipped_to_byte_range(wire):
    sock, settings = wire
    out = make_output(settings(pixels=2))
    out.send(np.array([[-50, 900], [0, 0], [0, 0]], dtype=float))
    payload = b"".join(drain(sock))
    assert payload[1] == 0 and payload[5] == 255


def test_round_trip_through_the_virtual_strip(wire):
    """What the output emits is what the virtual strip decodes."""
    sock, settings = wire
    out = make_output(settings(pixels=6))
    strip = VirtualStrip(6)
    frame = np.array([[10, 20, 30, 40, 50, 60],
                      [1, 2, 3, 4, 5, 6],
                      [200, 201, 202, 203, 204, 205]], dtype=float)
    out.send(frame)
    for packet in drain(sock):
        strip.ingest(packet)
    assert [strip.pixel(i) for i in range(6)] == [
        (10, 1, 200), (20, 2, 201), (30, 3, 202),
        (40, 4, 203), (50, 5, 204), (60, 6, 205)]


def test_null_output_counts_without_sending():
    out = NullOutput(Settings.load(overrides={"output": {"device": "none", "pixels": 4}}))
    out.send(np.full((3, 4), 1))
    assert out.packets == 1 and out.bytes == 16


def test_unknown_backend():
    s = Settings.load()
    s.output.device = "carrier-pigeon"
    with pytest.raises(ValueError, match="unknown output.device"):
        make_output(s)


def test_fractional_values_round_rather_than_truncate(wire):
    """A pixel at 0.9 is faintly lit, not off.

    ``astype(int)`` truncates toward zero, costing up to a full step
    everywhere. That is invisible near 255 but not at the luminance 1-3 an
    effect sits at during a quiet passage, where a step is a third of the
    brightness -- and it turns sub-1.0 values fully black.
    """
    sock, settings = wire
    out = make_output(settings(pixels=4))
    out.send(np.array([[0.4, 0.9, 1.9, 2.6]] * 3, dtype=float))
    payload = b"".join(drain(sock))
    assert [payload[i * 4 + 1] for i in range(4)] == [0, 1, 2, 3]


def test_rounding_does_not_break_the_diff(wire):
    """Values that round to the same byte must not retransmit."""
    sock, settings = wire
    out = make_output(settings(pixels=2))
    out.send(np.full((3, 2), 10.4))
    drain(sock)
    out.send(np.full((3, 2), 10.4999))
    assert drain(sock) == []

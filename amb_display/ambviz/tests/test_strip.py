import pytest

from ambviz.strip import VirtualStrip


def record(index, r, g, b):
    return bytes((index, r, g, b))


def test_decodes_records():
    strip = VirtualStrip(4)
    strip.ingest(record(0, 255, 0, 0) + record(3, 0, 0, 9), ("127.0.0.1", 5))
    assert strip.pixel(0) == (255, 0, 0)
    assert strip.pixel(3) == (0, 0, 9)
    assert strip.snapshot()["stats"]["updates"] == 2
    assert strip.snapshot()["stats"]["peer"] == "127.0.0.1:5"


def test_state_persists_between_packets():
    """The wire protocol is diff-only: unmentioned pixels must not be cleared."""
    strip = VirtualStrip(3)
    strip.ingest(record(0, 10, 20, 30))
    strip.ingest(record(1, 40, 50, 60))
    assert strip.pixel(0) == (10, 20, 30)


def test_trailing_bytes_counted_not_dropped_silently():
    strip = VirtualStrip(2)
    strip.ingest(record(0, 1, 2, 3) + b"\x01\x02")
    stats = strip.snapshot()["stats"]
    assert stats["malformed"] == 1
    assert stats["updates"] == 1
    assert strip.pixel(0) == (1, 2, 3)


def test_grow_extends_the_strip():
    strip = VirtualStrip(2, grow=True)
    strip.ingest(record(5, 7, 7, 7))
    assert strip.pixels == 6
    assert strip.pixel(5) == (7, 7, 7)
    assert strip.snapshot()["stats"]["out_of_range"] == 0


def test_fixed_rejects_out_of_range():
    strip = VirtualStrip(2, grow=False)
    strip.ingest(record(5, 7, 7, 7))
    assert strip.pixels == 2
    assert strip.snapshot()["stats"]["out_of_range"] == 1


def test_idle_until_a_packet_arrives():
    strip = VirtualStrip(2)
    assert strip.snapshot()["stats"]["state"] == "idle"
    strip.ingest(record(0, 1, 1, 1))
    assert strip.snapshot()["stats"]["state"] == "live"


def test_coverage_counts_lit_pixels():
    strip = VirtualStrip(4)
    strip.ingest(record(0, 1, 0, 0) + record(1, 0, 0, 1))
    assert strip.snapshot()["stats"]["coverage"] == pytest.approx(50.0)


def test_snapshot_pixels_are_base64_rgb():
    import base64

    strip = VirtualStrip(2)
    strip.ingest(record(0, 1, 2, 3))
    raw = base64.b64decode(strip.snapshot()["px"])
    assert len(raw) == 6
    assert tuple(raw[:3]) == (1, 2, 3)

import json
import socket
import urllib.error
import urllib.request

import pytest

from ambviz.api import ApiServer
from ambviz.control import CommandQueue
from ambviz.settings import CONTRACT, Settings
from ambviz.strip import UdpReceiver, VirtualStrip


class FakeEngine:
    """Stands in for Visualizer: the API only needs snapshot()."""

    def __init__(self):
        self.frames = 0

    def snapshot(self):
        self.frames += 1
        return {"frames": self.frames, "effect": "spectrum", "fps": 60.0, "mel": [0.1, 0.2]}


def serve(providers, **kwargs):
    return ApiServer(providers, host="127.0.0.1", port=0, **kwargs).start()


def get(server, path):
    with urllib.request.urlopen(f"{server.url}{path}", timeout=3) as r:
        return r.status, r.headers, r.read()


def get_json(server, path):
    status, headers, body = get(server, path)
    return status, headers, json.loads(body)


def post(server, path, payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(f"{server.url}{path}", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def strip_api():
    strip = VirtualStrip(4)
    server = serve({"strip": strip.snapshot}, settings=Settings.load())
    yield server, strip
    server.stop()


# ── provider composition ─────────────────────────────────────────────────────
def test_strip_only_state_has_no_engine(strip_api):
    server, _ = strip_api
    _, _, body = get_json(server, "/api/state")
    assert set(body) == {"strip"}


def test_engine_only_state_has_no_strip():
    server = serve({"engine": FakeEngine().snapshot})
    try:
        _, _, body = get_json(server, "/api/state")
        assert set(body) == {"engine"}
        assert body["engine"]["effect"] == "spectrum"
    finally:
        server.stop()


def test_both_providers_are_namespaced():
    strip = VirtualStrip(4)
    server = serve({"strip": strip.snapshot, "engine": FakeEngine().snapshot})
    try:
        _, _, body = get_json(server, "/api/state")
        assert set(body) == {"strip", "engine"}
        assert "px" in body["strip"] and "mel" in body["engine"]
    finally:
        server.stop()


def test_at_least_one_provider_is_required():
    with pytest.raises(ValueError, match="at least one telemetry provider"):
        ApiServer({}, host="127.0.0.1", port=0)


# ── health / contract ────────────────────────────────────────────────────────
def test_health_reports_contract_and_providers(strip_api):
    server, _ = strip_api
    status, headers, body = get_json(server, "/api/health")
    assert status == 200
    assert body["ok"] is True
    assert body["contract"] == CONTRACT
    assert body["providers"] == ["strip"]
    assert body["controllable"] is False
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_role_names_the_providers():
    strip = VirtualStrip(2)
    server = serve({"strip": strip.snapshot, "engine": FakeEngine().snapshot})
    try:
        assert get_json(server, "/api/health")[2]["role"] == "engine+strip"
    finally:
        server.stop()


def test_root_documents_itself(strip_api):
    server, _ = strip_api
    _, _, body = get_json(server, "/")
    assert "/api/state" in body["endpoints"]
    assert body["contract"] == CONTRACT


# ── settings ─────────────────────────────────────────────────────────────────
def test_settings_names_its_source_as_strip(strip_api):
    server, _ = strip_api
    _, _, body = get_json(server, "/api/settings")
    assert body["source"] == "strip"
    assert body["settings"]["output"]["pixels"] == 60


def test_engine_settings_win_when_both_present():
    """Regression: the API used to report the serve process's settings even when
    a visualizer with different ones was driving the strip."""
    strip = VirtualStrip(4)
    server = serve({"strip": strip.snapshot, "engine": FakeEngine().snapshot},
                   settings=Settings.load(overrides={"output": {"pixels": 108}}))
    try:
        _, _, body = get_json(server, "/api/settings")
        assert body["source"] == "engine"
        assert body["settings"]["output"]["pixels"] == 108
    finally:
        server.stop()


def test_settings_absent_is_404():
    strip = VirtualStrip(2)
    server = serve({"strip": strip.snapshot})
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{server.url}/api/settings", timeout=3)
        assert exc.value.code == 404
    finally:
        server.stop()


def test_settings_toml_is_downloadable_and_reloadable(strip_api, tmp_path):
    server, _ = strip_api
    status, headers, body = get(server, "/api/settings.toml")
    assert status == 200
    assert headers["Content-Type"].startswith("text/plain")
    assert "attachment" in headers["Content-Disposition"]
    path = tmp_path / "dump.toml"
    path.write_bytes(body)
    assert Settings.load(path, env=False).output.pixels == 60


# ── control ──────────────────────────────────────────────────────────────────
@pytest.fixture
def control_api():
    settings = Settings.load()
    commands = CommandQueue(settings)
    server = serve({"engine": FakeEngine().snapshot}, settings=settings, commands=commands)
    yield server, commands
    server.stop()


def test_post_effect_queues_a_patch(control_api):
    server, commands = control_api
    status, body = post(server, "/api/effect", {"name": "energy"})
    assert status == 200
    assert body["applied"] == {"effect": {"name": "energy"}}
    assert commands.drain() == [{"effect": {"name": "energy"}}]


def test_post_settings_echoes_the_new_settings(control_api):
    server, _ = control_api
    status, body = post(server, "/api/settings", {"effect": {"brightness": 0.25}})
    assert status == 200
    assert body["settings"]["effect"]["brightness"] == 0.25


@pytest.mark.parametrize(
    "payload, status",
    [
        ({"effect": {"brightness": 9}}, 400),          # out of range
        ({"dsp": {"max_frequency": 30000}}, 400),      # above Nyquist
        ({"effect": {"nope": 1}}, 400),                # unknown setting
        ({"output": {"pixels": 120}}, 409),            # exists but restart-only
        ({}, 400),                                     # empty patch
    ],
)
def test_bad_patches_are_rejected_and_never_queued(control_api, payload, status):
    server, commands = control_api
    assert post(server, "/api/settings", payload)[0] == status
    assert commands.drain() == []


def test_malformed_json_is_400(control_api):
    server, commands = control_api
    assert post(server, "/api/settings", b"not json")[0] == 400
    assert commands.drain() == []


def test_effect_without_a_name_is_400(control_api):
    server, _ = control_api
    assert post(server, "/api/effect", {"nom": "energy"})[0] == 400


def test_read_only_process_refuses_writes(strip_api):
    server, _ = strip_api
    status, body = post(server, "/api/effect", {"name": "energy"})
    assert status == 503
    assert "read-only" in body["error"]


def test_unknown_post_route_is_404(control_api):
    server, _ = control_api
    assert post(server, "/api/nope", {})[0] == 404


# ── transport ────────────────────────────────────────────────────────────────
def test_unknown_route_is_404(strip_api):
    server, _ = strip_api
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{server.url}/nope", timeout=3)
    assert exc.value.code == 404


def test_cors_preflight_allows_post(strip_api):
    server, _ = strip_api
    req = urllib.request.Request(f"{server.url}/api/state", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=3) as r:
        assert r.status == 204
        assert r.headers["Access-Control-Allow-Origin"] == "*"
        assert "POST" in r.headers["Access-Control-Allow-Methods"]


def test_stream_emits_namespaced_events(strip_api):
    server, strip = strip_api
    strip.ingest(bytes((0, 1, 2, 3)))
    with urllib.request.urlopen(f"{server.url}/api/stream", timeout=5) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        for line in r:
            if line.startswith(b"data: "):
                payload = json.loads(line[6:])
                assert set(payload) == {"strip"}
                assert "px" in payload["strip"]
                break


def test_udp_receiver_feeds_the_strip():
    strip = VirtualStrip(4)
    receiver = UdpReceiver(strip, "127.0.0.1", 0)
    receiver.start()
    try:
        host, port = receiver.address[0], receiver.address[1]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(bytes((2, 4, 5, 6)), (host, port))
        for _ in range(50):
            if strip.packets:
                break
            import time
            time.sleep(0.02)
        assert strip.pixel(2) == (4, 5, 6)
    finally:
        receiver.stop()

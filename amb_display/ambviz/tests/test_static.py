import json
import urllib.error
import urllib.request

import pytest

from ambviz.api import ApiServer
from ambviz.settings import Settings
from ambviz.strip import VirtualStrip


@pytest.fixture
def site(tmp_path):
    (tmp_path / "index.html").write_text("<h1>dashboard</h1>")
    (tmp_path / "app.js").write_text("console.log(1)")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "index.html").write_text("<h1>sub</h1>")
    # A secret one directory up, to prove traversal cannot reach it.
    (tmp_path.parent / "outside.txt").write_text("SHOULD NOT BE SERVED")

    strip = VirtualStrip(4)
    server = ApiServer({"strip": strip.snapshot}, host="127.0.0.1", port=0,
                       settings=Settings.load(), static_dir=tmp_path).start()
    yield server, tmp_path
    server.stop()


def get(server, path):
    try:
        with urllib.request.urlopen(f"{server.url}{path}", timeout=3) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def test_root_serves_index(site):
    server, _ = site
    status, ctype, body = get(server, "/")
    assert status == 200
    assert b"dashboard" in body
    assert ctype.startswith("text/html")


def test_named_file_and_mime_type(site):
    server, _ = site
    status, ctype, body = get(server, "/app.js")
    assert status == 200
    assert b"console.log" in body
    assert "javascript" in ctype


def test_directory_serves_its_index(site):
    server, _ = site
    assert b"sub" in get(server, "/sub/")[2]


def test_api_routes_win_over_the_file_system(site):
    """A file on disk must never shadow the API."""
    server, root = site
    (root / "api").mkdir()
    (root / "api" / "health").write_text("decoy")
    status, _, body = get(server, "/api/health")
    assert status == 200
    assert json.loads(body)["ok"] is True


@pytest.mark.parametrize("path", [
    "/../outside.txt",
    "/sub/../../outside.txt",
    "/%2e%2e/outside.txt",
    "/....//outside.txt",
])
def test_traversal_cannot_escape_the_root(site, path):
    server, _ = site
    status, _, body = get(server, path)
    assert status in (403, 404)
    assert b"SHOULD NOT BE SERVED" not in body


def test_missing_file_is_404(site):
    assert get(site[0], "/nope.html")[0] == 404


def test_without_a_static_root_the_json_pointer_is_served():
    strip = VirtualStrip(2)
    server = ApiServer({"strip": strip.snapshot}, host="127.0.0.1", port=0).start()
    try:
        assert json.loads(get(server, "/")[2])["service"] == "ambviz"
    finally:
        server.stop()


def test_missing_static_dir_is_rejected(tmp_path):
    strip = VirtualStrip(2)
    with pytest.raises(ValueError, match="static directory not found"):
        ApiServer({"strip": strip.snapshot}, host="127.0.0.1", port=0,
                  static_dir=tmp_path / "nope")


# ── private network access ───────────────────────────────────────────────────
def preflight(server, ask):
    headers = {"Origin": "https://nmathewa.github.io"}
    if ask:
        headers["Access-Control-Request-Private-Network"] = "true"
    req = urllib.request.Request(f"{server.url}/api/state", method="OPTIONS", headers=headers)
    with urllib.request.urlopen(req, timeout=3) as r:
        return r.headers


def test_pna_header_answered_when_asked(site):
    assert preflight(site[0], ask=True)["Access-Control-Allow-Private-Network"] == "true"


def test_pna_header_absent_otherwise(site):
    assert preflight(site[0], ask=False).get("Access-Control-Allow-Private-Network") is None

"""HTTP telemetry and control API.

The package publishes state here; it renders nothing. Any dashboard is a
separate client that calls these endpoints -- which is why every response
carries permissive CORS headers, so a UI served from a different origin (or
opened straight off disk) can read them.

A process registers whichever **providers** it has. ``ambviz serve`` has a
``strip``; ``ambviz run --api`` has an ``engine``; ``ambviz run --virtual`` has
both. ``/api/state`` returns exactly the providers present, namespaced, so a
client can tell what it is looking at instead of guessing.

Endpoints
---------
``GET  /api/health``        liveness, version, contract, providers
``GET  /api/state``         ``{"strip": {...}, "engine": {...}}`` -- present providers only
``GET  /api/stream``        the same payload as Server-Sent Events, rate-capped
``GET  /api/settings``      effective settings, and whose they are
``GET  /api/settings.toml`` the same, ready to save to disk
``POST /api/settings``      partial patch, e.g. ``{"effect": {"brightness": 0.4}}``
``POST /api/effect``        ``{"name": "energy"}``, sugar over the above

With ``static_dir`` set, anything not matching ``/api/*`` is served from that
directory. The package still ships no HTML -- it serves files the operator points
at -- but hosting a dashboard from the API means the page and the API share an
origin, which sidesteps browser mixed-content rules entirely.

The POST routes exist only when the process was given a command queue; writing
to a read-only process answers 503.

Standard library only.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from ambviz import __version__
from ambviz.control import CommandQueue, NotControllable
from ambviz.settings import CONTRACT, Settings

Provider = Callable[[], dict[str, Any]]

MAX_BODY = 64 * 1024
# Cheap per-provider markers used to detect "has anything changed?" without
# serialising the whole payload on every stream tick.
_MARKERS = {"strip": "seq", "engine": "frames"}


class ApiServer:
    """Serves telemetry from a set of named providers.

    Parameters
    ----------
    providers:
        ``{"strip": strip.snapshot, "engine": visualizer.snapshot}`` -- any
        subset. Each callable returns a JSON-serialisable dict.
    settings:
        The settings this process resolved, published at ``/api/settings``.
    commands:
        A :class:`~ambviz.control.CommandQueue`. Its presence is what enables
        the POST routes; without one the API is read-only.
    static_dir:
        A directory to serve for non-API routes -- typically a dashboard. Same
        origin as the API, so no mixed-content rules apply.
    """

    def __init__(
        self,
        providers: dict[str, Provider],
        host: str = "127.0.0.1",
        port: int = 8080,
        stream_fps: float = 30.0,
        settings: Settings | None = None,
        commands: CommandQueue | None = None,
        static_dir: str | Path | None = None,
    ):
        if not providers:
            raise ValueError("at least one telemetry provider is required")
        self.providers = providers
        self.stream_fps = stream_fps
        self.settings = settings
        self.commands = commands

        self.static_dir: Path | None = None
        if static_dir is not None:
            root = Path(static_dir).expanduser().resolve()
            if not root.is_dir():
                raise ValueError(f"static directory not found: {root}")
            self.static_dir = root

        self._server = ThreadingHTTPServer((host, port), _make_handler(self))
        self.address = self._server.server_address
        self._thread: threading.Thread | None = None
        self._serving = False

    # ── introspection ────────────────────────────────────────────────────────
    @property
    def url(self) -> str:
        return f"http://{self.address[0]}:{self.address[1]}"

    @property
    def role(self) -> str:
        return "+".join(sorted(self.providers))

    @property
    def settings_source(self) -> str:
        """Whose settings ``/api/settings`` reports.

        The engine's win when both are present: it is the process actually
        driving pixels, so its pixel count and effect are the true ones.
        """
        return "engine" if "engine" in self.providers else next(iter(sorted(self.providers)))

    def state(self) -> dict[str, Any]:
        return {name: provider() for name, provider in self.providers.items()}

    def effects(self) -> list[str]:
        """Effect names this build offers, or ``[]`` on a numpy-free process."""
        try:
            from ambviz.effects import EFFECTS
        except ImportError:
            return []          # `serve` runs without numpy; it renders nothing
        return sorted(EFFECTS)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def serve_forever(self) -> None:
        self._serving = True
        try:
            self._server.serve_forever()
        finally:
            self._serving = False

    def start(self) -> "ApiServer":
        self._thread = threading.Thread(target=self.serve_forever, daemon=True, name="ambviz-api")
        self._thread.start()
        return self

    def stop(self) -> None:
        # shutdown() blocks until the serve_forever loop acknowledges it, so
        # calling it on a server that was never started hangs forever.
        if self._serving:
            self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "ApiServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _make_handler(api: ApiServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"ambviz/{__version__}"

        def log_message(self, *args: object) -> None:
            pass  # clients poll continuously; access logs are noise

        # ── CORS: the dashboard is a separate origin by design ───────────────
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            # Chrome gates requests from a public HTTPS page to a loopback
            # address behind this handshake; without the answer it drops them.
            # Only sent when asked for, so it never appears on ordinary requests.
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        # ── reads ────────────────────────────────────────────────────────────
        def do_GET(self) -> None:  # noqa: N802
            route = self.path.split("?")[0].rstrip("/") or "/"
            if route == "/api/health":
                self._json({
                    "ok": True,
                    "version": __version__,
                    "contract": CONTRACT,
                    "role": api.role,
                    "providers": sorted(api.providers),
                    "controllable": api.commands is not None,
                    "stream_fps": api.stream_fps,
                    # So a client can offer every effect this build has rather
                    # than a list hardcoded when it was written.
                    "effects": api.effects(),
                })
            elif route == "/api/state":
                self._json(api.state())
            elif route == "/api/settings":
                if api.settings is None:
                    self._json({"error": "this process publishes no settings"}, 404)
                else:
                    self._json({
                        "source": api.settings_source,
                        "controllable": api.commands is not None,
                        "settings": api.settings.to_dict(),
                    })
            elif route == "/api/settings.toml":
                if api.settings is None:
                    self._json({"error": "this process publishes no settings"}, 404)
                else:
                    self._text(api.settings.to_toml(), "text/plain; charset=utf-8",
                               filename="ambviz.toml")
            elif route == "/api/stream":
                self._stream()
            elif api.static_dir is not None and not route.startswith("/api/"):
                # /api/* always wins, so a file on disk can never shadow the API.
                self._static(route)
            elif route == "/":
                self._json({
                    "service": "ambviz",
                    "version": __version__,
                    "contract": CONTRACT,
                    "role": api.role,
                    "endpoints": ["/api/health", "/api/state", "/api/stream",
                                  "/api/settings", "/api/settings.toml"],
                    "note": "This package serves data only. Point a dashboard at these endpoints.",
                })
            else:
                self._json({"error": "not found", "path": route}, 404)

        def _static(self, route: str) -> None:
            """Serve a file from the static root, and nothing outside it."""
            root = api.static_dir
            assert root is not None
            relative = route.lstrip("/") or "index.html"
            candidate = (root / relative).resolve()

            # This is a hand-written handler, so containment is checked here
            # rather than inherited from SimpleHTTPRequestHandler.
            if candidate != root and root not in candidate.parents:
                self._json({"error": "forbidden", "path": route}, 403)
                return
            if candidate.is_dir():
                candidate = candidate / "index.html"
            if not candidate.is_file():
                self._json({"error": "not found", "path": route}, 404)
                return

            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._send(candidate.read_bytes(), 200, content_type)

        # ── writes ───────────────────────────────────────────────────────────
        def do_POST(self) -> None:  # noqa: N802
            route = self.path.split("?")[0].rstrip("/") or "/"
            if route not in ("/api/settings", "/api/effect"):
                self._json({"error": "not found", "path": route}, 404)
                return
            if api.commands is None:
                self._json({"error": "this process is read-only; no visualizer is attached"}, 503)
                return

            try:
                body = self._body()
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return

            if route == "/api/effect":
                name = body.get("name")
                if not isinstance(name, str):
                    self._json({"error": 'expected {"name": "<effect>"}'}, 400)
                    return
                patch = {"effect": {"name": name}}
            else:
                patch = body

            try:
                applied = api.commands.submit(patch)
            except NotControllable as exc:
                self._json({"error": str(exc.args[0])}, 409)
                return
            except KeyError as exc:
                self._json({"error": str(exc.args[0])}, 400)
                return
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return

            # `settings` is the state after the run loop drains the queue, which
            # it does on its next frame -- so a client can reflect the change
            # immediately instead of polling for it.
            self._json({
                "applied": applied,
                "queued": len(api.commands),
                "settings": api.commands.pending.to_dict(),
            })

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("empty request body")
            if length > MAX_BODY:
                raise ValueError(f"request body larger than {MAX_BODY} bytes")
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {exc.msg}") from None
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            return payload

        # ── responses ────────────────────────────────────────────────────────
        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(json.dumps(payload).encode(), status, "application/json")

        def _text(self, text: str, content_type: str, filename: str | None = None) -> None:
            extra = {"Content-Disposition": f'attachment; filename="{filename}"'} if filename else {}
            self._send(text.encode(), 200, content_type, extra)

        def _send(self, body: bytes, status: int, content_type: str,
                  extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            """One SSE message per changed frame, plus a keepalive each second."""
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            interval = 1.0 / api.stream_fps
            last_mark, last_sent = None, 0.0
            try:
                while True:
                    state = api.state()
                    mark = tuple(
                        state.get(name, {}).get(key)
                        for name, key in _MARKERS.items() if name in state
                    )
                    now = time.monotonic()
                    # Resend on change, and at least once a second so idle
                    # timers and rate counters keep ticking on the client.
                    if mark != last_mark or now - last_sent > 1.0:
                        last_mark, last_sent = mark, now
                        self.wfile.write(f"data: {json.dumps(state)}\n\n".encode())
                        self.wfile.flush()
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler

"""Audio-reactive LED strip driver.

The package covers everything from audio capture to the UDP packets a WS2812
controller consumes, plus a virtual strip that decodes those same packets for
testing without hardware.

It deliberately contains **no user interface**. State is published over a small
read-only HTTP API (:mod:`ambviz.api`); the dashboard is a separate, purely
client-side artefact that calls it.

Submodules are imported lazily so that the standard-library-only parts
(:mod:`ambviz.settings`, :mod:`ambviz.strip`, :mod:`ambviz.api`) stay usable on a
machine with no numpy or scipy installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "Visualizer",
    "VirtualStrip",
    "ApiServer",
    "make_output",
    "make_source",
    "__version__",
]


def __getattr__(name: str):  # noqa: D103 - lazy re-exports
    if name == "Settings":
        from ambviz.settings import Settings
        return Settings
    if name == "VirtualStrip":
        from ambviz.strip import VirtualStrip
        return VirtualStrip
    if name == "ApiServer":
        from ambviz.api import ApiServer
        return ApiServer
    if name == "Visualizer":
        from ambviz.pipeline import Visualizer
        return Visualizer
    if name == "make_output":
        from ambviz.outputs import make_output
        return make_output
    if name == "make_source":
        from ambviz.sources import make_source
        return make_source
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

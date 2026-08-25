# ambviz

Audio-reactive LED strip driver: live audio in, WS2812 pixels out over UDP to an
ESP8266 — plus a virtual strip so the whole chain is testable with no LED
hardware, no ESP and no microphone.

```
audio ──▶ Mel filterbank ──▶ effect ──▶ UDP |i|r|g|b| ──▶ ESP8266
mic|synth|wav        │                                └──▶ virtual strip
                     └────────── engine telemetry ─────────────┴──▶ HTTP API
```

**This package has no user interface.** It publishes state over a small HTTP API
and accepts setting changes there; any dashboard is a separate client that calls
it. The one in this repo lives at [`../dashboard/`](../dashboard/).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .          # add [mic] for live microphone input
.venv/bin/pip install -e ".[dev]"   # and pytest
```

Requires Python 3.11+. `numpy` and `scipy` are needed to *drive* a strip;
`ambviz serve` deliberately imports neither, so the virtual strip and API run on
a bare interpreter.

## Use

```bash
ambviz run --virtual --source synth
```

One process: the visualizer, a virtual strip receiving its packets, and the API
serving both. No microphone, no LED hardware. Then open
`../dashboard/index.html`.

To watch real hardware from another machine instead, run the two halves apart:
`ambviz serve` there, `ambviz run --host <strip>` here.

Against real hardware:

```bash
ambviz run --host ambviz.local --pixels 60 --source mic     # add --api for telemetry
```

Other commands:

```bash
ambviz config --pixels 60      # show the effective settings as TOML
ambviz devices                 # list audio inputs
ambviz --help
```

## Configuration

Precedence, lowest to highest:

```
dataclass defaults  <  --config file.toml  <  AMBVIZ_* environment  <  CLI flags
```

Copy `config.example.toml` per rig instead of duplicating code:

```bash
ambviz run --config strip-livingroom.toml
AMBVIZ_OUTPUT_HOST=192.168.1.35 ambviz run --pixels 108
```

Settings are validated on load, so mistakes fail immediately and by name: a
frequency above Nyquist, an FPS the strip physically cannot accept, an unknown
key, a pixel count past the firmware's index limit.

## API

Default `http://127.0.0.1:8080`. Permissive CORS, so a dashboard on any origin
(including `file://`) can read it.

A process registers whichever **providers** it has, and `/api/state` returns
exactly those, namespaced — so a client can tell what it is looking at:

| Command | Providers |
|---|---|
| `ambviz serve` | `strip` |
| `ambviz run --api` | `engine` |
| `ambviz run --virtual` | `engine` + `strip` |

| Endpoint | Returns |
|---|---|
| `GET /api/health` | version, **contract**, providers, whether writes are accepted |
| `GET /api/state` | `{"strip": {...}, "engine": {...}}` — present providers only |
| `GET /api/stream` | the same payload as Server-Sent Events, rate-capped |
| `GET /api/settings` | effective settings, and **whose** they are |
| `GET /api/settings.toml` | the same, ready to save to disk |
| `POST /api/settings` | partial patch, e.g. `{"effect": {"brightness": 0.4}}` |
| `POST /api/effect` | `{"name": "energy"}`, sugar over the above |

`strip` carries `packet_rate`, `update_rate`, `byte_rate`, `coverage`, totals,
`malformed`, `out_of_range`, `peer` and `state` (`live` / `idle`).
`engine` carries `fps`, `volume`, `silent`, the active effect, the Mel band
levels and their **centre frequencies in Hz**.

`contract` is an integer that changes when a response shape changes
incompatibly, so an older dashboard can say so instead of misrendering.

### Live control

POST routes exist only where a visualizer is attached; a read-only process
answers 503. Patches are validated against a copy of the settings *before* being
queued, so a bad value never reaches the audio thread — 400 for an invalid or
unknown setting, 409 for one that exists but is restart-only.

Changeable while running: effect, brightness, mirror, frequency range, band
count, `mel_exponent`, gamma, and all eight smoothing alphas. Pixel count,
sample rate, FPS and the audio source resize buffers or reopen devices, so they
stay restart-only.

## Layout

| Module | Role | Needs numpy |
|---|---|---|
| `settings.py` | every tunable, loader, validation | no |
| `strip.py` | virtual strip: UDP decode + statistics | no |
| `control.py` | validated setting patches, queued for the run loop | no |
| `api.py` | HTTP/SSE telemetry and control | no |
| `sources.py` | `mic` / `synth` / `wav` audio input | yes |
| `dsp.py` | `ExpFilter`, `MelBank` | yes |
| `effects.py` | `spectrum`, `energy`, `scroll` | yes |
| `pipeline.py` | `Visualizer` — audio frame to pixels | yes |
| `outputs.py` | UDP / null backends, diff-only encoding | yes |
| `cli.py` | `ambviz run \| serve \| config \| devices` | lazy |

Threading: the API runs on its own thread and never touches the visualizer
directly. It validates a patch, queues it, and the run loop applies it between
frames — so changing the effect or the frequency range cannot land mid-render.

## Effects

| Name | Behaviour |
|---|---|
| `spectrum` | filterbank across the strip: red = spectral contrast, green = frame-to-frame change, blue = smoothed level |
| `energy` | bars growing from the origin, one per frequency third |
| `scroll` | colour injected at the origin, drifting outward and decaying |

## Protocol

Flat 4-byte records, `|index|r|g|b|`, over UDP — what the ESP8266 firmware
expects. Only changed pixels are sent, so a dropped datagram would leave a pixel
stale; `output.full_refresh_interval` (default 2 s) bounds that.

The index is one byte, so 255 pixels is the protocol ceiling; settings warn above
128. The firmware in `../esp_tests/esp_pro_audio/` now drops out-of-range indices
and reports the count over serial instead of wrapping.

## Tests

```bash
.venv/bin/python -m pytest
```

139 tests covering settings precedence and validation, the diff-only wire
encoding, packet splitting, a round trip from `UdpOutput` through
`VirtualStrip`, effect output shapes at odd and even pixel counts, filter
isolation between visualizers, provider composition, control patches being
rejected before they queue, selective rebuilds on `apply`, and every API
endpoint including CORS and SSE.

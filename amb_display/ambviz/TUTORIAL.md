# ambviz in ten minutes

Start with no hardware at all and add pieces one at a time. Each step says what
you should see, so a step that looks wrong is caught before the next one builds
on it.

[`README.md`](README.md) is the reference — every setting, endpoint and module.
This is the path through it.

---

## 1. Install

```bash
cd amb_display/ambviz
python3 -m venv .venv
.venv/bin/pip install -e ".[mic,dev]"
```

Python 3.11+. `mic` adds `sounddevice`, which is also what live audio capture
uses; `dev` adds pytest. Check it:

```bash
.venv/bin/python -m pytest -q      # ~85 s
```

Optional extras, neither needed to start:

- `pip install -e ".[scene]"` — YAMNet scene classification, ~40 MB.
- `pip install demucs` — Demucs stem separation. Pulls in torch, and the model
  weights are a few hundred MB on first use.

---

## 2. Run it with nothing attached

```bash
.venv/bin/ambviz run --virtual --source synth --static ../dashboard
```

One process: a generated audio source, the visualizer, a virtual strip receiving
the UDP packets, and the API serving the dashboard from the same origin.

Open **http://127.0.0.1:8080/**. You should see a strip of moving colour, a
packet rate near 60/s, and the frame rate settling around 60 fps in the terminal.

If the page loads but the strip is black, the engine is fine and the audio is
not — go to step 5.

---

## 3. Point it at real music

**Linux (PulseAudio/PipeWire)** captures what the machine is playing, in stereo,
with no microphone involved:

```bash
.venv/bin/ambviz monitors                       # what can be captured
.venv/bin/ambviz run --virtual --source loopback --effect auto --static ../dashboard
```

**macOS** has no monitor source. Install [BlackHole](https://existential.audio/blackhole/)
(`brew install blackhole-2ch`), make a Multi-Output Device in *Audio MIDI Setup*
so sound still reaches your speakers, then capture it as an ordinary input:

```bash
.venv/bin/ambviz devices                        # find its exact name
.venv/bin/ambviz run --virtual --source mic --input-device "BlackHole" \
    --effect auto --static ../dashboard
```

Either way, play something. `--effect auto` puts the director in charge: it
watches what the music is doing and changes animation when the music changes,
not on a timer.

To reach the dashboard from a phone or laptop, add `--api-host 0.0.0.0` and open
`http://<this-machine>:8080/`. The API is unauthenticated and accepts setting
changes, so if you have a firewall, scope the rule to your own subnet rather
than opening the port to everything.

---

## 4. Change things while it runs

Use the dashboard's controls, or the API directly:

```bash
curl -s -X POST -H 'Content-Type: application/json' \
     -d '{"effect": {"brightness": 0.4}}' localhost:8080/api/settings
curl -s localhost:8080/api/settings.toml -o my-rig.toml    # keep what you tuned
.venv/bin/ambviz run --config my-rig.toml
```

A patch is validated before it is queued, so a bad value never reaches the audio
thread: **400** for an invalid or unknown setting, **409** for one that exists
but can only be set at startup. Anything that resizes the strip — pixel count,
`output.segments`, sample rate — is in that second group.

---

## 5. When nothing seems to happen

Press **Flow** in the dashboard header (or add `?debug=1` to the URL). It draws
the pipeline as it is running right now, in three bands — fast path, slow layer,
render — and every box is one of:

| | |
|---|---|
| **bright** | carrying signal, with its current reading |
| **dim** | running, nothing to show this instant |
| **dashed** | not running, and the box says why |

Read it left to right and stop at the first box that is not what you expect.
`Source · silent` means no audio is arriving at all. `Mid / side · mono source`
means you have audio but only one channel, which also explains a dark
`Vocal suppress` and `Stereo image`. `Stems · mood.stem_weight is 0` is a
setting, not a fault.

Boxes flash amber when something fires: a beat, a director switch, an accent.

---

## 6. A room instead of a strip

One strip in front of you can only say *what* the music is doing. Add a wall
either side and it can say *where*:

```bash
.venv/bin/ambviz run --config config.room.toml --virtual --static ../dashboard
```

`config.room.toml` describes three runs wired as one chain —
`segments = [30, 60, 30]`, left wall, front, right wall. The widest run is the
front and carries the animation exactly as a single strip would; the walls are
driven by their own channel, so a hard-panned guitar lights one side and not the
other. Mono material falls back to a wash off the nearest end of the front.

Sides need stereo. On a mono source the walls will wash rather than split, which
the flow panel will tell you.

---

## 7. Drive real LEDs

Flash `../esp_tests/esp_pro_audio/` to an ESP8266, then:

```bash
.venv/bin/ambviz run --host <esp-ip> --pixels 60 --source loopback --api
```

Pixels go out as flat `|index|r|g|b|` records over UDP, and only changed pixels
are sent. `--virtual` and `--host` are not exclusive: keep `--virtual` to watch
the same output in the browser while the strip runs.

Start at a low `--brightness`. A 60-pixel strip at full white pulls about 3.6 A,
which is more than most USB supplies will give you.

---

## Where next

- [`README.md`](README.md) — settings precedence, the full API, the module map.
- `ambviz --help`, and `--help` on any subcommand.
- `ambviz config` prints the effective settings as TOML, which is the quickest
  way to see what a flag actually changed.

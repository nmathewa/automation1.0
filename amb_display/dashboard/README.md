# amb_display · strip dashboard

A single static HTML file. It renders the LED strip, per-channel levels and
traffic statistics by polling the **ambviz** telemetry API — it holds no state of
its own and contains no simulation logic. Everything it shows comes from
`GET /api/state` and `GET /api/stream`.

## Use

Start the package's API first:

```bash
cd ../ambviz && .venv/bin/ambviz run --virtual --source synth
```

(`ambviz serve` also works, but publishes only the strip — see below.)

Then open `index.html` — double-click it, or serve the folder:

```bash
python3 -m http.server 9000     # then http://127.0.0.1:9000/
```

Both work: the API sends permissive CORS headers precisely so this file can live
anywhere.

## Pointing it somewhere else

The API URL is editable in the header and remembered in `localStorage`. It can
also be set per-link:

```
index.html?api=http://192.168.1.50:8080
index.html?theme=dark
```

Resolution order: `?api=` → saved value → the page's own origin (when served over
HTTP) → `http://127.0.0.1:8080`.

## What it shows

Panels appear only when the API publishes the provider they need, so the layout
matches whatever process you point at. A banner names anything missing.

From the **strip** provider (`serve`, or `run --virtual`):

- **Strip** — the LEDs on a dark plate, plus a wall-wash approximation of the
  light they throw. Toggle to index labels.
- **Channel level by pixel** — R/G/B across the strip, with a hover crosshair and
  a table view of exact values.
- **Traffic** — packets/s, pixel updates/s, throughput and coverage, with
  one-second sparklines, plus the sending peer and connection state.

From the **engine** provider (`run --api`, or `run --virtual`):

- **Mel filterbank** — band levels with their centre frequencies in Hz.
- **Controls** — effect, brightness, frequency range, band count and the eight
  smoothing alphas, changed live while audio plays. The server stays
  authoritative: each control POSTs a patch and redraws from the response, so a
  rejected value snaps back and the reason appears under the panel.
  **Download TOML** saves the current settings as a config file.

Follows the system light/dark preference; the toggle overrides and persists.

## Contract

The dashboard checks `contract` from `/api/health` against the version it was
built for and warns on a mismatch rather than misrendering. If you see that
banner, update whichever side is older.

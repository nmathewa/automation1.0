# ambviz strip controller (ESP8266)

Receives pixel data over UDP and drives a WS2812 strip. This is the firmware the
[`ambviz`](../../ambviz/) package talks to.

## First time

```bash
cp include/secrets.h.example include/secrets.h   # then fill in your WiFi
pio run -e nodemcuv2 -t upload
pio device monitor
```

The strip joins by DHCP and announces itself over mDNS, so there is no IP to
configure anywhere:

```bash
ambviz run --host ambviz.local --source mic
```

## Configuration

Nothing in `src/main.cpp` needs editing. Strip geometry and networking are build
flags in `platformio.ini`:

| Flag | Default | Meaning |
|---|---|---|
| `LED_COUNT` | 60 | must match ambviz `output.pixels` |
| `LED_PIN` | 3 | ESP8266 DMA output is fixed to GPIO3 (RX); the flag is for ESP32 |
| `UDP_PORT` | 7777 | must match ambviz `output.port` |
| `MDNS_NAME` | `ambviz` | the strip answers to `<name>.local` |

Override without editing the file:

```bash
PLATFORMIO_BUILD_FLAGS="-DLED_COUNT=108" pio run -e nodemcuv2 -t upload
```

WiFi credentials live in `include/secrets.h`, which is gitignored. **The
credentials that used to be hardcoded in this firmware are still in the
repository's git history — rotate that password.**

## Protocol

Flat 4-byte records, several per datagram:

```
| index | r | g | b |
```

Only changed pixels are sent, so state persists between datagrams. The index is
a single byte, so 255 pixels is the ceiling — `static_assert` enforces it at
compile time, and records naming a pixel past the end of the strip are dropped
and counted rather than wrapping into memory they do not own. The count is
reported once a second over serial alongside the packet rate.

## Environments

| Env | Builds |
|---|---|
| `nodemcuv2` | `src/main.cpp`, the UDP receiver — **this is the real firmware** |
| `d1_mini` | the same, for a Wemos D1 mini |
| `color_waves` | `examples/color_waves.cpp`, a standalone palette demo with no WiFi and no audio |

The demo used to sit in `src/` while the receiver sat in `test/`, which
PlatformIO excludes from builds — so a plain `pio run -t upload` flashed a lamp,
not a strip controller. That is why they swapped.

## Troubleshooting

- **Nothing on the strip, no serial output** — check the board actually matches
  the environment. A Nano flashed with a Pro Mini build looks exactly like this
  (see issue #4).
- **`ttyUSB` not found** — install the PlatformIO udev rules and add yourself to
  `dialout` (issue #5).
- **Pixels stick after a burst** — expected on packet loss for up to
  `output.full_refresh_interval` seconds; ambviz resends the whole strip on that
  interval.

#!/usr/bin/env python3
"""Generate a seamless music loop for driving the visualizer.

A sine sweep tells you nothing about how the visualizer behaves on music. This
writes something with the structure that actually matters -- kick on every beat,
an eighth-note bass line, a sustained chord bed, offbeat hats, and a lead that
enters every other bar so the spectrum visibly changes.

    python tools/make-test-loop.py            -> ambviz-loop.wav
    python tools/make-test-loop.py --bpm 140 --bars 16 --out /tmp/fast.wav

Play it into the virtual sink with tools/virtual-audio.sh play.
"""
from __future__ import annotations

import argparse
import wave

import numpy as np

# Am F C G -- a progression whose roots move enough to shift the low bands.
ROOTS = (110.00, 87.31, 130.81, 98.00)
LEAD = (4, 3, 5, 4)


def envelope(samples: int, rate: int, attack: float = 0.005, decay: float | None = None):
    e = np.ones(samples)
    a = max(1, int(attack * rate))
    e[:a] = np.linspace(0, 1, a)
    d = max(1, int((decay if decay is not None else samples / rate) * rate))
    e[-d:] *= np.linspace(1, 0, d) ** 2
    return e


def build(bpm: float, bars: int, rate: int, seed: int = 3) -> np.ndarray:
    beat = 60.0 / bpm
    total = int(rate * beat * 4 * bars)
    out = np.zeros(total)
    rng = np.random.default_rng(seed)

    def add(sig: np.ndarray, at: float) -> None:
        i = int(at * rate)
        if i >= total:
            return
        end = min(total, i + len(sig))
        out[i:end] += sig[: end - i]

    def tone(freq: float, dur: float, gain: float, attack: float = 0.005) -> np.ndarray:
        e = envelope(int(dur * rate), rate, attack)
        return np.sin(2 * np.pi * freq * np.arange(len(e)) / rate) * e * gain

    for bar in range(bars):
        origin = bar * 4 * beat
        root = ROOTS[bar % len(ROOTS)]

        for b in range(4):
            at = origin + b * beat
            # Kick: a fast downward pitch sweep, which is what puts energy in
            # the lowest bands on every beat.
            e = envelope(int(0.18 * rate), rate, 0.001)
            sweep = np.cumsum(np.linspace(120, 45, len(e))) / rate
            add(np.sin(2 * np.pi * sweep) * e * 0.9, at)

            for half in (0.0, 0.5):
                add(tone(root, 0.22, 0.35), at + (b + half) * beat)

            hat = envelope(int(0.05 * rate), rate, 0.001)
            add(rng.normal(0, 1, len(hat)) * hat * 0.12, at + (b + 0.5) * beat)

        pad = envelope(int(4 * beat * rate), rate, 0.08, 0.6)
        for mult in (2.0, 2.5, 3.0):
            add(np.sin(2 * np.pi * root * mult * np.arange(len(pad)) / rate) * pad * 0.10, origin)

        if bar % 2:
            for step, semitone in enumerate(LEAD):
                add(tone(root * 4 * 2 ** (semitone / 12), 0.4, 0.18, 0.01), origin + step * beat)

    return out / (np.abs(out).max() * 1.05)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bpm", type=float, default=120.0)
    p.add_argument("--bars", type=int, default=8)
    p.add_argument("--rate", type=int, default=44100)
    p.add_argument("--out", default="ambviz-loop.wav")
    args = p.parse_args()

    mono = build(args.bpm, args.bars, args.rate)
    stereo = np.stack([mono, mono * 0.97], axis=1)   # slight width, still mono-safe
    pcm = (stereo * 32767).astype(np.int16)

    with wave.open(args.out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(args.rate)
        w.writeframes(pcm.tobytes())

    print(f"{args.out}: {len(mono) / args.rate:.1f}s, {args.bpm:g} BPM, {args.bars} bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

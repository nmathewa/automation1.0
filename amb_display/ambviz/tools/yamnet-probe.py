#!/usr/bin/env python3
"""Watch what YAMNet hears, next to what the DSP features think.

A diagnostic, not part of the package. The point is to decide whether a
semantic classifier earns its place before anything is wired in -- and to do
that on real material, because synthetic material does not settle it. A tone
with tremolo has the statistics of speech without being speech, and YAMNet
correctly calls it a synthesizer while the hand-rolled features call it
dialogue.

    tools/yamnet-probe.py                       # follow the default output
    tools/yamnet-probe.py --device ambviz       # a specific monitor
    tools/yamnet-probe.py --api http://127.0.0.1:8080   # show DSP features too

Play a film through the speakers and read the two columns against each other.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

MODEL_URL = ("https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1"
             "?lite-format=tflite")
MODEL_PATH = Path.home() / ".cache" / "ambviz-models" / "yamnet.tflite"
RATE = 16000                     # YAMNet is fixed at 16 kHz mono


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching YAMNet to {MODEL_PATH} ...", file=sys.stderr)
    with urllib.request.urlopen(MODEL_URL, timeout=120) as r:
        MODEL_PATH.write_bytes(r.read())
    return MODEL_PATH


def default_monitor() -> str:
    sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                          text=True, check=True).stdout.strip()
    return sink + ".monitor"


def dsp_features(api: str) -> str:
    """The current hand-rolled verdict, for side-by-side comparison."""
    try:
        with urllib.request.urlopen(f"{api}/api/state", timeout=1.0) as r:
            e = json.load(r)["engine"]
        return (f"energy {e.get('energy', 0):.2f}  dialogue {e.get('dialogue', 0):.2f}  "
                f"spread {e.get('spread', 0):>5.0f}Hz")
    except Exception:
        return "(no ambviz API)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", help="monitor source; default follows the default output")
    p.add_argument("--api", default="http://127.0.0.1:8080", help="ambviz API for comparison")
    p.add_argument("--top", type=int, default=3, help="classes to show")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between predictions")
    args = p.parse_args()

    if shutil.which("parec") is None:
        print("parec not found; this needs PulseAudio or PipeWire", file=sys.stderr)
        return 1

    from ai_edge_litert.interpreter import Interpreter

    model = ensure_model()
    labels = zipfile.ZipFile(model).read("yamnet_label_list.txt").decode().splitlines()
    interp = Interpreter(model_path=str(model))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    window = int(inp["shape"][0])

    device = args.device or default_monitor()
    if "." not in device:                       # a bare name like "ambviz"
        device = f"{device}_virtual.monitor" if not device.endswith("monitor") else device
    print(f"listening to {device}\n", file=sys.stderr)

    proc = subprocess.Popen(
        ["parec", f"--device={device}", "--format=s16le", f"--rate={RATE}",
         "--channels=1", "--latency-msec=50", "--raw"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    hop = max(1, int(args.interval * RATE))
    buf = np.zeros(window, dtype=np.float32)
    try:
        while True:
            raw = proc.stdout.read(hop * 2)
            if not raw or len(raw) < hop * 2:
                break
            block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 2 ** 15
            buf = np.concatenate([buf[len(block):], block])[-window:]

            interp.set_tensor(inp["index"], buf)
            interp.invoke()
            scores = interp.get_tensor(out["index"])[0]
            top = np.argsort(scores)[::-1][:args.top]
            heard = ", ".join(f"{labels[i]} {scores[i]:.2f}" for i in top)
            print(f"{heard:<62} | {dsp_features(args.api)}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

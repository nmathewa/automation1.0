"""Command-line interface.

    ambviz run --virtual               # everything in one process: no mic, no hardware
    ambviz run --host 192.168.1.35     # drive real hardware
    ambviz run --api                   # drive hardware, publish engine telemetry
    ambviz serve                       # virtual strip + API only (no numpy needed)
    ambviz config                      # show the effective settings
    ambviz devices                     # list audio inputs

Settings precedence: defaults < --config file < AMBVIZ_* env < flags.

Heavy imports (numpy, scipy, pyaudio) happen inside the subcommand that needs
them, so ``ambviz serve`` runs on a bare Python.
"""

from __future__ import annotations

import argparse
import sys
import time

from ambviz import __version__
from ambviz.settings import Settings

def _effect_names() -> tuple[str, ...] | None:
    """Effect names for --help, or None where numpy is absent.

    argparse builds its choices at import time, but this module must stay
    importable without numpy so `serve` runs on a bare interpreter. When the
    import is unavailable the flag simply accepts anything and settings
    validation produces the error, listing the valid names.
    """
    try:
        from ambviz.effects import EFFECTS
    except ImportError:
        return None
    return tuple(sorted(EFFECTS))


EFFECT_NAMES = _effect_names()


# ── argument plumbing ────────────────────────────────────────────────────────
def _add_settings_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", metavar="PATH", help="TOML or JSON settings file")

    out = p.add_argument_group("output")
    out.add_argument("--device", choices=["udp", "none"], help="where pixels go")
    out.add_argument("--sim", action="store_true",
                     help="shorthand for --host 127.0.0.1, i.e. the virtual strip")
    out.add_argument("--host", help="strip address (ESP8266 IP, or 127.0.0.1)")
    out.add_argument("--port", type=int, help="strip UDP port")
    out.add_argument("--pixels", type=int, help="number of LEDs to drive")
    out.add_argument("--gamma", action="store_true", help="apply the software gamma table")
    out.add_argument("--brightness", type=float, metavar="0-1")

    aud = p.add_argument_group("audio")
    aud.add_argument("--source", choices=["mic", "loopback", "synth", "wav"],
                     help="'loopback' captures what the machine is playing; "
                          "'synth' generates a test signal and needs no audio at all")
    aud.add_argument("--wav", metavar="PATH", help="16-bit PCM .wav to loop")
    aud.add_argument("--input-device", metavar="NAME|N",
                     help="what to capture: a mic device index or name; or with "
                          "--source loopback, an output device or application name")
    aud.add_argument("--rate", type=int, help="sample rate in Hz")
    aud.add_argument("--fps", type=int, help="target frames per second")

    dsp = p.add_argument_group("dsp / effect")
    dsp.add_argument("--effect", choices=EFFECT_NAMES, metavar="NAME",
                     help="effect to run" + (f" ({', '.join(EFFECT_NAMES)})" if EFFECT_NAMES else ""))
    dsp.add_argument("--bins", type=int, help="number of Mel bands")
    dsp.add_argument("--min-freq", type=float, help="filterbank low edge, Hz")
    dsp.add_argument("--max-freq", type=float, help="filterbank high edge, Hz")
    dsp.add_argument("--no-mirror", action="store_true", help="do not mirror about the centre")


def _overrides(args: argparse.Namespace) -> dict:
    """Map flags onto the nested settings structure, skipping anything unset."""
    output, audio, dsp, effect = {}, {}, {}, {}

    if getattr(args, "sim", False):
        output.setdefault("host", "127.0.0.1")
    for flag, section, key in (
        ("device", output, "device"), ("host", output, "host"), ("port", output, "port"),
        ("pixels", output, "pixels"), ("input_device", audio, "input_device"),
        ("rate", audio, "rate"), ("fps", audio, "fps"), ("source", audio, "source"),
        ("bins", dsp, "fft_bins"), ("min_freq", dsp, "min_frequency"),
        ("max_freq", dsp, "max_frequency"), ("effect", effect, "name"),
        ("brightness", effect, "brightness"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            section[key] = value

    if getattr(args, "gamma", False):
        output["gamma_correction"] = True
    if getattr(args, "wav", None):
        audio["wav_path"] = args.wav
        audio.setdefault("source", "wav")
    if getattr(args, "no_mirror", False):
        effect["mirror"] = False

    return {k: v for k, v in
            {"output": output, "audio": audio, "dsp": dsp, "effect": effect}.items() if v}


def _load(args: argparse.Namespace) -> Settings:
    return Settings.load(args.config, overrides=_overrides(args))


# ── subcommands ──────────────────────────────────────────────────────────────
def cmd_run(args: argparse.Namespace) -> int:
    """Audio in, UDP packets out -- optionally publishing telemetry as it goes."""
    from ambviz.outputs import make_output
    from ambviz.pipeline import Visualizer
    from ambviz.sources import make_source

    # --virtual is the whole no-hardware story: aim at localhost, receive it
    # here, and publish both halves on one port.
    if args.virtual:
        args.api = True
        args.sim = True

    settings = _load(args)
    for warning in settings.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    visualizer = Visualizer(settings)
    try:
        source = make_source(settings)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: cannot open audio source: {exc}", file=sys.stderr)
        return 1

    output = make_output(settings)
    print(f"{settings.audio.source} -> {settings.effect.name} -> {output}", file=sys.stderr)

    api, commands, receiver, sampler, strip = None, None, None, None, None
    if args.api:
        from ambviz.api import ApiServer
        from ambviz.control import CommandQueue

        commands = CommandQueue(settings)
        providers = {"engine": visualizer.snapshot}
        if args.virtual:
            from ambviz.strip import HistorySampler, UdpReceiver, VirtualStrip

            strip = VirtualStrip(settings.output.pixels, grow=True)
            receiver = UdpReceiver(strip, "127.0.0.1", settings.output.port)
            receiver.start()
            sampler = HistorySampler(strip)
            sampler.start()
            providers["strip"] = strip.snapshot
        api = ApiServer(providers, host=args.api_host, port=args.api_port,
                        stream_fps=args.stream_fps, settings=settings,
                        commands=commands, static_dir=args.static)
        api.start()
        print(f"api {api.role} -> {api.url}/api/state", file=sys.stderr)
        if api.static_dir:
            print(f"dashboard -> {api.url}/", file=sys.stderr)

    frames = 0
    started = last_report = time.monotonic()
    try:
        with source, output:
            for samples in source.frames():
                # Drain control commands here so every mutation happens on this
                # thread, never concurrently with process().
                if commands is not None:
                    for patch in commands.drain():
                        visualizer.apply(patch)
                output.send(visualizer.process(samples))
                frames += 1
                now = time.monotonic()
                if not args.quiet and now - last_report >= 1.0:
                    print(f"\r{frames / (now - started):5.1f} fps  "
                          f"vol {visualizer.volume:7.4f}  "
                          f"{'silent' if visualizer.silent else 'active'}   ",
                          end="", file=sys.stderr, flush=True)
                    last_report = now
                if args.duration and now - started >= args.duration:
                    break
            output.blank()
    except KeyboardInterrupt:
        output.blank()
        output.close()
    finally:
        for service in (receiver, sampler, api):
            if service is not None:
                service.stop()
        elapsed = time.monotonic() - started
        if not args.quiet:
            print(f"\n{frames} frames in {elapsed:.1f}s "
                  f"({frames / max(elapsed, 1e-9):.1f} fps), "
                  f"{output.packets} packets / {output.bytes} bytes", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Virtual strip + telemetry API. No audio, no numpy."""
    from ambviz.api import ApiServer
    from ambviz.strip import HistorySampler, UdpReceiver, VirtualStrip

    settings = _load(args)
    strip = VirtualStrip(settings.output.pixels, grow=not args.fixed)
    receiver = UdpReceiver(strip, args.udp_host, settings.output.port)
    receiver.start()
    sampler = HistorySampler(strip)
    sampler.start()

    api = ApiServer(
        {"strip": strip.snapshot},
        host=args.api_host,
        port=args.api_port,
        stream_fps=args.stream_fps,
        settings=settings,
        static_dir=args.static,
    )
    print(f"virtual strip : {strip.pixels} px ({'fixed' if args.fixed else 'auto-grow'})")
    print(f"listening on  : udp://{args.udp_host}:{settings.output.port}")
    print(f"api           : {api.url}/api/state")
    if api.static_dir:
        print(f"dashboard     : {api.url}/  (from {api.static_dir})")
    print("Ctrl-C to stop")
    try:
        api.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        receiver.stop()
        sampler.stop()
        api.stop()
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    settings = _load(args)
    for warning in settings.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(settings.to_toml(), end="")
    return 0


def cmd_monitors(args: argparse.Namespace) -> int:
    """List what loopback capture can tap."""
    from ambviz.sources import monitor_sources, resolve_monitor

    try:
        monitors = monitor_sources()
        default = resolve_monitor(None)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not monitors:
        print("no playback sources found")
    for name in monitors:
        print(f"  {name}{'  <- default output' if name == default else ''}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    from ambviz.sources import list_input_devices

    try:
        devices = list_input_devices()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not devices:
        print("no audio input devices found")
    for index, name, channels, is_default in devices:
        mark = "  <- default" if is_default else ""
        print(f"  [{index:2d}] {name}  ({channels} ch){mark}")
    return 0


# ── entry point ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ambviz",
        description="Audio-reactive LED strip driver (WS2812 over UDP).",
        epilog="Settings precedence: defaults < --config file < AMBVIZ_* env < flags.",
    )
    parser.add_argument("--version", action="version", version=f"ambviz {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("monitors", help="list capturable playback sources").set_defaults(
        func=cmd_monitors)

    run = subs.add_parser("run", help="drive a strip from audio")
    _add_settings_flags(run)
    run.add_argument("--api", action="store_true",
                     help="publish engine telemetry and accept live setting changes")
    run.add_argument("--virtual", action="store_true",
                     help="implies --api and --sim, and receives the strip data here too "
                          "-- the whole no-hardware setup in one process")
    run.add_argument("--static", metavar="DIR", help="serve a dashboard directory from the API, so the page and the API share an origin; --api-host 0.0.0.0 to reach it from another device")
    run.add_argument("--api-host", default="127.0.0.1", help="API bind address")
    run.add_argument("--api-port", type=int, default=8080, help="API port")
    run.add_argument("--stream-fps", type=float, default=30.0, help="cap on SSE pushes/second")
    run.add_argument("--duration", type=float, metavar="SEC", help="stop after this long")
    run.add_argument("--quiet", action="store_true", help="do not print the frame rate")
    run.set_defaults(func=cmd_run)

    serve = subs.add_parser("serve", help="virtual strip + telemetry API for a dashboard")
    _add_settings_flags(serve)
    serve.add_argument("--static", metavar="DIR", help="serve a dashboard directory from the API, so the page and the API share an origin; --api-host 0.0.0.0 to reach it from another device")
    serve.add_argument("--api-host", default="127.0.0.1", help="API bind address")
    serve.add_argument("--api-port", type=int, default=8080, help="API port")
    serve.add_argument("--udp-host", default="0.0.0.0", help="address to receive strip data on")
    serve.add_argument("--stream-fps", type=float, default=30.0, help="cap on SSE pushes/second")
    serve.add_argument("--fixed", action="store_true",
                       help="reject indices past --pixels instead of growing the strip")
    serve.set_defaults(func=cmd_serve)

    config = subs.add_parser("config", help="print the effective settings as TOML")
    _add_settings_flags(config)
    config.set_defaults(func=cmd_config)

    devices = subs.add_parser("devices", help="list audio input devices")
    devices.set_defaults(func=cmd_devices)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

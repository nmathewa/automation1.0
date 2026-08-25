"""Audio sources.

``mic`` captures from a real input device through PortAudio. Two bindings are
supported: :mod:`sounddevice` is preferred because it binds the PortAudio
*runtime* and therefore installs from a wheel, while :mod:`pyaudio` has to
compile against ``portaudio.h`` and fails on any machine without the dev
package. Either works; whichever is importable is used.

``loopback`` captures what the machine is *playing* rather than what a
microphone hears -- clean stereo straight from the mixer, with no room noise and
no dependence on hardware at all. It reads a PulseAudio/PipeWire monitor source
through ``parec``.

``synth`` and ``wav`` need neither, which is what makes the whole pipeline
testable with no hardware and no microphone attached.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import wave
from typing import Iterator

import numpy as np

from ambviz.settings import Settings


class Source:
    """Yields frames of ``samples_per_frame`` int16-scaled samples."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.frame_size = settings.audio.samples_per_frame

    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _SoundDeviceMic(Source):
    """Capture via :mod:`sounddevice`."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        import sounddevice as sd

        self._sd = sd
        self._stream = sd.InputStream(
            samplerate=settings.audio.rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_size,
            device=resolve_input_device(settings.audio.input_device),
        )
        self._stream.start()
        self.overflows = 0

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            block, overflowed = self._stream.read(self.frame_size)
            if overflowed:
                self.overflows += 1
            # Drop any backlog so the visualization tracks live audio instead
            # of falling further behind it.
            if (available := self._stream.read_available) > self.frame_size:
                self._stream.read(available)
            yield block.reshape(-1).astype(np.float32)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class _PyAudioMic(Source):
    """Capture via :mod:`pyaudio`."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=settings.audio.rate,
            input=True,
            input_device_index=resolve_input_device(settings.audio.input_device),
            frames_per_buffer=self.frame_size,
        )
        self.overflows = 0

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            try:
                raw = self._stream.read(self.frame_size, exception_on_overflow=False)
                if (available := self._stream.get_read_available()) > self.frame_size:
                    self._stream.read(available, exception_on_overflow=False)
                yield np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            except IOError:
                self.overflows += 1

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()


# sounddevice first: it needs no system dev package, so it works where pyaudio
# cannot even be installed.
MIC_BACKENDS: tuple[tuple[str, type[Source]], ...] = (
    ("sounddevice", _SoundDeviceMic),
    ("pyaudio", _PyAudioMic),
)


def available_mic_backend() -> tuple[str, type[Source]] | None:
    """The first importable capture binding, or ``None`` if there is none."""
    for name, cls in MIC_BACKENDS:
        try:
            __import__(name)
            return name, cls
        except ImportError:
            continue
    return None


def MicSource(settings: Settings) -> Source:  # noqa: N802 - used as a class would be
    """Open the microphone with whichever PortAudio binding is installed."""
    backend = available_mic_backend()
    if backend is None:
        raise ImportError(
            "no audio capture backend. Install one with:\n"
            "    pip install 'ambviz[mic]'        (sounddevice, needs no system packages)\n"
            "Or use --source synth to run without a microphone."
        )
    return backend[1](settings)


def _pactl(*args: str) -> str:
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True,
                              check=True, timeout=5).stdout
    except FileNotFoundError:
        raise RuntimeError(
            "pactl not found. Loopback capture needs PulseAudio or PipeWire; "
            "use --source mic for a hardware input instead."
        ) from None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"pactl {' '.join(args)} failed: {exc}") from None


def monitor_sources() -> list[str]:
    """Every ``.monitor`` source, one per output device."""
    return [line.split()[1] for line in _pactl("list", "short", "sources").splitlines()
            if len(line.split()) > 1 and line.split()[1].endswith(".monitor")]


def resolve_monitor(target: str | int | None = None) -> str:
    """Pick the monitor source to capture.

    ``None`` follows the default output, which is what "visualize whatever is
    playing" means. A string matches a monitor, a sink, or the name of a running
    application -- in which case the sink that application is playing to is used.
    """
    monitors = monitor_sources()
    if not monitors:
        raise RuntimeError("no monitor sources; is a sound server running?")

    if target is None or target == "":
        return _pactl("get-default-sink").strip() + ".monitor"

    wanted = str(target).strip().lower()

    for name in monitors:                                    # a monitor by name
        if wanted in name.lower():
            return name

    for line in _pactl("list", "short", "sinks").splitlines():  # a sink by name
        parts = line.split()
        if len(parts) > 1 and wanted in parts[1].lower():
            return parts[1] + ".monitor"

    sink = _sink_for_application(wanted)                      # an app by name
    if sink:
        return sink + ".monitor"

    raise ValueError(
        f"nothing matching {target!r} to capture. Monitors available:\n  "
        + "\n  ".join(monitors)
    )


def _sink_for_application(wanted: str) -> str | None:
    """The sink a named application is playing to, if it is playing."""
    inputs = _pactl("list", "sink-inputs")
    sink_index, matched = None, False
    for line in inputs.splitlines():
        stripped = line.strip()
        if stripped.startswith("Sink Input #"):
            sink_index, matched = None, False
        elif stripped.startswith("Sink:"):
            sink_index = stripped.split(":", 1)[1].strip()
        elif "application.name" in stripped or "media.name" in stripped:
            if wanted in stripped.lower():
                matched = True
        if matched and sink_index is not None:
            for line2 in _pactl("list", "short", "sinks").splitlines():
                parts = line2.split()
                if parts and parts[0] == sink_index:
                    return parts[1]
    return None


class LoopbackSource(Source):
    """Capture the machine's own playback from a monitor source.

    Independent of any microphone: it taps the mixer, so it hears exactly what
    is being played, in stereo, with none of the room noise, hum or automatic
    gain a microphone introduces.
    """

    # How often to re-check the default output, in frames. Plugging in
    # headphones usually swaps the default sink for a different one entirely,
    # and the old monitor then yields silence forever.
    DEFAULT_CHECK_FRAMES = 60

    def __init__(self, settings: Settings, target: str | int | None = None):
        super().__init__(settings)
        if shutil.which("parec") is None:
            raise RuntimeError(
                "parec not found. Loopback capture needs PulseAudio or PipeWire "
                "(Debian/Ubuntu: apt install pulseaudio-utils). "
                "Use --source mic for a hardware input instead."
            )
        wanted = target if target is not None else settings.audio.input_device
        # With no explicit target the intent is "whatever is playing", so track
        # the default output as it changes rather than pinning to one device.
        self.follow_default = wanted is None or wanted == ""
        self.device = resolve_monitor(wanted)
        self.reconnects = 0
        self._proc = self._spawn()

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "parec",
                f"--device={self.device}",
                "--format=s16le",
                f"--rate={self.settings.audio.rate}",
                "--channels=1",
                # Keep the server-side buffer short so the lights track the
                # audio rather than trailing it.
                "--latency-msec=20",
                "--raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _follow(self) -> bool:
        """Re-point at the default output if it has changed. True if it moved."""
        try:
            current = resolve_monitor(None)
        except (RuntimeError, ValueError):
            return False
        if current == self.device:
            return False
        self.device = current
        self.reconnects += 1
        self._stop_proc()
        self._proc = self._spawn()
        return True

    def frames(self) -> Iterator[np.ndarray]:
        want = self.frame_size * 2          # int16
        stream = self._proc.stdout
        assert stream is not None
        # A monitor of an idle sink emits silence far faster than real time, so
        # an unpaced loop spins at thousands of frames a second and makes the
        # reported frame rate meaningless. Audio is real time by definition;
        # capping at the nominal rate costs nothing when data is genuinely
        # flowing and stops the spin when it is not.
        interval = self.frame_size / self.settings.audio.rate
        next_due = time.monotonic()
        seen = 0
        while True:
            if self.follow_default:
                seen += 1
                if seen >= self.DEFAULT_CHECK_FRAMES:
                    seen = 0
                    if self._follow():
                        stream = self._proc.stdout
                        assert stream is not None
                        next_due = time.monotonic()

            raw = stream.read(want)
            if not raw or len(raw) < want:
                if self._proc.poll() is not None:
                    if self.follow_default and self._follow():
                        stream = self._proc.stdout
                        assert stream is not None
                        continue
                    raise RuntimeError(
                        f"parec stopped while reading {self.device!r}; "
                        f"the source may have gone away"
                    )
                continue
            yield np.frombuffer(raw, dtype=np.int16).astype(np.float32)

            next_due += interval
            if (delay := next_due - time.monotonic()) > 0:
                time.sleep(delay)
            else:
                next_due = time.monotonic()

    def _stop_proc(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def close(self) -> None:
        self._stop_proc()


class SynthSource(Source):
    """Generated test signal: a four-on-the-floor kick, a bass note, a sweeping
    mid tone and hats. Deterministic, paced to real time.

    This exists so the visualizer and the simulator can be exercised end to end
    with no microphone and no LED hardware.
    """

    def __init__(self, settings: Settings, bpm: float | None = None, amplitude: float | None = None):
        super().__init__(settings)
        self.bpm = settings.audio.synth_bpm if bpm is None else bpm
        self.amplitude = settings.audio.synth_amplitude if amplitude is None else amplitude
        self._t = 0.0
        self._rng = np.random.default_rng(12345)

    def frames(self) -> Iterator[np.ndarray]:
        rate = self.settings.audio.rate
        dt = self.frame_size / rate
        period = 60.0 / self.bpm
        next_due = time.monotonic()
        while True:
            t = self._t + np.arange(self.frame_size) / rate
            beat_phase = (t % period) / period

            kick = np.sin(2 * np.pi * 55 * t) * np.exp(-12 * beat_phase)
            bass = 0.4 * np.sin(2 * np.pi * 110 * t)
            sweep_hz = 400 + 1600 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 9.0))
            mid = 0.3 * np.sin(2 * np.pi * sweep_hz * t)
            hat = 0.15 * self._rng.normal(0, 1, self.frame_size) * (beat_phase > 0.5)

            self._t += dt
            yield ((kick + bass + mid + hat) * self.amplitude).astype(np.float32)

            next_due += dt
            if (delay := next_due - time.monotonic()) > 0:
                time.sleep(delay)
            else:
                next_due = time.monotonic()


class WavSource(Source):
    """Loops a .wav file in real time using the stdlib ``wave`` module."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._wav = wave.open(settings.audio.wav_path, "rb")
        if self._wav.getsampwidth() != 2:
            raise ValueError(
                f"{settings.audio.wav_path}: need 16-bit PCM, "
                f"got {self._wav.getsampwidth() * 8}-bit"
            )
        self.channels = self._wav.getnchannels()
        self.file_rate = self._wav.getframerate()
        if self.file_rate != settings.audio.rate:
            raise ValueError(
                f"{settings.audio.wav_path} is {self.file_rate} Hz but audio.rate "
                f"is {settings.audio.rate} Hz; set audio.rate to match"
            )

    def frames(self) -> Iterator[np.ndarray]:
        dt = self.frame_size / self.settings.audio.rate
        next_due = time.monotonic()
        while True:
            raw = self._wav.readframes(self.frame_size)
            if len(raw) < self.frame_size * 2 * self.channels:
                self._wav.rewind()
                raw = self._wav.readframes(self.frame_size)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if self.channels > 1:
                samples = samples.reshape(-1, self.channels).mean(axis=1)
            yield samples

            next_due += dt
            if (delay := next_due - time.monotonic()) > 0:
                time.sleep(delay)
            else:
                next_due = time.monotonic()

    def close(self) -> None:
        self._wav.close()


def make_source(settings: Settings) -> Source:
    sources = {"mic": MicSource, "loopback": LoopbackSource,
               "synth": SynthSource, "wav": WavSource}
    try:
        return sources[settings.audio.source](settings)
    except KeyError:
        raise ValueError(
            f"unknown audio.source {settings.audio.source!r}; "
            f"expected one of {sorted(sources)}"
        ) from None


def resolve_input_device(device: int | str | None) -> int | None:
    """Turn a device name into an index, leaving indices and ``None`` alone."""
    if device is None or isinstance(device, int):
        return device
    wanted = device.strip().lower()
    if not wanted:
        return None
    matches = [(i, name) for i, name, _, _ in list_input_devices() if wanted in name.lower()]
    if not matches:
        available = ", ".join(name for _, name, _, _ in list_input_devices()) or "none"
        raise ValueError(f"no input device matching {device!r}; available: {available}")
    return matches[0][0]


def list_input_devices() -> list[tuple[int, str, int, bool]]:
    """``(index, name, channels, is_default)`` for every PortAudio input.

    Indices are backend-specific, so they mean what the active backend says they
    mean -- which is the one :func:`MicSource` will use.
    """
    backend = available_mic_backend()
    if backend is None:
        raise ImportError("no audio capture backend installed")
    name = backend[0]

    if name == "sounddevice":
        import sounddevice as sd

        default = sd.default.device[0]
        return [
            (i, str(d["name"]), int(d["max_input_channels"]), i == default)
            for i, d in enumerate(sd.query_devices())
            if int(d["max_input_channels"]) > 0
        ]

    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        default = pa.get_default_input_device_info().get("index", -1)
        return [
            (i, str(info["name"]), int(info["maxInputChannels"]), i == default)
            for i in range(pa.get_device_count())
            if int((info := pa.get_device_info_by_index(i))["maxInputChannels"]) > 0
        ]
    finally:
        pa.terminate()

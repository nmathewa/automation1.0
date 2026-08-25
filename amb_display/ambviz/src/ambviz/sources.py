"""Audio sources.

``mic`` needs PyAudio and a working input device (install the ``mic`` extra).
``synth`` and ``wav`` need neither, which is what makes the whole pipeline
testable with no hardware and no microphone attached.
"""

from __future__ import annotations

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


class MicSource(Source):
    """Live PortAudio capture."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        import pyaudio  # imported lazily so synth/wav work without PortAudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=settings.audio.rate,
            input=True,
            input_device_index=settings.audio.input_device,
            frames_per_buffer=self.frame_size,
        )
        self.overflows = 0

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            try:
                raw = self._stream.read(self.frame_size, exception_on_overflow=False)
                # Drop any backlog so the visualization tracks live audio
                # instead of falling further behind.
                if (available := self._stream.get_read_available()) > self.frame_size:
                    self._stream.read(available, exception_on_overflow=False)
                yield np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            except IOError:
                self.overflows += 1

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()


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
    sources = {"mic": MicSource, "synth": SynthSource, "wav": WavSource}
    try:
        return sources[settings.audio.source](settings)
    except KeyError:
        raise ValueError(
            f"unknown audio.source {settings.audio.source!r}; "
            f"expected one of {sorted(sources)}"
        ) from None


def list_input_devices() -> list[tuple[int, str, int]]:
    """``(index, name, max_input_channels)`` for every PortAudio input device."""
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info["maxInputChannels"]) > 0:
                devices.append((i, str(info["name"]), int(info["maxInputChannels"])))
        return devices
    finally:
        pa.terminate()

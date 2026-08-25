"""Runtime settings.

Every tunable in the package, resolved at *run* time rather than at import time,
so one process can drive any strip on any host without a duplicated module tree.

Precedence, lowest to highest::

    dataclass defaults  <  TOML/JSON file  <  environment  <  CLI overrides

Environment variables are named ``AMBVIZ_<SECTION>_<KEY>``, e.g.
``AMBVIZ_OUTPUT_HOST=192.168.1.35``.

Standard library only -- :mod:`ambviz.strip` and :mod:`ambviz.api` depend on this
module and must stay importable without numpy.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CONTRACT = 2
"""Version of the HTTP telemetry contract (see :mod:`ambviz.api`).

Bumped whenever the shape of an API response changes incompatibly, so a
dashboard built against an older package can say so instead of misrendering.

1. flat ``/api/state`` carrying strip fields at the top level
2. ``/api/state`` namespaced per provider; ``/api/settings`` reports its source
"""

# (alpha_decay, alpha_rise) for each ExpFilter in the pipeline. Small = smoother.
Alpha = tuple[float, float]


@dataclass
class Output:
    """Where pixels go."""

    device: str = "udp"
    """``udp`` (real ESP8266 or the simulator), or ``none`` (headless / benchmarking)."""

    host: str = "127.0.0.1"
    """Target address. Point at the ESP for hardware, at localhost for the simulator."""

    port: int = 7777
    """UDP port. Must match ``localPort`` in the ESP firmware."""

    pixels: int = 60
    """LEDs actually being driven. May be fewer than the strip's physical length."""

    max_pixels_per_packet: int = 126
    """Pixels per UDP datagram. 126 x 4 bytes = 504, inside the firmware's 1024 buffer."""

    gamma_correction: bool = False
    """Apply the software gamma table. Leave off if the firmware already does it."""

    gamma_table: str = "gamma_table.npy"
    """Gamma LUT path; relative paths resolve against this file's directory."""

    full_refresh_interval: float = 2.0
    """Seconds between forced full-strip resends. The wire protocol is diff-only,
    so a dropped packet leaves a pixel stale until it next changes; this bounds
    how long that can last. 0 disables."""

    def gamma_table_path(self) -> Path:
        """Resolve the gamma table, falling back to the one shipped in the package."""
        p = Path(self.gamma_table).expanduser()
        if p.is_absolute():
            return p
        local = Path.cwd() / p
        return local if local.exists() else DATA / p


@dataclass
class Audio:
    """Where samples come from."""

    source: str = "mic"
    """Where audio comes from:

    ``mic``       a hardware input, via PortAudio
    ``loopback``  what the machine is playing, tapped from the mixer -- no
                  microphone involved, and far cleaner than one
    ``synth``     a generated test signal, needing no hardware at all
    ``wav``       a .wav file on a loop
    """

    input_device: int | str | None = None
    """What to capture from.

    With ``source = "mic"``: a PortAudio device index, or part of a device name.
    With ``source = "loopback"``: part of a monitor, an output device, or the
    name of a running application -- empty follows the default output.

    Names are matched case-insensitively, which survives the reordering that
    indices are prone to -- ``"pulse"`` keeps working, ``13`` may not.
    ``None`` uses the system default. See ``ambviz devices``."""

    wav_path: str = ""
    """Source file when ``source = "wav"``."""

    rate: int = 44100
    """Sample rate in Hz."""

    fps: int = 60
    """Target visualization frame rate. One audio buffer is read per frame,
    so this also sets the buffer size (``rate / fps`` samples)."""

    rolling_history: int = 2
    """Audio frames kept in the FFT window. More = better frequency resolution,
    more latency."""

    min_volume: float = 1e-7
    """Peak amplitude below which the strip is blanked."""

    synth_bpm: float = 120.0
    """Tempo of the generated test signal, when ``source = "synth"``."""

    synth_amplitude: float = 8000.0
    """Peak amplitude of the generated test signal, in int16 units."""

    @property
    def samples_per_frame(self) -> int:
        return int(self.rate / self.fps)


@dataclass
class Dsp:
    """Time domain to frequency domain."""

    min_frequency: float = 200.0
    """Low edge of the Mel filterbank, Hz."""

    max_frequency: float = 12000.0
    """High edge of the Mel filterbank, Hz. Must stay under the Nyquist limit."""

    fft_bins: int = 24
    """Mel bands. More bands = finer frequency detail, coarser amplitude detail.
    No point exceeding ``output.pixels``."""

    mel_exponent: float = 2.0
    """Power applied to the filterbank output before gain control. Higher values
    exaggerate peaks and suppress quiet bands."""

    gain_sigma: float = 1.0
    """Gaussian blur sigma used when finding the peak for automatic gain control."""

    onset_sensitivity: float = 1.4
    """How far spectral flux must exceed its running average to count as an onset.

    Swept against a 120 BPM loop whose true onset grid is one every 0.250 s:
    1.2 caught 42 in 12 s, 1.4 caught 38, 1.8 caught 28. The median gap stayed
    at 0.250 s throughout, so the detector locks regardless and this only sets
    how many it catches. 1.4 keeps headroom before noise starts triggering it."""

    onset_refractory: float = 0.12
    """Minimum seconds between beats. 0.12 caps at 500 BPM."""

    vocal_suppression: float = 0.9
    """How far to cancel centre-panned content, 0-1. Needs a stereo source.

    Applies to **colour only**. L - R removes everything centred, which in a real
    mix is the kick, the snare and often the lead as well as the voice -- so
    suppressing the whole analysis removes the song, not the singer. Level,
    onsets and energy always come from the full mix; only the hue is derived from
    the suppressed spectrum. That is why the default can be this aggressive
    without the strip going quiet.

    The blend is linear in amplitude, so 0.5 really does leave half the voice
    behind. Values below about 0.8 do very little.

    Vocals sit in the centre of almost every mix, so ``L - R`` removes them
    without a model. 1.0 replaces the band entirely with the side channel;
    0.7-0.8 usually reads better, leaving a trace so the result does not sound
    -- or look -- hollow."""

    vocal_band: tuple[float, float] = (180.0, 5000.0)
    """Where suppression applies, in Hz.

    Restricting it matters: the kick and bass are centre-panned too, so
    cancelling everywhere would remove exactly what drives the low bands.
    Outside this range the mid channel is used untouched."""


@dataclass
class Mood:
    """The slow layer: what the light does over a scene rather than a beat.

    Analysis-level and rig-wide, like ``[dsp]`` -- one mood feeding every node,
    for the same reason one analysis pass does.
    """

    response_seconds: float = 8.0
    """Roughly how long a full colour traverse takes. The subtlety knob."""

    attack: float = 0.12
    """Seconds for brightness to reach a new peak. Short: a drum hit should land."""

    release: float = 2.0
    """Seconds for brightness to fall back. Long, so it breathes rather than
    flickers. The gap between this and attack is what makes movement read as
    dynamic rather than merely fast."""

    accent: float = 0.18
    """How much a hit brightens the strip, 0-1.

    Deliberately small. Punctuation, not rhythm -- when a scene genuinely wants
    rhythm the director switches to an effect built for it, rather than making
    the wash beat. An earlier default of 0.5 made every scene throb."""

    hue_rate: float = 0.125
    """Maximum hue movement per second, in turns. 0.125 crosses the circle in
    8 s. Capping the *rate* is what makes it read as calm rather than merely
    smoothed."""

    deadband: float = 0.01
    """Hue changes smaller than this produce no movement at all, so the light
    holds instead of shimmering. Hyperion calls the equivalent hysteresis."""

    floor: float = 0.06
    """Minimum brightness, 0-1. A strip snapping fully dark during a quiet line
    is more distracting than one that drifts -- Hyperion's backlight threshold."""

    dialogue_damping: float = 0.8
    """How far speech suppresses the fast layer, 0-1. Film dialogue is centred,
    so it is detectable from the mid-to-side ratio without recognising anything."""

    detail: float = 1.0
    """Ceiling on the spectral layer, before scene energy and dialogue scale it.

    1.0 lets a loud, wideband scene reach a fully spectral display; the scene
    itself decides how close it gets. Lower it to cap how energetic the effect is
    ever allowed to become."""

    audio_weight: float = 1.0
    """Weight of the audio-derived mood. Exists so a picture feed can be blended
    in later without reworking the effect -- see mood.py."""

    range_seconds: float = 45.0
    """Window the adaptive range learns over. Long enough to span a scene."""

    scene_weight: float = 0.7
    """How far YAMNet's opinion overrides the hand-rolled features, 0-1.

    The DSP features describe how audio behaves; the model describes what it is.
    The model is the better judge of "is this speech", but it works on ~1 s
    windows, so it is blended rather than trusted outright. 0 ignores it."""

    switch_dwell: float = 8.0
    """Minimum seconds on one animation before another may take over.

    Without it the selector flaps at every threshold crossing, which looks far
    worse than one mediocre animation held steady."""

    switch_margin: float = 0.15
    """How much better a candidate must score than the current animation.

    Hysteresis: ties keep what is already on screen."""

    crossfade: float = 1.2
    """Seconds to fade between animations. Effects carry internal state, so
    swapping instantly shows a visible discontinuity; fading hides it."""

    scene_interval: float = 0.5
    """Seconds between classifications. The model costs under a millisecond, so
    this is about how fast a scene label should move, not about CPU."""


@dataclass
class Effect:
    """Frequency domain to pixels."""

    name: str = "spectrum"
    """Which effect to run. ``ambviz effects`` lists them."""

    mirror: bool = True
    """Mirror the pattern about the centre of the strip."""

    scroll_decay: float = 0.98
    """Per-frame brightness decay for the scroll effect."""

    scroll_sigma: float = 0.2
    """Blur applied to the scroll trail."""

    energy_scale: float = 0.9
    """Exponent applied to band energy before it is mapped to a bar length."""

    energy_sigma: float = 4.0
    """Blur applied to the energy effect's edges."""

    brightness: float = 1.0
    """Global multiplier applied to every channel, 0.0-1.0."""


@dataclass
class Smoothing:
    """``(alpha_decay, alpha_rise)`` for each exponential filter in the chain.

    These were previously scattered as literals through ``visualization.py``.
    """

    red: Alpha = (0.20, 0.99)
    green: Alpha = (0.05, 0.30)
    blue: Alpha = (0.10, 0.50)
    common_mode: Alpha = (0.99, 0.01)
    """Tracks the spectral floor. Subtracting it is what gives the red channel its bite."""
    pixel: Alpha = (0.10, 0.99)
    gain: Alpha = (0.001, 0.99)
    mel_gain: Alpha = (0.01, 0.99)
    mel_smoothing: Alpha = (0.50, 0.99)


@dataclass
class Settings:
    output: Output = field(default_factory=Output)
    audio: Audio = field(default_factory=Audio)
    dsp: Dsp = field(default_factory=Dsp)
    effect: Effect = field(default_factory=Effect)
    smoothing: Smoothing = field(default_factory=Smoothing)
    mood: Mood = field(default_factory=Mood)
    display_fps: bool = True

    def __post_init__(self) -> None:
        # Populated by validate(); not a field, so it stays out of to_dict()/to_toml().
        self.warnings: list[str] = []

    # ── loading ──────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str] | None = None,
        overrides: dict[str, Any] | None = None,
        env: bool = True,
    ) -> "Settings":
        """Build settings from file, environment and explicit overrides."""
        s = cls()
        if path:
            s._apply(_read_file(Path(path)))
        if env:
            s._apply(_read_env())
        if overrides:
            s._apply(overrides)
        s.validate()
        return s

    def _apply(self, data: dict[str, Any]) -> None:
        for section, values in data.items():
            if not isinstance(values, dict):
                if not hasattr(self, section):
                    raise KeyError(f"unknown setting: {section!r}")
                setattr(self, section, values)
                continue
            target = getattr(self, section, None)
            if not is_dataclass(target):
                raise KeyError(f"unknown settings section: [{section}]")
            known = {f.name for f in fields(target)}
            for key, value in values.items():
                if key not in known:
                    raise KeyError(f"unknown setting: {section}.{key}")
                current = getattr(target, key)
                if isinstance(current, tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(target, key, value)

    # ── validation ───────────────────────────────────────────────────────────
    def validate(self) -> None:
        """Fail loudly on combinations that would silently misbehave."""
        problems, warnings = [], []

        # TOML has no null: an empty string means "unset" for optional values.
        if self.audio.input_device == "":
            self.audio.input_device = None
        if self.audio.input_device is not None and not isinstance(self.audio.input_device, (int, str)):
            problems.append("audio.input_device must be a device index, a name, or empty")

        if self.output.pixels < 1:
            problems.append("output.pixels must be >= 1")
        if self.output.device not in ("udp", "none"):
            problems.append(f"output.device must be 'udp' or 'none', got {self.output.device!r}")
        if self.audio.source not in ("mic", "loopback", "synth", "wav"):
            problems.append(
                f"audio.source must be 'mic', 'loopback', 'synth' or 'wav', "
                f"got {self.audio.source!r}"
            )
        if self.audio.source == "wav" and not self.audio.wav_path:
            problems.append("audio.source is 'wav' but audio.wav_path is empty")
        if self.audio.synth_bpm <= 0:
            problems.append("audio.synth_bpm must be positive")
        if not 0 < self.audio.synth_amplitude <= 32767:
            problems.append("audio.synth_amplitude must be between 0 and 32767")
        # Imported lazily: settings.py must stay importable without numpy.
        from ambviz.effects import EFFECTS  # noqa: PLC0415

        if self.effect.name not in EFFECTS:
            problems.append(
                f"unknown effect.name {self.effect.name!r}; "
                f"expected one of {sorted(EFFECTS)}"
            )
        if not 0.0 <= self.effect.brightness <= 1.0:
            problems.append("effect.brightness must be between 0.0 and 1.0")

        nyquist = self.audio.rate / 2
        if self.dsp.max_frequency > nyquist:
            problems.append(
                f"dsp.max_frequency ({self.dsp.max_frequency:.0f} Hz) exceeds the Nyquist "
                f"limit for audio.rate {self.audio.rate} Hz ({nyquist:.0f} Hz)"
            )
        if self.dsp.min_frequency >= self.dsp.max_frequency:
            problems.append("dsp.min_frequency must be below dsp.max_frequency")

        m = self.mood
        if m.response_seconds <= 0:
            problems.append("mood.response_seconds must be positive")
        if m.hue_rate <= 0:
            problems.append("mood.hue_rate must be positive")
        if m.attack <= 0 or m.release <= 0:
            problems.append("mood.attack and mood.release must be positive")
        if m.attack > m.release:
            warnings.append(
                "mood.attack is longer than mood.release, so the strip fades "
                "faster than it lights; that is usually the wrong way round"
            )
        for name in ("deadband", "floor", "dialogue_damping", "detail",
                     "audio_weight", "accent"):
            if not 0.0 <= getattr(m, name) <= 1.0:
                problems.append(f"mood.{name} must be between 0.0 and 1.0")
        if m.switch_dwell < 0 or m.crossfade <= 0:
            problems.append("mood.switch_dwell must not be negative and crossfade must be positive")
        if not 0.0 <= m.switch_margin <= 1.0:
            problems.append("mood.switch_margin must be between 0.0 and 1.0")
        if not 0.0 <= m.scene_weight <= 1.0:
            problems.append("mood.scene_weight must be between 0.0 and 1.0")
        if m.scene_interval <= 0:
            problems.append("mood.scene_interval must be positive")
        if m.range_seconds < 1.0:
            problems.append("mood.range_seconds must be at least 1 second")

        if not 0.0 <= self.dsp.vocal_suppression <= 1.0:
            problems.append("dsp.vocal_suppression must be between 0.0 and 1.0")
        low, high = self.dsp.vocal_band
        if low >= high:
            problems.append("dsp.vocal_band must be (low, high) with low below high")
        if low < 0:
            problems.append("dsp.vocal_band lower edge must not be negative")

        if self.dsp.onset_sensitivity <= 1.0:
            problems.append("dsp.onset_sensitivity must be greater than 1.0")
        if self.dsp.onset_refractory < 0:
            problems.append("dsp.onset_refractory must not be negative")

        for name, alpha in vars(self.smoothing).items():
            lo, hi = alpha
            if not (0.0 < lo < 1.0 and 0.0 < hi < 1.0):
                problems.append(f"smoothing.{name} values must be strictly between 0 and 1")

        # WS2812 needs ~30us per pixel plus a ~50us reset gap.
        max_fps = int(((self.output.pixels * 30e-6) + 50e-6) ** -1.0)
        if self.audio.fps > max_fps:
            problems.append(
                f"audio.fps {self.audio.fps} exceeds what {self.output.pixels} WS2812 "
                f"pixels can accept ({max_fps} fps)"
            )
        if self.dsp.fft_bins > self.output.pixels:
            warnings.append(
                f"dsp.fft_bins ({self.dsp.fft_bins}) exceeds output.pixels "
                f"({self.output.pixels}); the extra bands cannot be resolved"
            )
        if self.output.pixels % 2 and self.effect.mirror:
            warnings.append(
                f"output.pixels ({self.output.pixels}) is odd; the centre pixel of a "
                f"mirrored effect is shared between both halves"
            )
        if self.output.pixels > 128:
            warnings.append(
                f"output.pixels ({self.output.pixels}) exceeds 128; the ESP firmware "
                f"reads pixel indices into a signed char and will wrap"
            )

        self.warnings = warnings
        if problems:
            raise ValueError("invalid settings:\n  - " + "\n  - ".join(problems))

    # ── serialisation ────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        def unpack(obj: Any) -> Any:
            if is_dataclass(obj):
                return {f.name: unpack(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        return unpack(self)

    def to_toml(self) -> str:
        """Render the effective settings as TOML, for ``run.py --dump-config``."""
        lines: list[str] = []
        for section, values in self.to_dict().items():
            if not isinstance(values, dict):
                lines.append(f"{section} = {_toml_value(values)}")
        for section, values in self.to_dict().items():
            if isinstance(values, dict):
                lines.append(f"\n[{section}]")
                lines += [f"{k} = {_toml_value(v)}" for k, v in values.items()]
        return "\n".join(lines) + "\n"


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if v is None:
        return '""'
    return repr(v)


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _read_env(prefix: str = "AMBVIZ_") -> dict[str, Any]:
    """Read ``AMBVIZ_SECTION_KEY`` variables into a nested dict."""
    sections = {f.name for f in fields(Settings) if is_dataclass(getattr(Settings(), f.name))}
    out: dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        body = raw_key[len(prefix):].lower()
        section = next((s for s in sections if body.startswith(s + "_")), None)
        if section is None:
            continue
        key = body[len(section) + 1:]
        out.setdefault(section, {})[key] = _coerce(raw_value)
    return out


def _coerce(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value

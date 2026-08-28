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

    segments: tuple[int, ...] = ()
    """Pixel counts of each run in a multi-sided rig, in wiring order.

    Empty means one strip of ``pixels``, which is what every earlier config
    describes and stays the default. ``(30, 60, 30)`` is a three-sided room:
    left wall, front wall, right wall, wired as one chain.

    ``pixels`` must equal the sum, and is filled in automatically when it does
    not -- a rig described twice, once as a total and once as its parts, is a
    rig whose two descriptions will eventually disagree.

    The **widest** segment is the front and carries the animation exactly as a
    single strip would. Every other side is a *wall wash*: the light spilling
    off the nearest end of the front, heavily blurred. Copying the animation
    onto the sides instead makes three competing focal points; a wash makes one
    picture with the room lit around it."""

    side_mode: str = "stereo"
    """What the side walls show: ``"stereo"`` or ``"wash"``.

    ``stereo`` is the point of having two of them. Each wall is driven by its
    own channel, so a hard-panned guitar lights one side and not the other, and
    coloured by the *second* most prominent stem -- the thing the mix contains
    that the front wall is not already telling you about. Together they read as
    the stereo image standing in the room rather than as decoration.

    ``wash`` is the simpler fallback: blurred light off the nearest end of the
    front. Used automatically when the source is mono or no separator is
    running, since neither channel nor stem information exists then."""

    side_animation: str = "freqwave"
    """Which animation the side walls run.

    Fixed, and deliberately not the one the front is running. Cloning the
    front's animation onto three surfaces gives a room with three competing
    focal points and no centre; the front should be what you look *at* and the
    sides what you see around it. A quiet, structureless effect is the right
    shape for that -- position must stop meaning frequency the moment you leave
    the wall you are facing.

    ``freqwave`` is the default because it is structureless in the right way:
    one hue across the whole run, so position never means frequency, but that
    hue follows the channel's own pitch and its brightness follows the
    channel's own level. A pure noise field is quieter still and throws the
    per-channel colour away, which is most of what the sides have to say.

    Set to ``""`` to follow the front instead."""

    side_brightness: float = 1.0
    """How bright the sides run relative to the front, 0-1.

    At 1.0 the walls sit at the same level as the centre, which is what a rig
    of one continuous strip actually looks like. Lower it if the centre should
    lead on brightness as well as on detail."""

    track_gap: float = 0.35
    """Seconds of silence taken to mean a new track has started.

    Everything about the uplift is judged against how loud this song has been,
    and playback does not stop between songs -- so without this a quiet track
    followed by a loud one produces an enormous surge in its first bar, from a
    floor belonging to a different piece of music entirely.

    A third of a second is longer than any musical rest and shorter than the
    pause between tracks in almost any player. **It cannot see a boundary that
    has no gap**: gapless albums and crossfaded playlists run on as one piece,
    which is wrong but harmless -- the floor still adapts within a few seconds,
    it just does not get a fresh warm-up. 0 disables the reset."""

    uplift_warmup: float = 12.0
    """Seconds of music before any uplift may fire.

    Measured from the start of the *current track*, not from startup, so every
    song gets its own baseline.

    The surge is measured against how quiet the song has recently been, and at
    the start there is no "recently" -- the floor has heard one passage and has
    no idea whether it was loud or soft for this track. Waiting gives it a
    baseline to judge against, which matters because the moments worth catching
    almost never arrive in the first few seconds anyway: a song builds first."""

    uplift_surge_db: float = 11.0
    """How far the level must jump above its recent floor to lift the room, dB.

    The most reliable thing about a drop is what comes *before* it. Songs build,
    strip back to almost nothing, and then everything arrives at once -- so a
    large rise measured against how quiet it just was finds the moment far more
    dependably than any measure of how much is playing, which is high through
    a whole loud chorus and says nothing about where the chorus began.

    11 dB is roughly a threefold jump in level. Lower to catch gentler lifts;
    raise it to hold out for the big ones. 0 disables the criterion."""

    uplift_floor_recovery: float = 4.0
    """How fast the quiet reference climbs back, dB per second.

    The floor drops to any new low at once and recovers slowly, so it remembers
    the quiet the song just passed through. Recovering quickly would let a
    breakdown be forgotten before the drop lands on it -- which is precisely
    the pair of moments this exists to connect."""

    unison_full_trigger: float = 0.88
    """Fullness that lifts the room on its own, 0-1.

    The second way in. Most uplifts are a rise out of quiet and are found by
    ``uplift_surge_db``, but some passages simply are enormous -- a final
    chorus with every part playing, arrived at gradually rather than dropped
    into -- and there is no surge to find because nothing ever got quiet.

    Set well above ``unison_threshold`` on purpose. The two are not
    alternatives at the same level: at 0.68 fullness alone put the room in
    unison for 79% of an ordinary verse, because it is high through anything
    loud and says nothing about where that passage began. This is the bar for
    "there is nothing left to add". 1.0 disables it and leaves only the surge."""

    unison_threshold: float = 0.55
    """How full the music must be for an uplift to count, 0-1.

    A gate, not a trigger. Fullness is high through an entire chorus and says
    nothing about where the chorus began, so on its own it put the room in
    unison for 79% of an ordinary verse. What fires an uplift is the level
    surge; this only stops one being claimed by a sudden loud noise in an
    otherwise empty passage.

    At a drop, a chorus, a passage with everything playing at once, the sides
    stop having their own job and mirror the front exactly. Splitting the room
    into a focus and its surroundings is right for most of a song and wrong at
    its biggest moment -- that is when it should read as one surface, and the
    walls agreeing with the centre is what makes it lift.

    Fullness is energy, onset density and -- when a separator is running -- how
    many instruments are actually present at once, which is the part a spectrum
    cannot see. 1.0 disables it."""

    unison_beats: float = 4.0
    """How long the room stays in unison, in beats of the music.

    An uplift is a moment, not a state. Held for as long as the music stays
    loud it stops being an event and becomes the new normal -- and then the
    room has no focus for the rest of the chorus. Four beats is about a bar:
    long enough to land, short enough that the room returns and the *next* one
    can land too.

    It re-arms only once the music has dropped back below the threshold, so a
    long loud section lifts once at its start rather than pulsing throughout."""

    unison_response: float = 1.0
    """Seconds for the room to ease *out* of unison.

    It swells in about four times faster. Snapping between two arrangements of
    the room on a single loud bar reads as a fault, and a slow swell reads as
    the music arriving -- but a symmetric response slower than the hold it has
    to fill can never reach full at all, which is a subtler way of the effect
    never being seen."""

    accent_animation: str = "scroll"
    """Which animation bursts across a wall when an event lands there.

    Brightness alone says "something happened" and nothing more. A travelling
    animation says what *kind* of something, and reads as an event crossing the
    room rather than a lamp being turned up. Any of the fast effects suit it --
    ``scroll``, ``pixelwave``, ``puddles``, ``waterfall``.

    It is paced to cross the wall once per ``accent_decay``, whatever the wall's
    length and whatever the library's own travel speed is: a burst that has not
    arrived by the time it fades has not happened. Set to ``""`` to go back to
    a plain brightness lift."""

    accent_strength: float = 1.0
    """How completely an event takes over the wall it lands on, 0-1.

    The walls carry the width of the mix, which is a slow and quiet signal --
    true to the audio and, on its own, not much to look at. This is the other
    half: when something happens in the song, one wall answers.

    It is a crossfade, not a lift. There is no headroom to add into -- the
    strip already runs at full brightness, so an additive accent only clips and
    a lit wall cannot get brighter. What makes the event visible instead is
    that the walls' ambience is held at ``side_brightness`` of the front,
    leaving the top of the range free, and that what fades in is a *travelling*
    animation rather than a level.

    0 disables it and leaves the walls purely ambient."""

    accent_alternate: bool = True
    """Send consecutive events to opposite walls.

    A hit on both walls at once is just a brighter room. Alternating makes the
    music appear to move across the space -- which is the effect a pair of side
    walls exists to produce, and the one thing they can do that the front
    cannot. Set False to light both together."""

    accent_beats: float = 2.0
    """Minimum gap between accents, in beats of the music itself.

    A gap fixed in seconds is right at one tempo and wrong at every other. It
    worked on fast material -- where a beat happens to be about as long as the
    fade -- and fired far too often on anything slower, because the onset
    detector still finds the subdivisions between the beats and each one took a
    turn.

    Measured from the actual spacing between onsets, so 2.0 means "at most one
    accent every other beat" and stays true whether that is 0.3 s or 1.2 s.
    Two beats is close enough together that the alternation between the walls
    reads as a pattern rather than as isolated flashes."""

    accent_stem_change: float = 0.0
    """How far the instrument balance must move to count as an event, 0-1.

    Off by default. The idea is sound -- an arrangement change is what a
    listener would call "the music changed", where a spectral onset fires on
    every hi-hat -- but in practice the separator misses events and reports the
    ones it catches about a second late, because it works on a one-second
    window and is a second behind by construction. Judged against onsets, which
    are exact and immediate, it read as both incomplete and out of time.

    The mechanism is kept because the model may be run faster or a causal one
    substituted later; raise this above 0 to switch it back on."""

    accent_threshold: float = 0.35
    """How strong an onset must be to move the accent, 0-1.

    Every hi-hat would otherwise take a turn and the alternation would blur
    into a flicker. Only events that stand out get a wall."""

    accent_length_beats: float = 1.0
    """How long an accent lasts, in beats of the music.

    Drives the fade *and* how fast the burst crosses the wall, so both follow
    the tempo. This is what was missing: with the length fixed in seconds the
    burst was tuned for one tempo and wrong at every other -- it looked right
    on fast material, where a beat happens to be about as long as the fade, and
    on slower music the same burst crawled and lost its connection to the beat
    entirely.

    At 1.0 an accent occupies exactly one beat, so it has finished by the time
    the next one is due."""

    accent_decay: float = 0.15
    """Shortest an accent may last, in seconds.

    A floor under ``accent_length_beats`` only, so that very fast material does
    not reduce the burst to a flicker too brief to see."""

    accent_attack: float = 0.12
    """Seconds for an accent to reach full.

    Not zero. Jumping straight to full brightness reads as a strobe however
    slowly it then fades -- the eye takes the edge, not the envelope. A short
    rise turns the same event into something that arrives."""

    stereo_emphasis: float = 0.7
    """How much of each side shows what is *unique* to its channel, 0-1.

    At 0 a wall is driven by everything its speaker plays. That sounds like the
    right answer and looks like the wrong one: in a normal mix both channels
    carry nearly the whole arrangement, so both walls end up showing what the
    front is already showing. Measured on real material, the two walls came out
    96% identical even with the channels correctly separated.

    Raising it subtracts the other channel band by band, so a wall shows only
    what its own side has that the other does not -- centred material drops out
    of both walls and lives on the front where it belongs, while a hard-panned
    guitar lights one wall alone. That contrast is the stereo image; the shared
    part of the mix was never carrying any."""

    stereo_threshold: float = 0.10
    """How differently the channels must behave before the sides split, 0-1.

    Measured as each band's level imbalance, weighted by that band's own
    energy, so it answers "how much of this mix is actually panned" rather than
    "do the channels differ anywhere at all".

    Real material sits lower than intuition suggests. A track with obvious
    stereo width measured **0.01**: its channels were decorrelated -- reverb,
    spread, phase -- but carried the same spectrum at the same level in every
    band. Magnitude cannot see phase, so there was nothing for a wall to show
    that the front was not already showing, and the honest answer is to fall
    through to the stem instead. Hard-panned arrangements clear this easily;
    most modern masters will not, and should not."""

    stereo_threshold: float = 0.18
    """How differently the channels must behave before the sides split, 0-1.

    Below this the two walls would be showing the same thing twice, which is
    worse than showing one thing well. Most studio mixes sit low; a wide or
    hard-panned arrangement clears it easily."""

    wash_span: float = 0.35
    """Fraction of the front wall that feeds each side's wash, 0-1.

    0.35 means each side takes its colour from the outer third of the front,
    which is roughly what a wall that distance away would actually catch.
    Larger pulls colour from further across the front and makes the two sides
    resemble each other; smaller ties each side tightly to its own corner."""

    wash_softness: float = 0.6
    """How hard the wash is blurred, as a fraction of the span, 0-1.

    The point of a wash is that it carries colour and level without structure
    -- position must stop meaning frequency the moment you leave the front
    wall. At 0 the sides show a recognisable squashed copy of the animation,
    which is the thing this exists to avoid."""

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

    hpss_frames: int = 9
    """Trailing frames the harmonic median looks back over.

    At 60 fps nine frames is 150 ms, so a partial must hold its bin for about
    that long to count as sustained. Longer is a stricter test of "held" and
    costs a little more; the window is forced odd, because an even one has no
    single middle element."""

    hpss_kernel: int = 17
    """FFT bins the percussive median spans.

    17 bins is about 366 Hz at the default 2048-point FFT, wide enough that a
    single harmonic is erased by its neighbours but narrow enough that a real
    broadband hit survives."""

    percussive_smoothing: float = 3.0
    """Seconds of smoothing on the percussive fraction.

    The raw ratio is near-binary per frame -- measured 0.99 on a hit and 0.0
    between -- because HPSS describes one frame, not a passage. Smoothed, it
    becomes percussive *density*, which is what "how rhythmic is this music"
    actually means: against a synthetic one-hit-in-four signal the smoothed
    value settles at 0.248."""

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

    animations: tuple[str, ...] = ("bars", "energy", "spectrum", "freqwave", "puddles")
    """Which animations "auto" may choose between, in no particular order.

    A setting rather than a fixed list so the shortlist can change without a
    code edit. Names must exist in the effect registry; unknown ones are
    rejected at load rather than silently ignored.

    ``waterfall`` was dropped first. Over 43 s of real audio it had the
    joint-highest mean score (0.519) *and* a p95 frame-to-frame jitter of
    0.323 -- more than twice ``switch_margin``, so hysteresis could not filter
    it and single noisy frames won switches that dwell then held for eight
    seconds. That was the scroll-waterfall-scroll-waterfall oscillation.

    ``pacifica`` replaced it, was dropped for the same kind of reason -- its
    layers drift once every 18-42 s under a 2.8 s level filter, so it read as a
    still image rather than as a swell -- and has since been removed from the
    library altogether. Its one good idea, a wash built from several sine
    layers at different scales, now lives in ``cinema`` where it is driven by
    the audio rather than by the clock.

    ``scroll`` went with it. It is scored 0.55 on ``onset_rate``, and that
    feature is passed through an ``AdaptiveRange`` that stretches a
    near-constant input to fill 0-1 -- measured at 0.92 on a phase-continuous
    held chord whose raw rate was 0.51. Scroll therefore wins on sustained
    material it does not suit.

    ``freqwave`` and ``puddles`` take their places, covering the pitch and
    sparse families that nothing else in the shortlist covers."""

    score_smoothing: float = 2.0
    """Seconds of smoothing applied to each candidate's score before comparing.

    Without this the selector is a coin flip. Scores are recomputed per frame
    from features that jitter, and measured over 43 s of real audio no
    candidate ever held the lead for as long as one second -- longest unbroken
    spell as top scorer was 0.9 s for the winner and 0.0-0.4 s for the rest,
    against an eight second dwell. So whichever candidate happened to be ahead
    at the instant the dwell expired won, which is why animations switched
    without the audio changing and why some never appeared at all.

    Smoothing is what makes ``switch_margin`` and ``switch_dwell`` mean
    something: they can only arbitrate between candidates whose ordering is
    stable for longer than a frame.

    The honest cost, measured on the same clip: with the noise gone the
    ordering mostly stops changing, so switches drop from 5 to 1 and the strip
    settles on the genuine winner for long stretches. The pre-smoothing
    variety was churn, not responsiveness -- candidates were being picked at
    random moments, which looked lively and meant nothing. If the settled
    behaviour is too static the knob to reach for is this one, downward."""

    change_threshold: float = 0.25
    """How far the audio's character must move before a switch is considered.

    Distance between the current smoothed feature vector (energy, onset rate,
    brightness, dialogue) and the one captured at the last switch. On the
    43 s reference clip 0.25 fires about every 11 seconds; a film's dialogue
    scene holds one anchor for minutes.

    This replaced hysteresis-on-scores as the thing that decides *whether* to
    switch. Hand-written suitability scores turned out to be undependable at
    that job: all candidates sit within ~0.2 of each other while individual
    scores move more than that with the material, so every weighting produced
    some starved animation and a winner locked to the song. Scores now only
    rank what comes next -- they never re-elect the incumbent."""

    change_hold: float = 1.5
    """Seconds the drift must stay past ``change_threshold`` before switching.

    Without it a single noisy frame is a scene change. Drift is built from
    smoothed features, but smoothed is not still, and a momentary excursion
    across the threshold committed the strip to a new animation for the whole
    dwell -- switches that looked unprovoked because they *were*, the audio
    having gone nowhere.

    Requiring the excursion to persist costs a little latency on a real change
    and rejects essentially every spurious one, because noise crosses briefly
    and a scene change stays crossed."""

    max_dwell: float = 45.0
    """Switch anyway after this many seconds, however static the audio.

    The rotation guarantee, and deliberately long. It was load-bearing when the
    selector picked by rank, because a candidate that never scored second could
    only ever reach the strip when this timer fired. Recency-ordered selection
    distributes screen time on its own now, so this is a backstop rather than
    the mechanism -- measured over 170 s of real audio, eleven of twelve
    candidates ran for 3-11% of the time each.

    Set it short and it becomes the mechanism again: the strip rotates on the
    clock whatever the music does, which is the "switches for no reason"
    complaint in a different costume."""

    switch_dwell: float = 8.0
    """Minimum seconds on one animation before another may take over.

    Without it the selector flaps at every threshold crossing, which looks far
    worse than one mediocre animation held steady."""

    switch_margin: float = 0.15
    """How wide a band below the leading candidate still counts as suitable.

    Everything inside the band is a defensible next animation, so the one shown
    least recently wins and the band is what buys variety. Widen it to rotate
    through more of the shortlist, narrow it to always take the best-scoring
    option.

    It has to work this way because ranking alone cannot distribute screen
    time. Measured over 75 s of real audio under the old rule -- pick the
    best-scoring candidate other than the incumbent -- ``puddles`` led almost
    every frame, so every switch away from it came straight back to it:
    puddles 46% of the time, ``bars`` 10%, ``energy`` 0%. A candidate that is
    never rank two is unreachable however long it runs, which is the starvation
    the rotation guarantee was supposed to prevent and did not.

    (Before that it meant something else entirely -- how much better a candidate
    had to score before it could take over -- and was left unread when the rule
    changed to switch on audio change rather than on score crossings.)"""

    crossfade: float = 1.2
    """Seconds to fade between animations. Effects carry internal state, so
    swapping instantly shows a visible discontinuity; fading hides it."""

    stem_weight: float = 0.0
    """How far the Demucs stem balance replaces the DSP estimate, 0-1.

    Off by default: it needs ``torch``, a GPU to be comfortable, and a few
    hundred MB of weights, none of which the visualizer should require. At 1.0
    the stem balance is trusted outright where it is available.

    What it improves is the *scoring* terms, which is where the classifier
    keeps failing -- measured against ground truth over 33 s of real music with
    2 s smoothing, drums correlate 0.98 and vocals 0.96, against YAMNet groups
    that read 0.000 through the same material. ``bass`` correlates 0.65 and is
    deliberately not consumed anywhere."""

    stem_window: float = 1.0
    """Seconds of audio each separation looks at."""

    stem_interval: float = 0.5
    """Seconds between separations. 19.3 ms of GPU each on a 4050, so this is
    about how fast an instrument balance should move, not about compute."""

    stem_smoothing: float = 2.0
    """Seconds of smoothing on the stem shares.

    The published balance is roughly a second stale by construction, and the
    unsmoothed number is close to useless at that lag (0.62 correlation). Two
    seconds of smoothing takes it to 0.85, four to 0.89."""

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

    speed: float = 1.0
    """Global multiplier on how fast every animation moves.

    One knob for the whole library, because "too fast" was never about a single
    effect. Most of them advanced their state once per *frame* -- a scroll
    shifted one pixel, fire ran a cooling pass, a wave stepped along -- which
    is not a speed but the frame rate in disguise, and it left the library
    running at whatever 60 fps happened to produce. Motion is now measured in
    seconds and this scales it.

    Below 1.0 is calmer, above is busier. It changes rate only: animation phase
    is integrated rather than derived from the clock, so turning this down mid
    scene slows the motion without jumping it."""

    travel_pixels_per_second: float = 24.0
    """How fast ``scroll`` and ``pixelwave`` move light along the strip.

    Both used to shift exactly one pixel per frame, which is not a speed at all
    -- it is the frame rate wearing a costume. At 60 fps on a 60-pixel strip
    that crossed the whole strip every second, far quicker than any musical
    event, so the motion read as a blur that happened to be near the music
    rather than with it. Expressed per second instead, the look no longer
    changes when fps or strip length does.

    24 px/s crosses a 60-pixel strip in about 2.1 s at typical onset density --
    roughly a bar at 120 BPM. It was 16 for a while, which measured 3.16x slower
    than the per-frame original and read as sluggish once the rest of the
    library stopped racing. Sub-pixel movement is carried between frames rather
    than rounded away, so slow speeds still travel smoothly."""

    travel_beat_response: float = 0.8
    """How much the travel speed follows the music, 0-1.

    At 0 the speed is constant. Higher makes busy passages move faster and
    sustained ones drift, which is what "in time with it" actually asks for --
    a fixed speed cannot be in time with anything."""

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
        if self.output.side_mode not in ("stereo", "wash"):
            problems.append(
                f"output.side_mode must be 'stereo' or 'wash', "
                f"got {self.output.side_mode!r}")
        if self.output.side_animation:
            from ambviz.effects import EFFECTS  # noqa: PLC0415

            if self.output.side_animation not in EFFECTS:
                problems.append(
                    f"unknown output.side_animation {self.output.side_animation!r}; "
                    f"expected \"\" or one of {sorted(n for n in EFFECTS if n != 'auto')}")
            elif self.output.side_animation == "auto":
                problems.append("output.side_animation must not be 'auto'")
        if not 0.0 < self.output.side_brightness <= 1.0:
            problems.append("output.side_brightness must be above 0 and at most 1.0")
        if self.output.track_gap < 0:
            problems.append("output.track_gap must not be negative")
        if self.output.uplift_warmup < 0:
            problems.append("output.uplift_warmup must not be negative")
        if self.output.uplift_surge_db < 0:
            problems.append("output.uplift_surge_db must not be negative")
        if self.output.uplift_floor_recovery <= 0:
            problems.append("output.uplift_floor_recovery must be positive")
        if not 0.0 <= self.output.unison_full_trigger <= 1.0:
            problems.append("output.unison_full_trigger must be between 0.0 and 1.0")
        if self.output.unison_full_trigger < self.output.unison_threshold:
            warnings.append(
                f"output.unison_full_trigger ({self.output.unison_full_trigger}) is "
                f"below unison_threshold ({self.output.unison_threshold}), so fullness "
                f"alone lifts the room wherever the gate would have allowed it"
            )
        if not 0.0 <= self.output.unison_threshold <= 1.0:
            problems.append("output.unison_threshold must be between 0.0 and 1.0")
        if self.output.unison_beats <= 0:
            problems.append("output.unison_beats must be positive")
        if self.output.unison_response <= 0.0:
            problems.append("output.unison_response must be positive")
        if self.output.accent_animation:
            from ambviz.effects import EFFECTS  # noqa: PLC0415

            if self.output.accent_animation not in EFFECTS:
                problems.append(
                    f"unknown output.accent_animation "
                    f"{self.output.accent_animation!r}; expected \"\" or one of "
                    f"{sorted(n for n in EFFECTS if n != 'auto')}")
            elif self.output.accent_animation == "auto":
                problems.append("output.accent_animation must not be 'auto'")
        if not 0.0 <= self.output.accent_strength <= 1.0:
            problems.append("output.accent_strength must be between 0.0 and 1.0")
        if self.output.accent_beats < 0:
            problems.append("output.accent_beats must not be negative")
        if not 0.0 <= self.output.accent_stem_change <= 1.0:
            problems.append("output.accent_stem_change must be between 0.0 and 1.0")
        if not 0.0 <= self.output.accent_threshold <= 1.0:
            problems.append("output.accent_threshold must be between 0.0 and 1.0")
        if self.output.accent_length_beats <= 0:
            problems.append("output.accent_length_beats must be positive")
        if self.output.accent_attack < 0.0:
            problems.append("output.accent_attack must not be negative")
        if self.output.accent_decay <= 0.0:
            problems.append("output.accent_decay must be positive")
        if not 0.0 <= self.output.stereo_emphasis <= 1.0:
            problems.append("output.stereo_emphasis must be between 0.0 and 1.0")
        if not 0.0 <= self.output.stereo_threshold <= 1.0:
            problems.append("output.stereo_threshold must be between 0.0 and 1.0")
        if not 0.0 < self.output.wash_span <= 1.0:
            problems.append("output.wash_span must be above 0 and at most 1.0")
        if not 0.0 <= self.output.wash_softness <= 1.0:
            problems.append("output.wash_softness must be between 0.0 and 1.0")
        if self.output.segments:
            if any(s < 1 for s in self.output.segments):
                problems.append("every output.segments entry must be >= 1")
            elif sum(self.output.segments) != self.output.pixels:
                # Derived rather than rejected: the segments are the physical
                # description and the total is a consequence of it.
                self.output.pixels = sum(self.output.segments)
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
        if self.effect.speed <= 0:
            problems.append("effect.speed must be positive")
        if self.effect.travel_pixels_per_second <= 0:
            problems.append("effect.travel_pixels_per_second must be positive")
        if not 0.0 <= self.effect.travel_beat_response <= 1.0:
            problems.append("effect.travel_beat_response must be between 0.0 and 1.0")

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
        from ambviz.effects import EFFECTS  # noqa: PLC0415 - keeps this numpy-free until needed

        if not m.animations:
            problems.append("mood.animations must list at least one animation")
        unknown = [n for n in m.animations if n not in EFFECTS]
        if unknown:
            problems.append(
                f"mood.animations names unknown effect(s) {unknown}; "
                f"expected from {sorted(n for n in EFFECTS if n != 'auto')}"
            )
        if "auto" in m.animations:
            problems.append("mood.animations must not contain 'auto'")
        if m.switch_dwell < 0 or m.crossfade <= 0:
            problems.append("mood.switch_dwell must not be negative and crossfade must be positive")
        if not 0.0 <= m.switch_margin <= 1.0:
            problems.append("mood.switch_margin must be between 0.0 and 1.0")
        if self.dsp.hpss_frames < 1 or self.dsp.hpss_kernel < 1:
            problems.append("dsp.hpss_frames and dsp.hpss_kernel must be at least 1")
        if self.dsp.percussive_smoothing <= 0.0:
            problems.append("dsp.percussive_smoothing must be positive")
        if m.change_hold < 0.0:
            problems.append("mood.change_hold must not be negative")
        if m.change_threshold <= 0.0:
            problems.append("mood.change_threshold must be positive")
        if m.max_dwell < m.switch_dwell:
            problems.append("mood.max_dwell must be at least mood.switch_dwell")
        if not 0.0 <= m.stem_weight <= 1.0:
            problems.append("mood.stem_weight must be between 0.0 and 1.0")
        if m.stem_window <= 0 or m.stem_interval <= 0 or m.stem_smoothing <= 0:
            problems.append("mood.stem_window, stem_interval and stem_smoothing "
                            "must be positive")
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

#!/usr/bin/env bash
#
# A virtual audio device for driving ambviz without making noise.
#
# Creates a null sink whose monitor behaves exactly like a microphone: anything
# played into the sink is captured by ambviz, at full quality and with no room
# noise. Useful for developing the visualizer, and the same mechanism lets you
# visualize whatever the machine is actually playing.
#
#   tools/virtual-audio.sh up            create the sink
#   tools/virtual-audio.sh play FILE     loop a .wav into it
#   tools/virtual-audio.sh capture       print the ambviz command to run
#   tools/virtual-audio.sh speakers      capture what the speakers are playing
#   tools/virtual-audio.sh down          remove the sink
#
# Needs PulseAudio or PipeWire (pactl). No root.
set -euo pipefail

SINK="${AMBVIZ_SINK:-ambviz_virtual}"

need_pactl() {
  command -v pactl >/dev/null 2>&1 || {
    echo "pactl not found -- this needs PulseAudio or PipeWire." >&2
    exit 1
  }
}

sink_exists() { pactl list short sinks 2>/dev/null | grep -qx "[0-9]*\s*$SINK\s.*"; }

case "${1:-}" in
  up)
    need_pactl
    if pactl list short sinks | awk '{print $2}' | grep -qx "$SINK"; then
      echo "sink '$SINK' already exists"
    else
      pactl load-module module-null-sink \
        sink_name="$SINK" \
        sink_properties=device.description=ambviz-virtual >/dev/null
      echo "created sink '$SINK' (monitor: $SINK.monitor)"
    fi
    ;;

  play)
    need_pactl
    file="${2:-}"
    [ -f "$file" ] || { echo "usage: $0 play FILE.wav" >&2; exit 1; }
    echo "looping $file into '$SINK' -- Ctrl-C to stop"
    while true; do paplay --device="$SINK" "$file"; done
    ;;

  capture)
    cat <<EOF
Capture the virtual sink:

    PULSE_SOURCE=$SINK.monitor \\
      ambviz run --virtual --source mic --input-device pulse --static ../dashboard

Then open http://127.0.0.1:8080/
EOF
    ;;

  speakers)
    need_pactl
    default_sink="$(pactl get-default-sink)"
    cat <<EOF
Visualize whatever is playing through the speakers:

    PULSE_SOURCE=$default_sink.monitor \\
      ambviz run --virtual --source mic --input-device pulse --static ../dashboard

This is loopback capture: clean stereo straight from the mixer, with none of the
room noise a microphone picks up.
EOF
    ;;

  down)
    need_pactl
    ids="$(pactl list short modules | awk -v s="sink_name=$SINK" '$0 ~ s {print $1}')"
    if [ -z "$ids" ]; then
      echo "sink '$SINK' is not loaded"
    else
      for id in $ids; do pactl unload-module "$id"; done
      echo "removed sink '$SINK'"
    fi
    ;;

  *)
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac

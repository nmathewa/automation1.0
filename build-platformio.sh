#!/usr/bin/env bash
#
# Build the firmware projects that are known to compile.
#
# Run from the repository root, by CI or by hand:
#
#     bash build-platformio.sh
#
set -euo pipefail

# Only projects that actually build are listed. See the notes at the bottom for
# the ones that do not, and why -- adding one here before fixing it just turns
# CI red for a known reason, which teaches nobody anything.
PROJECTS=(
  "amb_display/pro_led"
)

command -v pio >/dev/null 2>&1 || {
  echo "pio not found. Install it with: pip install platformio" >&2
  exit 1
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
failed=()

for project in "${PROJECTS[@]}"; do
  echo
  echo "=============================================================="
  echo "  $project"
  echo "=============================================================="

  # Credentials are gitignored, so CI has none. A placeholder lets the
  # firmware compile; it obviously cannot associate with a network.
  example="$root/$project/include/secrets.h.example"
  secrets="$root/$project/include/secrets.h"
  if [ -f "$example" ] && [ ! -f "$secrets" ]; then
    echo "using placeholder credentials from $(basename "$example")"
    cp "$example" "$secrets"
  fi

  if ! (cd "$root/$project" && pio run); then
    failed+=("$project")
  fi
done

echo
if [ ${#failed[@]} -ne 0 ]; then
  echo "FAILED: ${failed[*]}" >&2
  exit 1
fi
echo "All ${#PROJECTS[@]} project(s) built."

# ---------------------------------------------------------------------------
# Not built, and why. Each needs a real fix, not a CI change:
#
#   amb_display/esp_tests/esp_pro_audio
#       src/main_static.cpp uses TProgmemRGBGradientPalettePtr, which FastLED
#       has never had -- it is TProgmemRGBGradientPaletteRef. Fixed on the
#       branch that moves the UDP receiver into src/; add this project here
#       once that lands.
#
#   gate_way_node
#       lib_deps names tmrh20/RF24, which the PlatformIO registry no longer
#       resolves. Needs the current package name.
#
#   single_node
#       platformio.ini declares [env:pro mini 3.3]. Environment names may only
#       contain a-z, 0-9, hyphen and underscore, so PlatformIO rejects the file
#       before building anything.
# ---------------------------------------------------------------------------

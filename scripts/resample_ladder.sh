#!/usr/bin/env bash
# resample_ladder.sh: build a controlled resolution ladder for the COST axis.
#
# PHOENIX-2014 stores each clip as a FOLDER OF PNG FRAMES (native 210x260).
# For each clip we build 4 rungs (360/480/720/1080p) as re-encoded videos with a
# FIXED codec + high bitrate, so decode cost is comparable across rungs.
#
# VALID FOR COST ONLY. These outputs are tagged resampled:true by the cataloguer,
# and the harness (wer_guard.py) refuses to emit WER for them. Never score WER here.
#
# Usage:
#   ./resample_ladder.sh <clip_list.txt> <frames_root> <out_root>
#     clip_list.txt : one clip folder name per line (the ~200-clip subset)
#     frames_root   : dir containing the clip folders of PNGs
#     out_root      : where rung videos are written (out_root/<rung>/<clip>.mp4)
#
# Interpolation: bicubic (documented, reproducible; does not affect cost).
# Codec/bitrate/container: libx264 / 20 Mbit/s / mp4 (logged by cataloguer).

set -euo pipefail

CLIP_LIST="${1:?need clip_list.txt}"
FRAMES_ROOT="${2:?need frames_root}"
OUT_ROOT="${3:?need out_root}"

# --- fixed, documented parameters (keep in sync with catalogue_metadata.py) ---
FPS=25                       # PHOENIX native frame rate
INTERP="bicubic"             # interpolation flag for scale
CODEC="libx264"              # fixed codec
BITRATE="20M"                # high, fixed bitrate so decode cost is comparable
PIXFMT="yuv420p"
CONTAINER="mp4"
# rung name -> target height (width auto, keep aspect, -2 => even number)
RUNGS=("360p:360" "480p:480" "720p:720" "1080p:1080")

# Detect the frame filename pattern inside a clip folder (varies by release).
# Common PHOENIX patterns: *.png named like images0001.png / 1/*.png / %04d.png
detect_pattern() {
  local dir="$1"
  # pick the first .png, derive a printf-style glob ffmpeg can read
  local first
  first=$(find "$dir" -maxdepth 1 -name '*.png' | sort | head -1 || true)
  if [[ -z "$first" ]]; then
    # frames may be one level deeper (e.g. dir/1/*.png)
    first=$(find "$dir" -maxdepth 2 -name '*.png' | sort | head -1 || true)
  fi
  echo "$first"
}

mkdir -p "$OUT_ROOT"
for r in "${RUNGS[@]}"; do mkdir -p "$OUT_ROOT/${r%%:*}"; done

count=0
while IFS= read -r clip; do
  [[ -z "$clip" ]] && continue
  clip_dir="$FRAMES_ROOT/$clip"
  if [[ ! -d "$clip_dir" ]]; then
    echo "WARN: missing clip dir: $clip_dir" >&2
    continue
  fi
  first_png=$(detect_pattern "$clip_dir")
  if [[ -z "$first_png" ]]; then
    echo "WARN: no PNG frames in: $clip_dir" >&2
    continue
  fi
  # Use ffmpeg glob input to avoid guessing the numeric width.
  frames_glob="$(dirname "$first_png")/*.png"

  for r in "${RUNGS[@]}"; do
    rung="${r%%:*}"; height="${r##*:}"
    out="$OUT_ROOT/$rung/$clip.$CONTAINER"
    ffmpeg -y -nostdin -loglevel error \
      -framerate "$FPS" -pattern_type glob -i "$frames_glob" \
      -vf "scale=-2:${height}:flags=${INTERP}" \
      -c:v "$CODEC" -b:v "$BITRATE" -pix_fmt "$PIXFMT" \
      "$out"
  done
  count=$((count+1))
  echo "OK [$count]: $clip -> 360/480/720/1080p"
done < "$CLIP_LIST"

echo "Done. Resampled $count clips into 4 rungs under: $OUT_ROOT"
echo "REMINDER: these are resampled:true (source 210x260) — COST measurement ONLY, never WER."
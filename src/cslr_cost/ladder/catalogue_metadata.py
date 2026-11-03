#!/usr/bin/env python3
"""
catalogue_metadata.py — catalogue the resolution-ladder outputs.

Walks the rung folders produced by resample_ladder.sh, probes each video with
ffprobe, and writes one metadata record per (clip, rung). Every record is tagged:
    resampled:        true
    source_resolution: 210x260
...and logs codec / bitrate / container / actual resolution, so the ladder is
reproducible and self-describing.

The records are validated through wer_guard.emit_result, which REFUSES to write
any record that carries a WER field while resampled==true. That guarantee is the
whole point of the ladder.

Usage:
    python3 catalogue_metadata.py <out_root> <metadata.jsonl>
      out_root      : the OUT_ROOT passed to resample_ladder.sh
      metadata.jsonl: output path (one JSON record per line)
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

from cslr_cost.ladder.wer_guard import emit_result  # blocks WER on resampled records

SOURCE_RESOLUTION = "210x260"   # PHOENIX-2014 native frame size
RUNGS = ["360p", "480p", "720p", "1080p"]


def ffprobe(path: Path) -> dict:
    """Return a dict of stream/format info for a video via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def record_for(clip: str, rung: str, path: Path) -> dict:
    info = ffprobe(path)
    vstream = next((s for s in info.get("streams", [])
                    if s.get("codec_type") == "video"), {})
    fmt = info.get("format", {})
    width = vstream.get("width")
    height = vstream.get("height")
    return {
        "clip": clip,
        "resolution_rung": rung,
        "actual_resolution": f"{width}x{height}" if width and height else None,
        "codec": vstream.get("codec_name"),
        "bit_rate": fmt.get("bit_rate"),          # container bitrate (bits/s)
        "container": path.suffix.lstrip("."),
        "duration_s": fmt.get("duration"),
        "interpolation": "bicubic",               # matches resample_ladder.sh
        "fps": 25,
        # --- the tags that make this cost-only and self-describing ---
        "resampled": True,
        "source_resolution": SOURCE_RESOLUTION,
        # NOTE: deliberately NO 'wer' field. The guard would refuse it anyway.
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    out_root = Path(sys.argv[1])
    metadata_path = Path(sys.argv[2])

    n = 0
    with metadata_path.open("w") as f:
        writer = lambda rec: f.write(json.dumps(rec) + "\n")
        for rung in RUNGS:
            rung_dir = out_root / rung
            if not rung_dir.is_dir():
                print(f"WARN: missing rung dir {rung_dir}", file=sys.stderr)
                continue
            for vid in sorted(rung_dir.glob("*.mp4")):
                rec = record_for(vid.stem, rung, vid)
                # emit_result runs the resampled-WER guard, then writes the line.
                emit_result(rec, writer=writer)
                n += 1
    print(f"Catalogued {n} (clip,rung) records -> {metadata_path}")
    print("All records tagged resampled:true, source_resolution:210x260.")
    print("Guard active: any accidental WER field on these would have raised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
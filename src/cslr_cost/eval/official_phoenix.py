"""The official PHOENIX-2014 WER: the dataset's own `evaluatePhoenix2014.sh` and sclite.

    python -m cslr_cost.eval.official_phoenix HYPS.jsonl dev [--field hyp]

Why this exists: `eval.wer.clip_counts` (MSKA's
Python WER over MSKA's port of the cleanup) scores with markers removed, and scoring it
against the raw annotation is *not* the official protocol. On VAC + SMKD's dev
hypotheses it gives 19.33 where the dataset's script gives 19.80 (the
published figure, exactly): sclite aligns differently and its references are the
groundtruth .stm, not the corpus annotation. So every official-protocol
number comes from here.

What runs: the dataset's `evaluation/evaluatePhoenix2014.sh` (sed/perl/awk
cleanup of the hypotheses, then sclite against `phoenix2014-groundtruth-*.stm`),
unchanged except that its `mergectmstm.py` call (Python 2) is dropped: it only
inserts `[EMPTY]` for clips with no hypothesis, which `write_ctm` does up front.
Counts are read from sclite's `dtl` report, so WER/del/sub/ins carry two
decimals rather than the summary's one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

#: under the shared corpus tree (`CSLR_DATA`, default ./data;
#: see `harness.cli._shared_data`)
EVAL_DIR = Path(os.environ.get("CSLR_DATA") or Path("data")) / "phoenix2014/phoenix2014-release/phoenix-2014-multisigner/evaluation"


def write_ctm(rows: list[dict], field: str, path: Path) -> None:
    """One CTM line per hypothesis word, `[EMPTY]` for an empty hypothesis.
    Clip id: the key's folder ("dev/<id>/1/" -> "<id>"), as the .stm names it."""
    with open(path, "w") as f:
        for r in rows:
            fid = r["key"].split("/")[1]
            for i, w in enumerate(r[field].split() or ["[EMPTY]"]):
                f.write(f"{fid} 1 {i / 100:.3f} {(i + 1) / 100:.3f} {w}\n")


def parse_dtl(text: str) -> dict:
    def count(label: str) -> int:
        return int(re.search(label + r"\s*=.*?\(\s*(\d+)\)", text).group(1))

    ref = int(re.search(r"Ref\. words\s*=\s*\(\s*(\d+)\)", text).group(1))
    e, s, d, i = (count(k) for k in ("Percent Total Error", "Percent Substitution",
                                     "Percent Deletions", "Percent Insertions"))
    if e != s + d + i:
        raise ValueError(f"sclite counts do not add up: {e} != {s} + {d} + {i}")
    return {"wer": 100 * e / ref, "del": 100 * d / ref, "sub": 100 * s / ref, "ins": 100 * i / ref,
            "ref_words": ref, "errors": e}


def score(rows: list[dict], split: str, field: str = "hyp", eval_dir: Path = EVAL_DIR) -> dict:
    if shutil.which("sclite") is None:
        raise RuntimeError("sclite is not on PATH; run inside the container (scripts/dev.sh)")
    script = (eval_dir / "evaluatePhoenix2014.sh").read_text()
    script = "\n".join(l for l in script.splitlines() if not l.strip().startswith("mergectmstm.py"))
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "eval.sh").write_text(script)
        shutil.copy(eval_dir / f"phoenix2014-groundtruth-{split}.stm", t)
        write_ctm(rows, field, t / "hyp.ctm")
        subprocess.run(["bash", "eval.sh", "hyp.ctm", split], cwd=t, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dtl = subprocess.run(["sclite", "-h", "out.hyp.ctm", "ctm", "-r", "tmp.stm", "stm",
                              "-f", "0", "-o", "dtl", "stdout"],
                             cwd=t, check=True, capture_output=True, text=True).stdout
    res = parse_dtl(dtl)
    res["n_clips"] = len(rows)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps", type=Path)
    ap.add_argument("split", choices=("dev", "test"))
    ap.add_argument("--field", default="hyp")
    a = ap.parse_args()
    r = score([json.loads(l) for l in open(a.hyps)], a.split, a.field)
    print(f"{a.hyps.name} [{a.field}] official {r['wer']:.2f} ({r['del']:.2f} / {r['sub']:.2f} / {r['ins']:.2f})"
          f", {r['ref_words']} ref words, {r['n_clips']} clips")


if __name__ == "__main__":
    main()

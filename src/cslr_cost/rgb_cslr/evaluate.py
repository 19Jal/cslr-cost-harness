"""WER of a released RGB CSLR checkpoint on PHOENIX-2014, both scoring conventions.

    python -m cslr_cost.rgb_cslr.evaluate --model vac \
        --weights data/VAC_SMKD/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
        --split dev --out results/hyps/vac-smkd.dev.jsonl

Decoding is the repositories' own: softmax, `ctcdecode` beam 10, blank 0, best
beam, repeats merged. Batch 1 (the deployment; their evaluation batches 8 or 2,
which only changes CorrNet, whose correlation modules see the batch padding).

Scored twice:
  - official: the dataset's own evaluatePhoenix2014.sh and sclite
    (`eval.official_phoenix`), the protocol both papers report;
  - markers removed: `eval.wer.clip_counts` (MSKA's scorer), with
    `__ON__` / `__OFF__` dropped from the hypothesis, against MSKA's references
    (matched by clip key; see docs/data.md).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from cslr_cost.rgb_cslr import data, models

MARKERS = {"__ON__", "__OFF__"}
#: Markers-removed references: one JSON line per clip, {"key": ..., "ref": ...}, in
#: MSKA's cleaned form (see docs/data.md). Override with CSLR_MSKA_REFS.
MSKA_REFS = os.environ.get("CSLR_MSKA_REFS", "data/refs/phoenix-2014.{split}.mska-refs.jsonl")


def decoder(beam: int = 10):
    import ctcdecode

    vocab = [chr(x) for x in range(20000, 20000 + models.NUM_CLASSES)]
    return ctcdecode.CTCBeamDecoder(vocab, beam_width=beam, blank_id=0, num_processes=4)


def decode(dec, logits: torch.Tensor, i2g: dict[int, str]) -> str:
    probs = logits.float().softmax(-1).cpu()[None]
    res, _, _, lens = dec.decode(probs, torch.tensor([len(logits)]))
    ids = [int(x) for x in res[0][0][: lens[0][0]]]
    return " ".join(i2g[k] for k, _ in groupby(ids))


def strip_markers(s: str) -> str:
    return " ".join(w for w in s.split() if w not in MARKERS)


def score(rows: list[dict], split: str, mska_root: Path = Path("MSKA")) -> dict:
    from cslr_cost.eval.wer import clip_counts as counts, rates

    mska = {json.loads(l)["key"]: json.loads(l)["ref"] for l in open(MSKA_REFS.format(split=split))}
    missing = [r["key"] for r in rows if r["key"] not in mska]
    if missing:
        raise ValueError(f"{len(missing)} keys have no MSKA reference, e.g. {missing[:2]}")
    from cslr_cost.eval.official_phoenix import score as official_score

    official = official_score(rows, split, "hyp")
    mska_conv = rates(counts([mska[r["key"]] for r in rows], [strip_markers(r["hyp"]) for r in rows], mska_root))
    return {"official": official, "mska_convention": mska_conv, "n_clips": len(rows)}


def run(kind: str, weights: str, split: str, out: Path, device: str = "cuda") -> dict:
    model = models.load(kind, weights).to(device)
    g2i = data.gloss_dict()
    assert len(g2i) + 1 == models.NUM_CLASSES, len(g2i)
    i2g = {i: g for g, i in g2i.items()}
    dec = decoder()
    rows, t0 = [], time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f, torch.inference_mode():
        for n, clip in enumerate(data.corpus(split)):
            frames = torch.from_numpy(data.pad(data.load_clip(clip, split))).to(device)
            logits = model(frames)
            row = {"key": clip["key"], "ref_raw": clip["label"], "hyp": decode(dec, logits, i2g),
                   "n_frames": int(len(frames)), "t_out": int(len(logits))}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
            if n % 100 == 0:
                print(f"{split} {n} clips, {time.time() - t0:.0f} s", flush=True)
    res = score(rows, split)
    res.update({"model": kind, "weights": weights, "split": split, "hyps": str(out), "beam": 10, "batch": 1})
    return res


def run_trt(kind: str, onnx_dir: Path, engine_dir: Path, precision: str, split: str, out: Path,
            do_score: bool = True, name: str | None = None) -> dict:
    """The same WER through the harness's TensorRT stages (the pipeline whose cost
    is measured), on the original PNG frames (not resampled)."""
    import cv2

    from cslr_cost.harness.stages import rgb_trt

    name = name or {"vac": "vac-smkd", "corrnet": "corrnet"}[kind]
    feat = (512,) if kind == "vac" else (128, 28, 28)
    crop = rgb_trt.RgbResizeCropStage()
    fe = rgb_trt.TrtRgbFrontendStage(onnx_dir / f"{name}.frontend.onnx", engine_dir, name, precision)
    hd = rgb_trt.TrtRgbHeadStage(onnx_dir / f"{name}.head.onnx", engine_dir, name, feat, precision)
    fe.setup(); hd.setup()
    g2i = data.gloss_dict()
    i2g = {i: g for g, i in g2i.items()}
    dec = decoder()
    rows, t0 = [], time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f, torch.inference_mode():
        for n, clip in enumerate(data.corpus(split)):
            d = data.PHOENIX / "features/fullFrame-210x260px" / split / clip["folder"].replace("*.png", "")
            crops = [crop.run((cv2.imread(str(p)), None)) for p in sorted(d.glob("*.png"))]
            logits = hd.run(rgb_trt.assemble_features(fe.run_batch(crops)))[0]
            row = {"key": clip["key"], "ref_raw": clip["label"], "hyp": decode(dec, logits, i2g),
                   "n_frames": len(crops), "t_out": int(len(logits))}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
            if n % 100 == 0:
                print(f"{split} {n} clips, {time.time() - t0:.0f} s", flush=True)
    res = score(rows, split) if do_score else {}
    res.update({"model": kind, "name": name, "runtime": "tensorrt", "precision": precision, "split": split,
                "hyps": str(out), "frontend_engine": fe.engine_info.short_hash,
                "head_engine": hd.engine_info.short_hash, "beam": 10, "batch": 1})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("vac", "corrnet"), required=True)
    ap.add_argument("--weights", default=None, help="checkpoint (PyTorch runtime)")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rescore", action="store_true", help="score an existing --out file, do not run the model")
    ap.add_argument("--runtime", choices=("pytorch", "tensorrt"), default="pytorch")
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--onnx-dir", type=Path, default=Path("models/onnx/rgb"))
    ap.add_argument("--engine-dir", type=Path, default=Path("models/engines/rtx3060"))
    ap.add_argument("--name", default=None,
                    help="TensorRT: ONNX / engine stem (default: vac-smkd / corrnet); `vac` for VAC's paper checkpoint")
    ap.add_argument("--no-score", action="store_true",
                    help="write hypotheses only (a machine without sclite, e.g. the Orin); score them with --rescore")
    a = ap.parse_args()
    if a.runtime == "tensorrt" and not a.rescore:
        res = run_trt(a.model, a.onnx_dir, a.engine_dir, a.precision, a.split, a.out, not a.no_score, a.name)
        if a.no_score:
            print(json.dumps(res))
            return
    elif a.rescore:
        res = score([json.loads(l) for l in open(a.out)], a.split)
    else:
        res = run(a.model, a.weights, a.split, a.out)
    for conv in ("mska_convention", "official"):
        w = res[conv]
        print(f"{a.model} {a.split} {conv:16} {w['wer']:.2f} ({w['del']:.2f} / {w['sub']:.2f} / {w['ins']:.2f})")
    print(json.dumps(res))


if __name__ == "__main__":
    main()

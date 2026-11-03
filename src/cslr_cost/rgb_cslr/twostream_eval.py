"""TwoStream-SLR's WER through the harness's TensorRT stages.

    python -m cslr_cost.rgb_cslr.twostream_eval --split dev --precision fp16 \
        --out results/hyps/twostream-slr.fp16.dev.jsonl

The original 210x260 frames (not resampled) through the deployed path's own
stages: the RGB resize (`TwoStreamPoseStage._rgb`), heatmap rendering
(`TwoStreamPostprocStage`), the head engine, and the CTC stage at beam 5. Keypoints
are the dev box's FP32 TensorRT HRNet-w48 DARK + flip extraction (the MSKA baseline's,
the same stage the pipeline runs; `--keypoints`), so the HRNet pass is not repeated.

Vocabulary: TwoStream's `gloss2ids.pkl` (plain dict), read with an unpickler that
refuses every global. Scored on both conventions (`rgb_cslr.evaluate.score`).

`--variant keypoint`: TwoStream's keypoint stream alone (`twostream-kp.onnx`), the same
keypoints and heatmaps, no RGB input.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from cslr_cost.eval.dataset import iter_clips
from cslr_cost.harness.stages import twostream_trt as ts
from cslr_cost.harness.stages.ctc import CtcDecodeStage
from cslr_cost.rgb_cslr import data
from cslr_cost.rgb_cslr.evaluate import score

#: HRNet-w48 keypoints for the split, in MSKA's pickle layout (MSKA releases them).
KEYPOINTS = Path("data/keypoints/phoenix-2014.{split}")
GLOSS2IDS = Path("data/TwoStream-SLRT/SLRT/TwoStreamNetwork/data/phoenix-2014/gloss2ids.pkl")


class _Plain(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def id2gloss() -> dict[int, str]:
    g2i = _Plain(open(GLOSS2IDS, "rb")).load()
    # TwoStream's tokenizer inverts in insertion order, the last entry winning: id 4 is
    # both '<mask>' and 'REGEN' and decodes as 'REGEN' (its released hypotheses agree)
    i2g = {}
    for g, i in g2i.items():
        i2g[int(i)] = str(g)
    assert len(i2g) == ts.VOCAB and i2g[4] == "REGEN", (len(i2g), i2g.get(4))
    return i2g


def run(split: str, precision: str, out: Path, onnx: Path, engine_dir: Path, do_score: bool = True,
        keypoint_only: bool = False) -> dict:
    head = ts.TrtTwoStreamHeadStage(onnx, engine_dir, precision=precision, keypoint_only=keypoint_only)
    head.setup()
    post = ts.TwoStreamPostprocStage(frame_wh=(210, 260))
    ctc = CtcDecodeStage(vocab_size=ts.VOCAB, beam_width=5)
    ctc.setup()
    i2g = id2gloss()
    rgb_of = ts.TwoStreamPoseStage.__new__(ts.TwoStreamPoseStage)    # only its _rgb
    rgb_of.device = "cuda"
    frames_root = data.PHOENIX / "features/fullFrame-210x260px"
    rows, t0 = [], time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f, torch.no_grad():
        for n, clip in enumerate(iter_clips(Path(str(KEYPOINTS).format(split=split)))):
            d = frames_root / clip.key.rstrip("/")
            paths = sorted(d.glob("*.png"))
            if len(paths) != clip.num_frames or len(clip.keypoint) != clip.num_frames:
                raise ValueError(f"{clip.key}: {len(paths)} frames, {len(clip.keypoint)} keypoint rows, "
                                 f"record says {clip.num_frames}")
            if clip.num_frames > 400:
                raise ValueError(f"{clip.key}: over 400 frames; TwoStream centre-crops those (not implemented)")
            rgb = [None] * clip.num_frames if keypoint_only else rgb_of._rgb([cv2.imread(str(p)) for p in paths])
            units = [post.run((kp, r)) for kp, r in zip(clip.keypoint, rgb)]
            logits = head.run(ts.assemble_twostream(units))
            ids = ctc.run(logits)[0]
            row = {"key": clip.key, "ref_raw": None, "hyp": " ".join(i2g[i] for i in ids),
                   "n_frames": clip.num_frames, "t_out": int(logits.shape[1])}
            rows.append(row)
            if n % 100 == 0:
                print(f"{split} {n} clips, {time.time() - t0:.0f} s", flush=True)
    raw = {c["key"]: c["label"] for c in data.corpus(split)}
    with open(out, "w") as f:
        for r in rows:
            r["ref_raw"] = raw[r["key"]]
            f.write(json.dumps(r) + "\n")
    res = score(rows, split) if do_score else {}
    res.update({"model": "twostream-kp" if keypoint_only else "twostream-slr", "precision": precision, "split": split, "hyps": str(out),
                "head_engine": head.engine_info.short_hash, "beam": 5})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", choices=("twostream", "keypoint"), default="twostream")
    ap.add_argument("--onnx", type=Path, default=None,
                    help="default models/onnx/twostream/twostream-{slr,kp}.onnx by --variant")
    ap.add_argument("--engine-dir", type=Path, default=Path("models/engines/rtx3060"))
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    if a.rescore:
        res = score([json.loads(l) for l in open(a.out)], a.split)
    else:
        kp = a.variant == "keypoint"
        onnx = a.onnx or Path(f"models/onnx/twostream/twostream-{'kp' if kp else 'slr'}.onnx")
        res = run(a.split, a.precision, a.out, onnx, a.engine_dir, keypoint_only=kp)
    for conv in ("mska_convention", "official"):
        w = res[conv]
        print(f"{res.get('model', 'twostream')} {a.split} {conv:16} {w['wer']:.2f} ({w['del']:.2f} / {w['sub']:.2f} / {w['ins']:.2f})")
    print(json.dumps(res))


if __name__ == "__main__":
    main()

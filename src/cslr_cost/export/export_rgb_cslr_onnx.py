"""Export VAC + SMKD or CorrNet to two ONNX graphs, for TensorRT.

    python -m cslr_cost.export.export_rgb_cslr_onnx --model vac \
        --weights data/VAC_SMKD/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
        --out-dir models/onnx/rgb

Two graphs because the pipeline has two granularities (harness, Chunk 2):

  <name>.frontend.onnx  per frame: (N, 3, 224, 224) normalised RGB ->
                        VAC (N, 512) pooled ResNet18 features;
                        CorrNet (N, 128, 28, 28), everything before its first
                        correlation module (the frame-independent part).
  <name>.head.onnx      per clip: padded features (1, T, ...) -> (T', 1296)
                        sequence logits. CorrNet's correlation layers, layer 3,
                        layer 4, then the temporal conv, BiLSTM and classifier.

CorrNet's correlation is exported in its batched-matmul form
(`GetCorrelation.use_matmul`), held equal to the einsum by the tests.

Parity against PyTorch on a real dev clip is checked with onnxruntime (CPU)
before the sidecar is written; a graph that does not match is not kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cslr_cost.export.export_rtmpose_onnx import OPSET, sha256_file
from cslr_cost.rgb_cslr import data, models

NAMES = {"vac": "vac-smkd"}
PARITY_TOL = 1e-3      # max |logit| difference, fp32, CPU


def export(kind: str, weights: Path, out_dir: Path, clip_index: int = 3, name: str | None = None) -> dict:
    m = models.load(kind, str(weights))
    for mod in m.modules():
        if isinstance(mod, models.GetCorrelation):
            mod.use_matmul = True
    fe, hd = models.FrameFrontend(m).eval(), models.ClipHead(m).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or NAMES[kind]
    f_path, h_path = out_dir / f"{name}.frontend.onnx", out_dir / f"{name}.head.onnx"

    clip = data.corpus("dev")[clip_index]
    raw = torch.from_numpy(data.load_clip(clip, "dev"))
    with torch.no_grad():
        feats = torch.cat([fe(raw[i:i + 16]) for i in range(0, len(raw), 16)])
    t = len(feats)
    right = -(-t // 4) * 4 - t + data.LEFT_PAD
    padded = torch.cat([feats[:1].expand(data.LEFT_PAD, *feats.shape[1:]), feats,
                        feats[-1:].expand(right, *feats.shape[1:])])[None]

    torch.onnx.export(fe, raw[:2], str(f_path), opset_version=OPSET,
                      input_names=["input"], output_names=["features"],
                      dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
                      do_constant_folding=True)
    torch.onnx.export(hd, padded, str(h_path), opset_version=OPSET,
                      input_names=["features"], output_names=["logits"],
                      dynamic_axes={"features": {1: "frames"}, "logits": {0: "steps"}},
                      do_constant_folding=True)

    import onnxruntime as ort

    sf = ort.InferenceSession(str(f_path), providers=["CPUExecutionProvider"])
    sh = ort.InferenceSession(str(h_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref_f = fe(raw[:4]).numpy()
        ref_h = hd(padded).numpy()
    got_f = sf.run(None, {"input": raw[:4].numpy()})[0]
    got_h = sh.run(None, {"features": padded.numpy()})[0]
    d_f, d_h = float(np.abs(ref_f - got_f).max()), float(np.abs(ref_h - got_h).max())
    agree = float((ref_h.argmax(-1) == got_h.argmax(-1)).mean())
    if d_h > PARITY_TOL or agree < 1.0:
        f_path.unlink(); h_path.unlink()
        raise SystemExit(f"{name}: ONNX head parity failed (max |d| {d_h:.2e}, argmax agree {agree:.3f})")

    sidecar = {
        "schema": "rgb-cslr-onnx-export/1",
        "model": name,
        "checkpoint": str(weights),
        "checkpoint_sha256": sha256_file(weights),
        "frontend_onnx_sha256": sha256_file(f_path),
        "head_onnx_sha256": sha256_file(h_path),
        "opset": OPSET,
        "frontend_output": list(ref_f.shape[1:]),
        "parity_clip": clip["key"],
        "parity_frames": int(padded.shape[1]),
        "parity_frontend_max_abs": d_f,
        "parity_head_max_abs": d_h,
        "parity_head_argmax_agree": agree,
        "torch": torch.__version__,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=tuple(NAMES), required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("models/onnx/rgb"))
    ap.add_argument("--name", default=None,
                    help="file stem (default: vac-smkd); `--model vac --name vac` for VAC's paper checkpoint")
    a = ap.parse_args()
    s = export(a.model, a.weights, a.out_dir, name=a.name)
    print(f"{s['model']}: frontend {s['frontend_onnx_sha256'][:12]} (|d| {s['parity_frontend_max_abs']:.1e}), "
          f"head {s['head_onnx_sha256'][:12]} (|d| {s['parity_head_max_abs']:.1e}, argmax {s['parity_head_argmax_agree']:.3f})")


if __name__ == "__main__":
    main()

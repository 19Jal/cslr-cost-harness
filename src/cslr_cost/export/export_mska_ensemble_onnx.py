#!/usr/bin/env python3
"""Export MSKA's keypoint recognition network -- backbone and ALL FOUR visual heads -- to ONNX.

    scripts/dev.sh python -m cslr_cost.export.export_mska_ensemble_onnx \\
        --out models/onnx/mska-phoenix2014-ensemble.onnx

It uses a `padding="same"` patch, cudnn-off tracing, and dynamic batch and
time axes. It returns -- the four heads' logits (left, right, body, fuse) rather than
the fuse head's alone -- because the deployed output is the four-head ensemble,
and a fuse-only graph both decodes differently and costs three heads less.
`harness/stages/head_trt.py` does the ensemble arithmetic itself.

The network is built through the harness's own `MskaHeadStage`, so the exported
weights are the ones the PyTorch rows measured, loaded by the same code path.

Then parity: at several sequence lengths, not only the traced one, the ONNX
ensemble (via ONNX Runtime) is compared with `MskaHeadStage.run`. A length
baked in as a constant during tracing still exports and still passes
`onnx.checker`; it fails only at T != the dummy, which is exactly the shape a
real clip has.

Parity is judged where the decoder looks, and all of it is printed. The raw
max |delta| over every class is dominated by classes with probability ~1e-10,
where the log turns a 1e-6 kernel difference into 1e-2 and no beam search ever
visits: it is reported, not gated on. The gate is (a) log-softmax agreement on
classes with p > 1e-4, (b) identical per-frame argmax, and (c) identical
`ctcdecode` glosses -- (c) being what the export must preserve.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

OUTPUT_NAMES = ["logits_left", "logits_right", "logits_body", "logits_fuse"]


def patch_same_padding(module) -> int:
    """`padding="same"` -> the equivalent integer padding, which ONNX can export.

    From `export_mska_onnx.py`. Valid for odd kernels at stride 1, which is the
    only case `same` is defined for in torch.
    """
    import torch.nn as nn

    n = 0
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and m.padding == "same":
            ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
            if any(k % 2 == 0 for k in ks):
                raise ValueError(f"even kernel {ks} under padding='same' is asymmetric")
            m.padding = tuple((k - 1) // 2 for k in ks)
            n += 1
    return n


def _ensemble_np(outs):
    import numpy as np

    probs = None
    for o in outs:  # already in OUTPUT_NAMES order
        z = o - o.max(axis=-1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
        probs = p if probs is None else probs + p
    return np.log(probs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mska", default="MSKA")
    ap.add_argument("--config", default="configs/phoenix-2014_s2g.yaml")
    ap.add_argument("--weights", default="pretrained_models/best.pth")
    ap.add_argument("--out", default="models/onnx/mska-phoenix2014-ensemble.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--trace-frames", type=int, default=64)
    ap.add_argument("--parity-frames", default="32,128,200")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |delta| of log-softmax on classes with p > 1e-4, FP32 export vs PyTorch")
    ap.add_argument("--strict-fp32", action="store_true",
                    help="disable TF32 in the PyTorch reference. On Ampere, torch 2.0 "
                         "runs cuDNN convolutions in TF32 by default, so its 'fp32' is "
                         "not the strict FP32 ONNX Runtime's CPU provider computes")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    if args.strict_fp32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    print(f"reference TF32: cudnn={torch.backends.cudnn.allow_tf32} "
          f"matmul={torch.backends.cuda.matmul.allow_tf32}")

    from cslr_cost.harness.stages.head import MskaHeadStage

    mska = Path(args.mska).resolve()
    stage = MskaHeadStage(mska, mska / args.config, mska / args.weights, precision="fp32")
    stage.setup()
    rec = stage.model

    class EnsembleHeads(torch.nn.Module):
        def __init__(self, rec):
            super().__init__()
            self.rec = rec

        def forward(self, keypoint):
            n, _c, t, _v = keypoint.shape
            out_len = t // 4
            mask = torch.ones(n, 1, out_len, dtype=torch.bool, device=keypoint.device)
            lengths = torch.full((n,), out_len, dtype=torch.long, device=keypoint.device)
            src = {"keypoint": keypoint, "mask": mask, "new_src_lengths": lengths}
            fuse, left, right, body = self.rec.visual_backbone_keypoint(src)
            kw = {"mask": mask, "valid_len_in": lengths}
            return (
                self.rec.left_visual_head(x=left, **kw)["gloss_logits"],
                self.rec.right_visual_head(x=right, **kw)["gloss_logits"],
                self.rec.body_visual_head(x=body, **kw)["gloss_logits"],
                self.rec.fuse_visual_head(x=fuse, **kw)["gloss_logits"],
            )

    wrapped = EnsembleHeads(rec).eval()
    print(f"patched {patch_same_padding(wrapped)} conv layers: padding 'same' -> explicit int")

    def dummy(t, seed=0):
        g = torch.Generator().manual_seed(seed)
        xy = torch.rand(1, 2, t, 133, generator=g) * 2 - 1        # normalised coords
        conf = torch.rand(1, 1, t, 133, generator=g)
        return torch.cat([xy, conf], dim=1).cuda()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False   # as in export_mska_onnx.py
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped, (dummy(args.trace_frames),), str(out),
                input_names=["keypoint"],
                output_names=OUTPUT_NAMES,
                dynamic_axes={"keypoint": {0: "batch", 2: "time"},
                              **{o: {0: "batch", 1: "time_out"} for o in OUTPUT_NAMES}},
                opset_version=args.opset,
                do_constant_folding=True,
            )
    finally:
        torch.backends.cudnn.enabled = cudnn
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    import onnx

    onnx.checker.check_model(str(out))
    print("onnx.checker: OK")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: parity NOT checked", file=sys.stderr)
        return 2

    from cslr_cost.harness.stages.ctc import CtcDecodeStage

    ctc = CtcDecodeStage(vocab_size=stage.vocab_size, beam_width=5)
    ctc.setup()

    def log_softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ok = True
    parity: list[dict] = []
    for i, t in enumerate(int(x) for x in args.parity_frames.split(",")):
        kp = dummy(t, seed=100 + i)
        ref = stage.run(kp).float().cpu().numpy()
        got = _ensemble_np(sess.run(OUTPUT_NAMES, {"keypoint": kp.cpu().numpy()}))
        if ref.shape != got.shape:
            print(f"T={t}: shape {got.shape} != PyTorch {ref.shape}", file=sys.stderr)
            return 1
        raw = float(np.abs(ref - got).max())
        ref_ls, got_ls = log_softmax(ref), log_softmax(got)
        live = np.exp(ref_ls) > 1e-4
        live_delta = float(np.abs(ref_ls - got_ls)[live].max())
        agree = float((ref.argmax(-1) == got.argmax(-1)).mean())
        same_glosses = ctc.run(torch.from_numpy(ref)) == ctc.run(torch.from_numpy(got))
        passed = live_delta <= args.tol and agree == 1.0 and same_glosses
        ok &= passed
        parity.append({"frames": t, "live_class_delta": live_delta, "argmax_agree": agree,
                       "same_glosses": bool(same_glosses), "pass": bool(passed)})
        print(f"T={t:>4}: out {got.shape}  live-class |delta| {live_delta:.2e}  "
              f"argmax {agree:.4f}  glosses {'same' if same_glosses else 'DIFFER'}  "
              f"(raw all-class max {raw:.2e})  {'PASS' if passed else 'FAIL'}")

    print(f"parity {'PASS' if ok else 'FAIL'} (tol {args.tol:.0e} on classes with p > 1e-4)")

    # Sidecar: which weights this ONNX is, so a device that holds only the ONNX
    # (the Orin) can still say which checkpoint its WER rows measured.
    import hashlib
    import json

    weights = mska / args.weights
    h = hashlib.sha256()
    with weights.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    out.with_suffix(".json").write_text(json.dumps({
        "weights": str(weights), "weights_sha256": h.hexdigest(),
        "config": args.config, "opset": args.opset,
        "parity": {"pass": bool(ok), "tol": args.tol, "per_length": parity},
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

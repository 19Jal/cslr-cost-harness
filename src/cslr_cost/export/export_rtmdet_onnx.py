#!/usr/bin/env python3
"""Export RTMDet-nano (person) to ONNX: backbone + neck + head, raw per-level tensors.

    scripts/dev.sh python -m cslr_cost.export.export_rtmdet_onnx \\
        --out models/onnx/rtmdet-nano-person-320.onnx

The third neural stage. `decode`, `crop`, `postproc` and `ctc` are not neural
and stay as they are, and RTMPose and the MSKA head are already exported -- this
closes the set, so the `--detector rtmdet` ablation can run on the same runtime
as the stages it is being compared against instead of mixing PyTorch detection
with TensorRT pose.

--------------------------------------------------------------------------
WHAT IS IN THE GRAPH, AND WHAT IS NOT
--------------------------------------------------------------------------
In: the network, from a normalised 320x320 NCHW tensor to six tensors --
`cls_8/16/32` and `bbox_8/16/32`, one pair per FPN level.

Out of it, in `stages/detect_trt.py`: the letterbox (resize keep-ratio, pad
114), the normalisation, and the decode (sigmoid, distance-to-box against the
stride grids, NMS, rescale to frame coordinates). Same division as RTMPose's
SimCC decode, for the same reason -- MMDeploy's graph-side NMS is a plugin that
has to exist on the target, and OpenMMLab is kept off the Orin. Both
halves are held to mmdet's own output by `tests/harness/test_detect_trt.py`.

`bbox_preds` leave the head as **distances in pixels** already
(`reg.exp() * stride` in `RTMDetHead.forward`), so the decode needs the priors
but no variance or log-space arithmetic.

--------------------------------------------------------------------------
STATIC 320x320, DYNAMIC BATCH
--------------------------------------------------------------------------
RTMDet's test pipeline letterboxes every frame to exactly 320x320, so the
spatial dims are static by construction and TensorRT gets to specialise for
them. Only the batch axis is dynamic, for P7's batch regime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: One (cls, bbox) pair per FPN level, named by stride.
OUTPUT_NAMES = ["cls_8", "bbox_8", "cls_16", "bbox_16", "cls_32", "bbox_32"]

DEFAULT_CONFIG = "data/rtmdet/rtmdet_nano_320-8xb32_coco-person.py"
DEFAULT_CKPT = (
    "data/rtmdet/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default="models/onnx/rtmdet-nano-person-320.onnx")
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max RELATIVE delta per output tensor, ONNX Runtime vs PyTorch. "
                         "Relative because bbox_* are distances in pixels (values to ~320), "
                         "where an absolute 1e-3 is a 1e-5 disagreement and means nothing")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from mmdet.apis import init_detector

    # Strict FP32: torch runs cuDNN convolutions in TF32 on Ampere by default,
    # which would show up here as a fictitious ~1e-2 "export error".
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model = init_detector(args.config, args.checkpoint, device="cuda").eval()

    class Detector(torch.nn.Module):
        """Everything between the normalised tensor and the head's raw outputs."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            cls_scores, bbox_preds = self.m.bbox_head(self.m.extract_feat(x))
            return (
                cls_scores[0], bbox_preds[0],
                cls_scores[1], bbox_preds[1],
                cls_scores[2], bbox_preds[2],
            )

    wrapped = Detector(model).eval()
    s = args.input_size
    dummy = torch.randn(1, 3, s, s, device="cuda")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapped, (dummy,), str(out),
            input_names=["input"],
            output_names=OUTPUT_NAMES,
            dynamic_axes={"input": {0: "batch"}, **{o: {0: "batch"} for o in OUTPUT_NAMES}},
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    import onnx

    onnx.checker.check_model(str(out))
    print("onnx.checker: OK")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: parity NOT checked", file=sys.stderr)
        return 2

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ok = True
    for b in (1, 4):
        x = torch.randn(b, 3, s, s)
        with torch.no_grad():
            ref = [t.float().cpu().numpy() for t in wrapped(x.cuda())]
        got = sess.run(OUTPUT_NAMES, {"input": x.numpy()})
        if not all(a.shape == g.shape for a, g in zip(ref, got)):
            print(f"  B={b}: SHAPE MISMATCH", file=sys.stderr)
            return 1
        worst_rel = 0.0
        parts = []
        for name, a, g in zip(OUTPUT_NAMES, ref, got):
            scale = max(float(np.abs(a).max()), 1e-6)
            abs_d = float(np.abs(a - g).max())
            rel = abs_d / scale
            worst_rel = max(worst_rel, rel)
            parts.append(f"{name} {abs_d:.1e}/{rel:.1e}")
        passed = worst_rel <= args.tol
        ok &= passed
        print(f"  B={b}: per-output abs/rel  " + "  ".join(parts))
        print(f"        worst relative {worst_rel:.2e}  {'PASS' if passed else 'FAIL'}")
    print("  shapes: " + ", ".join(f"{n}{list(g.shape)}" for n, g in zip(OUTPUT_NAMES, got)))
    print(f"parity {'PASS' if ok else 'FAIL'} (tol {args.tol:.0e} relative)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

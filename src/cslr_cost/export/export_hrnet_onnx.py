"""export_hrnet_onnx.py — the HRNet behind MSKA's released keypoints -> ONNX.

    python -m cslr_cost.export.export_hrnet_onnx \\
        --out models/onnx/hrnet-w48-dark-384x288.onnx

MSKA's released PHOENIX keypoints were produced by mmpose's HRNet-w48 DARK at
384x288 (COCO-WholeBody, `hrnet_w48_coco_wholebody_384x288_dark-f5726563`) with
flip test, on a whole-frame box: run that way, it reproduces them to 0.001 px
median and 0.0001 in confidence (2026-10-18, `docs/MSKA_Baseline_Results.MD`).
This exports the network so the harness can price that front end.

    input   "input"     float32 [batch, 3, 384, 288]   (normalised RGB, NCHW)
    output  "heatmaps"  float32 [batch, 133, 96, 72]
    opset 17, dynamic batch

Only backbone + head. The flip test (a second forward on the mirrored input,
flipped back and averaged) and the DARK decode are mmpose's post-network steps;
`harness/stages/pose_trt.TrtHrnetStage` reproduces them, held to mmpose by
`tests/harness/test_pose_hrnet.py`. Preprocessing stays out of the graph, as for
RTMPose. The sidecar JSON names the checkpoint and the parity of
the export against it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cslr_cost.export.export_rtmpose_onnx import OPSET, sha256_file

DEFAULT_CHECKPOINT = Path("data/coco-wholebody/"
                          "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth")
CONFIG_NAME = "td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
INPUT_HW = (384, 288)


def default_config() -> Path:
    import mmpose

    return (Path(os.path.dirname(mmpose.__file__))
            / ".mim/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody" / CONFIG_NAME)


class _HeatmapsOnly(torch.nn.Module):
    """Backbone + head: one forward, heatmaps out, no flip and no decode."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.head.forward(self.model.extract_feat(x))


def export(config: Path, checkpoint: Path, out: Path, max_abs_diff: float = 1e-4) -> dict:
    from mmpose.apis import init_model
    import onnxruntime as ort

    model = init_model(str(config), str(checkpoint), device="cpu").eval()
    wrapped = _HeatmapsOnly(model).eval()
    h, w = INPUT_HW
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, torch.zeros(1, 3, h, w), str(out), opset_version=OPSET,
        input_names=["input"], output_names=["heatmaps"],
        dynamic_axes={"input": {0: "batch"}, "heatmaps": {0: "batch"}},
        do_constant_folding=True,
    )

    x = np.random.default_rng(0).standard_normal((2, 3, h, w)).astype(np.float32)
    with torch.no_grad():
        ref = wrapped(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"input": x})
    diff = float(np.abs(ref - got).max())

    sidecar = {
        "schema": "hrnet-onnx-export/1",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx_sha256": sha256_file(out),
        "opset": OPSET,
        "input": f"{h}x{w}",
        "output": "heatmaps " + "x".join(str(d) for d in got.shape[1:]),
        "parity_vs_pytorch_max_abs": diff,
        "torch": torch.__version__,
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    if diff > max_abs_diff:
        raise SystemExit(f"REFUSED: export differs from its checkpoint by {diff:.2e} (> {max_abs_diff:g})")
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cslr_cost.export.export_hrnet_onnx",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=None, help=f"default: mmpose's {CONFIG_NAME}")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--out", required=True)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    s = export(Path(a.config) if a.config else default_config(), Path(a.checkpoint), Path(a.out))
    print(f"{a.out}  sha256 {s['onnx_sha256'][:12]}  parity {s['parity_vs_pytorch_max_abs']:.2e}  "
          f"{s['output']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

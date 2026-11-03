"""export_rtmpose_onnx.py — an mmpose RTMPose checkpoint -> the ONNX our pose stage compiles.

    python -m cslr_cost.export.export_rtmpose_onnx \\
        --config  <mmpose config.py> --checkpoint <.pth> --out models/onnx/rtmpose-m.coco-wb.onnx

The first RTMPose ONNX files were exported by hand,
with no exporter in the repository and no record of their source (blocker 14,
closed by hash and output parity instead -- `manifests/frontend_checkpoints.json`).
This is the exporter that should have existed: it reproduces their interface
exactly, so every downstream stage takes its output unchanged --

    input   "input"    float32 [batch, 3, 256, 192]   (normalised RGB, NCHW)
    output  "simcc_x"  float32 [batch, K, 2 * 192]
            "simcc_y"  float32 [batch, K, 2 * 256]
    opset 17, dynamic batch

-- and writes a sidecar JSON naming the checkpoint (path, sha256), the config,
and the parity of the export against the checkpoint run in PyTorch on the same
input. An export that does not reproduce its checkpoint is refused.

Preprocessing is not in the graph, as in the original exports: the pose stage
owns BGR->RGB and mean/std.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

OPSET = 17


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class _SimccOnly(torch.nn.Module):
    """Backbone + neck + head, returning the two SimCC tensors and nothing else."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        simcc_x, simcc_y = self.model.head.forward(self.model.extract_feat(x))
        return simcc_x, simcc_y


def export(config: Path, checkpoint: Path, out: Path, input_hw=(256, 192),
           max_abs_diff: float = 1e-4) -> dict:
    from mmpose.apis import init_model
    import onnxruntime as ort

    model = init_model(str(config), str(checkpoint), device="cpu").eval()
    wrapped = _SimccOnly(model).eval()
    h, w = input_hw
    dummy = torch.zeros(1, 3, h, w)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, dummy, str(out), opset_version=OPSET,
        input_names=["input"], output_names=["simcc_x", "simcc_y"],
        dynamic_axes={"input": {0: "batch"}, "simcc_x": {0: "batch"}, "simcc_y": {0: "batch"}},
        do_constant_folding=True,
    )

    # Parity: the checkpoint in PyTorch against the exported graph, same input.
    x = np.random.default_rng(0).standard_normal((4, 3, h, w)).astype(np.float32)
    with torch.no_grad():
        px, py = wrapped(torch.from_numpy(x))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ox, oy = sess.run(None, {"input": x})
    diff = float(max(np.abs(px.numpy() - ox).max(), np.abs(py.numpy() - oy).max()))

    sidecar = {
        "schema": "rtmpose-onnx-export/1",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx_sha256": sha256_file(out),
        "opset": OPSET,
        "input": f"{h}x{w}",
        "parity_vs_pytorch_max_abs": diff,
        "torch": torch.__version__,
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    if diff > max_abs_diff:
        raise SystemExit(f"REFUSED: export differs from its checkpoint by {diff:.2e} (> {max_abs_diff:g})")
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cslr_cost.export.export_rtmpose_onnx",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--input-hw", default="256x192")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    h, w = (int(v) for v in a.input_hw.split("x"))
    s = export(Path(a.config), Path(a.checkpoint), Path(a.out), (h, w))
    print(f"{a.out}  sha256 {s['onnx_sha256'][:12]}  parity {s['parity_vs_pytorch_max_abs']:.2e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

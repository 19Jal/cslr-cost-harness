"""coco_wholebody_trt.py — COCO-WholeBody AP of our TensorRT pose engines.

    python -m cslr_cost.eval.coco_wholebody_trt --scale m --precision int8

`scripts/eval_coco_wholebody.sh` measures AP for a PyTorch checkpoint through
mmpose's own `tools/test.py`. That cannot see what an *engine* does: FP16 and
INT8 exist only as TensorRT engines, and the pose sweep needs AP
in every cell. This runs the same protocol with one thing swapped.

  * **Same** dataset, pipeline and metric: mmpose's `CocoWholeBodyDataset`
    with ground-truth boxes (`bbox_file=None`), its `TopdownAffine` crop, and
    its `CocoWholeBodyMetric`, built from the extractor's own mmpose config with
    the overrides `eval_coco_wholebody.sh` passes.
  * **Swapped**: the forward pass. Each crop goes through the harness's
    `TrtRtmposeStage` -- the engine, preprocessing and SimCC decode that
    extraction and the cost grid use -- and back to image coordinates with its
    `_to_frame_coords`.

**No flip test.** mmpose's RTMPose test configs set `flip_test=True`: each crop
is also run mirrored and the two averaged, worth about 1.5 AP. The deployed
pipeline does not do it, so this measures without it. Checked on RTMPose-m:
PyTorch with `model.test_cfg.flip_test=False` scores 0.6158, the FP32 engine here
0.6157; with flip (mmpose's default) 0.6308. So the swap
changes nothing but the forward pass, and FP16/INT8 then measure what precision
alone does to localisation. Report both columns; compare neither across.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

DATA_ROOT = Path("data/coco-wholebody")
MMPOSE_CONFIGS = Path("/opt/venv/lib/python3.10/site-packages/mmpose/.mim/configs/wholebody_2d_keypoint/rtmpose")
CONFIG_FOR = {
    "dw-ucoco": MMPOSE_CONFIGS / "ubody/rtmpose-{s}_8xb64-270e_coco-ubody-wholebody-256x192.py",
    "coco-wb": MMPOSE_CONFIGS / "coco-wholebody/rtmpose-{s}_8xb64-270e_coco-wholebody-256x192.py",
}
#: DWPose-l is the 384x288 config, not the 256x192 one of the same name.
CONFIG_L = MMPOSE_CONFIGS / "ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
DEFAULT_OUT = Path("results/accuracy/coco_wholebody_trt.jsonl")


def evaluate(scale: str, precision: str, weights: str, onnx_dir: Path, engine_dir: Path,
             batch: int = 32, limit: Optional[int] = None, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    from mmengine.config import Config
    from mmengine.structures import InstanceData
    from mmpose.registry import DATASETS, METRICS
    from mmpose.utils import register_all_modules

    from cslr_cost.harness.stages.crop import Crop
    from cslr_cost.harness.stages.pose_trt import TrtRtmposeStage, onnx_for_scale

    register_all_modules()
    cfg = Config.fromfile(str(CONFIG_L if scale == "l" else CONFIG_FOR[weights]).format(s=scale))
    ds = cfg.test_dataloader.dataset
    ds.data_root = str(data_root) + "/"
    ds.ann_file = "coco_wholebody_val_v1.0.json"
    ds.data_prefix = dict(img="val2017/")
    ds.bbox_file = None
    dataset = DATASETS.build(ds)
    ev = cfg.test_evaluator
    ev.ann_file = str(data_root / "coco_wholebody_val_v1.0.json")
    metric = METRICS.build(ev)
    metric.dataset_meta = dataset.metainfo

    onnx = onnx_for_scale(onnx_dir, scale, precision, weights)
    pose = TrtRtmposeStage(onnx, engine_dir, scale=scale, precision=precision,
                           opt_batch=batch, max_batch=max(64, batch))
    pose.setup()

    n = len(dataset) if limit is None else min(limit, len(dataset))
    started = time.perf_counter()
    for lo in range(0, n, batch):
        items = [dataset[i] for i in range(lo, min(lo + batch, n))]
        crops, samples = [], []
        for it in items:
            s = it["data_samples"]
            img = it["inputs"].permute(1, 2, 0).numpy()                     # CHW -> HWC, BGR uint8
            crops.append(Crop(
                img=np.ascontiguousarray(img),
                center=np.asarray(s.metainfo["input_center"], dtype=np.float32).reshape(2),
                scale=np.asarray(s.metainfo["input_scale"], dtype=np.float32).reshape(2),
                input_size=tuple(s.metainfo["input_size"]),
                source_wh=tuple(s.metainfo["ori_shape"][::-1]),
            ))
            samples.append(s)
        kps = pose._infer(crops)
        out = []
        for s, kp in zip(samples, kps):
            pred = InstanceData(keypoints=kp[None, :, :2], keypoint_scores=kp[None, :, 2])
            if "bboxes" in s.gt_instances:
                pred.bboxes = s.gt_instances.bboxes
                pred.bbox_scores = s.gt_instances.bbox_scores
            s.pred_instances = pred
            out.append(s.to_dict())
        metric.process([], out)
        if (lo // batch) % 50 == 0:
            print(f"  {lo + len(items)}/{n}", file=sys.stderr, flush=True)
    results = metric.evaluate(size=n)
    pose.teardown()
    return {"results": {k: float(v) for k, v in results.items()},
            "n_instances": n, "seconds": round(time.perf_counter() - started, 1),
            "onnx": str(onnx), "engine": pose.engine_info.short_hash if pose.engine_info else None,
            "trt_version": pose.engine_info.trt_version if pose.engine_info else None}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cslr_cost.eval.coco_wholebody_trt",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--scale", required=True, choices=("t", "s", "m", "l"))
    p.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "int8"))
    p.add_argument("--weights", default="dw-ucoco", choices=tuple(CONFIG_FOR))
    p.add_argument("--onnx-dir", default="data/onnx")
    p.add_argument("--engine-dir", default="models/engines/rtx3060")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="first N instances: a smoke run, never a result")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    a = p.parse_args(argv)

    from cslr_cost.harness.schema import _git_dirty, _git_sha

    r = evaluate(a.scale, a.precision, a.weights, Path(a.onnx_dir), Path(a.engine_dir), a.batch, a.limit)
    row = {"schema": "coco-wholebody-trt/1",
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_sha": _git_sha(), "git_dirty": _git_dirty(),
           "extractor": f"rtmpose-{a.scale}", "weights": a.weights, "precision": a.precision,
           "boxes": "ground truth", "limit": a.limit, **r}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    ap = r["results"].get("coco-wholebody/AP")
    print(f"rtmpose-{a.scale} {a.weights} {a.precision}: whole-body AP {ap:.4f}  ({r['n_instances']} instances, "
          f"{r['seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

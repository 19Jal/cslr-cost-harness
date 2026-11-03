"""detect_trt.py — RTMDet-nano person detection as a TensorRT engine (stage 2).

The deployment form of `detect.py`'s `RtmdetPersonStage`: same `(frame, box)`
boundary, same whole-frame fallback with the same counter, same
`detector_policy` shape in the row. What differs is that the network runs as an
engine built on this device, and that the letterbox and the decode are written
out here instead of being done by mmdet's pipeline and `predict_by_feat`.

--------------------------------------------------------------------------
THE LETTERBOX IS THE DETECTOR'S COST, AND IT IS REPRODUCED EXACTLY
--------------------------------------------------------------------------
mmdet's test pipeline rescales the frame so its long edge is 320 (mmcv's
`rescale_size`, rounding with `int(x + 0.5)`), pads the remainder to 320x320
with 114, and normalises. All of it is charged to this stage, as it is in the
PyTorch adapter: it exists only because a detector is present.

Two details that are wrong by default rather than loudly:

  * RTMDet normalises in **BGR** -- `bgr_to_rgb=False`, mean
    (103.53, 116.28, 123.675) -- the opposite of RTMPose. Flipping channels
    here does not raise; it quietly moves every box.
  * The recorded scale factor is **per axis** (258/210, 320/260 for a PHOENIX
    frame), because width and height round independently. Rescaling boxes with
    one shared factor puts them a pixel or two out, which is exactly the size
    of error that looks like noise.

--------------------------------------------------------------------------
THE DECODE, AND WHY IT IS NOT MMDEPLOY'S
--------------------------------------------------------------------------
`sigmoid` on the class logits, distance-to-box against each level's stride grid
(`bbox_preds` are already pixel distances), per-level score filtering and
top-k, then NMS. That is mmdet's `_predict_by_feat_single` in ~30 lines, and it
is ours because the alternative -- MMDeploy's graph-side `NMS` -- is a
TensorRT plugin that has to be present on every target, which the ONNX/TensorRT split
exists to avoid. `tests/harness/test_detect_trt.py` holds this to mmdet's own
boxes on real frames.

NMS uses `torchvision.ops.nms`, which is already in the image and needs no
mmcv on the device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cslr_cost.harness.stages.base import Stage
from cslr_cost.harness.stages.detect import DEFAULT_SCORE_THR

#: The test pipeline letterboxes to a square of this side. Static in the graph.
INPUT_SIZE = 320

#: mmcv `Pad(pad_val=114)`, the value RTMDet was trained to see as "not image".
PAD_VALUE = 114

#: FPN strides, in the order the head emits its levels.
STRIDES = (8, 16, 32)

#: `data_preprocessor`: BGR order, no channel flip.
RTMDET_MEAN = (103.53, 116.28, 123.675)
RTMDET_STD = (57.375, 57.12, 58.395)

#: `test_cfg` from the config, reproduced rather than read, so the engine path
#: does not depend on mmdet being importable on the device.
NMS_PRE = 1000
NMS_IOU = 0.6
MAX_PER_IMG = 100
#: The detector's own floor, below which mmdet discards a candidate outright.
#: Distinct from the stage's `score_thr`, which decides whether the best
#: surviving box is trusted at all.
CANDIDATE_SCORE_THR = 0.05


def letterbox(frame: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Frame -> a 320x320 uint8 canvas, and the (x, y) scale factors applied.

    mmcv `Resize(scale=(320, 320), keep_ratio=True)` then `Pad(size=(320, 320),
    pad_val=114)`. The factors come back per axis because the resized width and
    height round independently.
    """
    import cv2

    h, w = frame.shape[:2]
    ratio = min(INPUT_SIZE / w, INPUT_SIZE / h)
    nw, nh = int(w * ratio + 0.5), int(h * ratio + 0.5)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, (nw / w, nh / h)


def grid_priors(size: int, stride: int, device: torch.device) -> torch.Tensor:
    """(H*W, 2) centres for one level. `MlvlPointGenerator(offset=0)`."""
    coords = torch.arange(size, device=device, dtype=torch.float32) * stride
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def decode_rtmdet(
    cls_scores: list[torch.Tensor],
    bbox_preds: list[torch.Tensor],
    *,
    nms_pre: int = NMS_PRE,
    score_thr: float = CANDIDATE_SCORE_THR,
    nms_iou: float = NMS_IOU,
    max_per_img: int = MAX_PER_IMG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-level (1, 1, H, W) and (1, 4, H, W) -> (N, 4) boxes, (N,) scores.

    Boxes are in the 320x320 letterbox frame, clipped to it, NMS'd, and sorted
    by score. `mmdet.dense_heads.base_dense_head._predict_by_feat_single`.
    """
    from torchvision.ops import nms

    boxes_all, scores_all = [], []
    for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, STRIDES):
        if cls_score.shape[0] != 1:
            raise ValueError("decode handles one image at a time, as mmdet does")
        size = cls_score.shape[-1]
        scores = cls_score.sigmoid().reshape(-1)                      # (H*W,)
        dist = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)           # l, t, r, b
        priors = grid_priors(size, stride, cls_score.device)

        keep = scores > score_thr
        scores, dist, priors = scores[keep], dist[keep], priors[keep]
        if scores.numel() > nms_pre:                                  # per level, as mmdet does
            scores, idx = scores.topk(nms_pre)
            dist, priors = dist[idx], priors[idx]

        boxes = torch.stack(
            (priors[:, 0] - dist[:, 0], priors[:, 1] - dist[:, 1],
             priors[:, 0] + dist[:, 2], priors[:, 1] + dist[:, 3]), dim=-1
        )
        boxes_all.append(boxes)
        scores_all.append(scores)

    boxes = torch.cat(boxes_all).clamp_(min=0, max=INPUT_SIZE)        # clip_border
    scores = torch.cat(scores_all)
    if scores.numel() == 0:
        return boxes, scores

    keep = nms(boxes, scores, nms_iou)[:max_per_img]
    return boxes[keep], scores[keep]


class TrtRtmdetPersonStage(Stage):
    """RTMDet-nano person detector as a TensorRT engine."""

    name = "detect"
    input_key = "frames"
    output_key = "boxes"
    granularity = "frame"
    uses_cuda = True

    def __init__(
        self,
        onnx_path: Path,
        engine_dir: Path,
        device: str = "cuda",
        score_thr: float = DEFAULT_SCORE_THR,
        model_name: str = "rtmdet-nano",
        precision: str = "fp32",
        opt_batch: int = 32,
        max_batch: int = 64,
    ) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(
                f"precision must be fp32 or fp16, got {precision!r}. INT8 needs a "
                "TensorRT calibration set, not a builder flag."
            )
        self.onnx_path = Path(onnx_path)
        self.engine_dir = Path(engine_dir)
        self.device = device
        self.score_thr = score_thr
        self.model_name = model_name
        self.precision = precision
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        self.module = None
        self.engine_info = None
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        #: Frames on which nothing scored above threshold and the whole frame
        #: was substituted. Reported, never swallowed.
        self.fallback_frames = 0
        self.frames_seen = 0

    def config(self) -> dict[str, Any]:
        info = self.engine_info
        return {
            "detector_policy": f"{self.model_name}/{self.precision}@{self.score_thr}",
            "detect_runtime": "tensorrt",
            "detect_engine": None if info is None else info.short_hash,
            "trt_version": None if info is None else info.trt_version,
        }

    @property
    def fallback_rate(self) -> Optional[float]:
        if not self.frames_seen:
            return None
        return self.fallback_frames / self.frames_seen

    # -- lifecycle -----------------------------------------------------------

    def setup(self) -> None:
        from cslr_cost.harness.engines import TrtModule, ensure_engine

        one = (1, 3, INPUT_SIZE, INPUT_SIZE)
        profiles = [
            {"input": (one, one, one)},
            {"input": ((2, 3, INPUT_SIZE, INPUT_SIZE),
                       (self.opt_batch, 3, INPUT_SIZE, INPUT_SIZE),
                       (self.max_batch, 3, INPUT_SIZE, INPUT_SIZE))},
        ]
        self.engine_info = ensure_engine(self.onnx_path, self.engine_dir, self.precision, profiles)
        self.module = TrtModule(self.engine_info.path, device=self.device)

        expected = {"cls_8", "bbox_8", "cls_16", "bbox_16", "cls_32", "bbox_32"}
        got = set(self.module.output_names)
        if got != expected:
            raise ValueError(
                f"{self.onnx_path.name} emits {sorted(got)}; expected per-level "
                f"{sorted(expected)}. A graph that already decodes would be decoded twice."
            )

        self._mean = torch.tensor(RTMDET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(RTMDET_STD, device=self.device).view(1, 3, 1, 1)

    def teardown(self) -> None:
        self.module = None
        self._mean = None
        self._std = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- the timed unit ------------------------------------------------------

    @torch.no_grad()
    def run(self, frame: Any) -> tuple[np.ndarray, np.ndarray]:
        if self.module is None:
            raise RuntimeError("setup() must run before the detector is measured")

        self.frames_seen += 1
        canvas, (sx, sy) = letterbox(frame)

        # uint8 up, normalise on device: the same bandwidth argument as pose.py,
        # and here the channel order stays BGR.
        x = torch.from_numpy(np.ascontiguousarray(canvas)).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1).unsqueeze(0).float()
        x = (x - self._mean) / self._std

        out = self.module(input=x)
        boxes, scores = decode_rtmdet(
            [out["cls_8"], out["cls_16"], out["cls_32"]],
            [out["bbox_8"], out["bbox_16"], out["bbox_32"]],
        )

        if scores.numel() == 0 or float(scores[0]) <= self.score_thr:
            self.fallback_frames += 1
            h, w = frame.shape[:2]
            return frame, np.array([0.0, 0.0, w, h], dtype=np.float32)

        best = boxes[int(scores.argmax())].detach().cpu().numpy()
        # letterbox space -> frame pixels; per axis, because the two rounded apart
        best = best / np.array([sx, sy, sx, sy], dtype=np.float32)
        return frame, best.astype(np.float32)

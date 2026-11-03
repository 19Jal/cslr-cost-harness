"""twostream_trt.py — TwoStream-SLR (RGB + keypoints) in the harness's seven slots.

One harness, one input: TwoStream-SLR's pipeline includes its pose stage and heatmap
rendering. Its keypoints are HRNet COCO-WholeBody (the same front end as
MSKA's released keypoints), so the
pose slot runs the same HRNet-w48 DARK + flip engine the MSKA baseline uses.

    slot       TwoStream-SLR                                              granularity
    decode     the same video decode                                       frame
    detect     whole frame                                                 frame
    crop       HRNet's 288x384 crop; the frame passes alongside            frame
    pose       HRNet-w48 DARK + flip (TensorRT) -> 133 keypoints, and      frame
               the frame resized to 224x224 on the GPU (TwoStream's RGB
               input: bilinear, /255, (x - 0.5) / 0.5, BGR)
    postproc   79 joints, scaled to the 210x260 frame TwoStream was        frame
               trained on, rendered as Gaussians (sigma 8) at 260x210,
               resized to 112x112, (h - 0.5) / 0.5  (GPU)
    head       both S3D streams + three heads, ensemble_last (TensorRT)    clip
    ctc        beam 5, TwoStream's (= MSKA's) decoding                     clip

`export/export_twostream_onnx.py` checks both preprocessing paths against
TwoStream's own code (max |d| 0) and the graph against its modules.

**The keypoint stream alone** (`twostream-kp`, `keypoint_only=True`): the same slots
without the RGB path. Pose is HRNet only, postproc renders the same heatmaps, and the
head is the keypoint S3D with its one head, decoded from its log_softmax as TwoStream
decodes a single stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cslr_cost.harness.stages.base import Stage

#: sorted(use_keypoints) = face_others_1_3, hand, mouth_half, pose (TwoStream's Dataset.py)
JOINTS_79 = (list(range(23, 71))[::3] + list(range(91, 133)) + list(range(71, 91))[::2] + list(range(11)))
RAW_HW = (260, 210)          # TwoStream's heatmap_cfg.raw_size: the frame it was trained on
HEATMAP = 112
SIGMA = 8.0
RGB = 224
VOCAB = 1235


class TwoStreamCropStage(Stage):
    """(frame, box) -> (HRNet crop, frame)."""

    name = "crop"
    input_key = "boxes"
    output_key = "crops"
    granularity = "frame"
    uses_cuda = False

    def __init__(self) -> None:
        from cslr_cost.harness.stages.crop import TopdownCropStage
        from cslr_cost.harness.stages.pose import crop_size_for

        self.inner = TopdownCropStage(input_size=crop_size_for("hrnet-w48"))

    def config(self) -> dict[str, Any]:
        return self.inner.config()

    def setup(self) -> None:
        self.inner.setup()

    def run(self, boxes: Any) -> tuple:
        return self.inner.run(boxes), boxes[0]


class TwoStreamPoseStage(Stage):
    """(crop, frame) -> (133 keypoints in frame pixels, the frame as TwoStream's RGB input)."""

    name = "pose"
    input_key = "crops"
    output_key = "keypoints_raw"
    granularity = "frame"
    uses_cuda = True
    supports_batching = True

    def __init__(self, onnx_dir: Path, engine_dir: Path, precision: str = "fp32", device: str = "cuda",
                 opt_batch: int = 32, max_batch: int = 64, rgb: bool = True) -> None:
        from cslr_cost.harness.stages.pose_trt import make_trt_pose_stage

        self.hrnet = make_trt_pose_stage(Path(onnx_dir), "hrnet-w48", precision, Path(engine_dir),
                                         opt_batch=opt_batch, max_batch=max_batch)
        self.device = device
        self.rgb = rgb                               # False: the keypoint stream alone

    def config(self) -> dict[str, Any]:
        cfg = dict(self.hrnet.config())
        cfg["frontend_model"] = "twostream-hrnet-w48-dark-flip" + ("+rgb224" if self.rgb else "")
        return cfg

    def setup(self) -> None:
        self.hrnet.setup()

    def teardown(self) -> None:
        self.hrnet.teardown()

    def _rgb(self, frames: list[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames)).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float() / 255
        x = torch.nn.functional.interpolate(x, size=(RGB, RGB), mode="bilinear", align_corners=False)
        return (x - 0.5) / 0.5                       # BGR, as TwoStream feeds its network

    @torch.no_grad()
    def run(self, unit: Any) -> tuple:
        crop, frame = unit
        return self.hrnet.run(crop), (self._rgb([frame])[0] if self.rgb else None)

    @torch.no_grad()
    def run_batch(self, units: Any) -> list[tuple]:
        units = list(units)
        kps = self.hrnet.run_batch([u[0] for u in units])
        if not self.rgb:
            return [(k, None) for k in kps]
        rgb = self._rgb([u[1] for u in units])
        return list(zip(kps, rgb))


def render_heatmaps(kp: torch.Tensor) -> torch.Tensor:
    """(N, 79, 2+) keypoints in 210x260 pixels -> (N, 79, 112, 112), normalised.
    TwoStream's gen_gaussian_hmap_op at raw size, then a bilinear resize."""
    H, W = RAW_HW
    y = torch.arange(H, dtype=torch.float32, device=kp.device).view(1, 1, H, 1)
    x = torch.arange(W, dtype=torch.float32, device=kp.device).view(1, 1, 1, W)
    cy, cx = kp[..., 1, None, None], kp[..., 0, None, None]
    hm = torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * SIGMA ** 2))
    hm = torch.nn.functional.interpolate(hm, size=(HEATMAP, HEATMAP), mode="bilinear", align_corners=False)
    return (hm - 0.5) / 0.5


class TwoStreamPostprocStage(Stage):
    """(keypoints, rgb) -> (rgb, heatmaps): joint selection, rescaling, heatmap rendering."""

    name = "postproc"
    input_key = "keypoints_raw"
    output_key = "keypoints"
    granularity = "frame"
    uses_cuda = True

    def __init__(self, frame_wh: tuple[int, int], device: str = "cuda") -> None:
        w, h = frame_wh
        self.scale = torch.tensor([RAW_HW[1] / w, RAW_HW[0] / h], device=device)
        self.idx = torch.tensor(JOINTS_79, device=device)
        self.device = device

    @torch.no_grad()
    def run(self, unit: Any) -> tuple:
        kp, rgb = unit
        k = torch.as_tensor(kp, device=self.device, dtype=torch.float32)[self.idx, :2] * self.scale
        return rgb, render_heatmaps(k[None])[0]


def assemble_twostream(units: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    """-> rgb (1, 3, T, 224, 224), heatmaps (1, 79, T, 112, 112); T trimmed to a multiple
    of 4, as TwoStream's evaluation selects frames (`get_selected_indexs`). rgb is None
    for the keypoint stream alone."""
    t = len(units) - len(units) % 4
    if t == 0:
        raise ValueError("TwoStream needs at least 4 frames")
    hm = torch.stack([u[1] for u in units[:t]], dim=1)[None]
    if units[0][0] is None:
        return None, hm.contiguous()
    rgb = torch.stack([u[0] for u in units[:t]], dim=1)[None]
    return rgb.contiguous(), hm.contiguous()


class TrtTwoStreamHeadStage(Stage):
    """Both S3D streams and the three heads as one TensorRT engine -> (1, T/4, 1235).
    `keypoint_only`: the keypoint S3D and its one head (`twostream-kp.onnx`)."""

    name = "head"
    input_key = "sequence"
    output_key = "gloss_logits"
    granularity = "clip"
    uses_cuda = True

    def __init__(self, onnx_path: Path, engine_dir: Path, precision: str = "fp32", device: str = "cuda",
                 min_frames: int = 16, opt_frames: int = 128, max_frames: int = 300,
                 keypoint_only: bool = False) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision fp32 or fp16, got {precision!r}")
        self.keypoint_only = keypoint_only
        self.onnx_path, self.engine_dir = Path(onnx_path), Path(engine_dir)
        self.precision, self.device = precision, device
        self.frames = (min_frames, opt_frames, max_frames)
        self.module = None
        self.engine_info = None

    @property
    def vocab_size(self) -> int:
        return VOCAB

    def config(self) -> dict[str, Any]:
        info = self.engine_info
        return {"head_model": "twostream-kp" if self.keypoint_only else "twostream-slr", "head_precision": self.precision, "head_runtime": "tensorrt",
                "head_engine": None if info is None else info.short_hash,
                "trt_version": None if info is None else info.trt_version}

    def setup(self) -> None:
        from cslr_cost.harness.engines import TrtModule, ensure_engine

        lo, opt, hi = self.frames
        profiles = [{"heatmaps": tuple((1, len(JOINTS_79), n, HEATMAP, HEATMAP) for n in (lo, opt, hi))}]
        if not self.keypoint_only:
            profiles[0]["rgb"] = tuple((1, 3, n, RGB, RGB) for n in (lo, opt, hi))
        self.engine_info = ensure_engine(self.onnx_path, self.engine_dir, self.precision, profiles)
        self.module = TrtModule(self.engine_info.path, device=self.device)

    def teardown(self) -> None:
        self.module = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self, sequence: Any) -> torch.Tensor:
        rgb, hm = sequence
        if self.keypoint_only:
            out = self.module(heatmaps=hm.float().contiguous())
            return out["logits_keypoint"].float().log_softmax(-1)       # a single stream's decoding
        out = self.module(rgb=rgb.float().contiguous(), heatmaps=hm.float().contiguous())
        # ensemble_last in float32, outside the engine (FP16 softmax underflows; head_trt.py)
        return sum(out[n].float().softmax(-1) for n in ("logits_rgb", "logits_keypoint", "logits_fuse")).log()

"""rgb_trt.py — an RGB CSLR pipeline (VAC + SMKD, CorrNet) in the harness's seven slots.

One harness, one input. An RGB model has no
detector and no pose stage; its pipeline is decode -> resize/crop -> 2D CNN per
frame -> temporal module -> CTC. It is fitted to the harness's slots so every
runner, energy window, reconciliation and report applies unchanged:

    slot       RGB pipeline                                   granularity
    decode     the same video decode                          frame
    detect     whole frame (the box is the frame; ~free)      frame
    crop       Lanczos-4 resize to 256x256, centre crop 224   frame
    pose       the per-frame CNN (TensorRT)                   frame
               VAC: ResNet18 -> 512; CorrNet: up to layer 2
    postproc   identity                                       frame
    head       temporal part (TensorRT): CorrNet's            clip
               correlation layers, then K5 P2 K5 P2, BiLSTM,
               classifier
    ctc        ctcdecode beam 10, pruned to the top 40        clip
               classes as both repositories decode

Rows say what ran: `frontend_model` is `vac-smkd-resnet18` / `corrnet-resnet18-l2`,
never a pose model, and a report reading the `pose` row of these runs must
call it the CNN.

Boundary choices (harness, Chunk 5): BGR->RGB and x/127.5 - 1 run on the GPU
inside the CNN stage, as RTMPose's do (uint8 crosses the bus, not float32).
Per-frame features stay on the GPU; the clip's padding (6 copies of the first
frame's features before, 6 + round-up-to-4 of the last after) is applied at
assembly, which equals padding the frames because the per-frame network is
frame-independent (checked on the dev box: every argmax agrees).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from cslr_cost.harness.stages.base import Stage

RESIZE = 256
CROP = 224
LEFT_PAD = 6
VOCAB = 1296

FRONTEND_LABEL = {"vac": "vac-resnet18", "vac-smkd": "vac-smkd-resnet18", "corrnet": "corrnet-resnet18-l2"}
HEAD_LABEL = {"vac": "vac", "vac-smkd": "vac-smkd", "corrnet": "corrnet"}


class RgbResizeCropStage(Stage):
    """(frame, box) -> uint8 BGR (224, 224, 3): VAC's preprocessing at test time."""

    name = "crop"
    input_key = "boxes"
    output_key = "crops"
    granularity = "frame"
    uses_cuda = False

    def config(self) -> dict[str, Any]:
        return {"notes": f"crop: resize {RESIZE}x{RESIZE} lanczos4, centre crop {CROP}"}

    def run(self, boxes: Any) -> np.ndarray:
        frame, _box = boxes
        img = cv2.resize(frame, (RESIZE, RESIZE), interpolation=cv2.INTER_LANCZOS4)
        o = (RESIZE - CROP) // 2
        return np.ascontiguousarray(img[o:o + CROP, o:o + CROP])


class IdentityPostprocStage(Stage):
    """Nothing to do between an RGB CNN and its head; kept so the slot exists."""

    name = "postproc"
    input_key = "keypoints_raw"
    output_key = "keypoints"
    granularity = "frame"
    uses_cuda = False

    def run(self, x: Any) -> Any:
        return x


def assemble_features(units: list[torch.Tensor]) -> torch.Tensor:
    """Per-frame features -> (1, T_padded, ...), padded as VAC / CorrNet pad frames."""
    if not units:
        raise ValueError("cannot assemble an empty sequence")
    f = torch.stack(units)
    t = len(f)
    right = -(-t // 4) * 4 - t + LEFT_PAD
    return torch.cat([f[:1].expand(LEFT_PAD, *f.shape[1:]), f,
                      f[-1:].expand(right, *f.shape[1:])])[None].contiguous()


class TrtRgbFrontendStage(Stage):
    """The per-frame CNN as a TensorRT engine: uint8 BGR crop -> features (on GPU)."""

    name = "pose"
    input_key = "crops"
    output_key = "keypoints_raw"
    granularity = "frame"
    uses_cuda = True
    supports_batching = True

    def __init__(self, onnx_path: Path, engine_dir: Path, model: str, precision: str = "fp32",
                 device: str = "cuda", opt_batch: int = 32, max_batch: int = 64) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision fp32 or fp16, got {precision!r}")
        self.onnx_path, self.engine_dir = Path(onnx_path), Path(engine_dir)
        self.model, self.precision, self.device = model, precision, device
        self.opt_batch, self.max_batch = opt_batch, max_batch
        self.module = None
        self.engine_info = None

    def config(self) -> dict[str, Any]:
        info = self.engine_info
        return {
            "frontend_model": FRONTEND_LABEL[self.model],
            "frontend_checkpoint": None if info is None else f"{self.onnx_path.stem}@{info.onnx_sha256[:12]}",
            "frontend_input": f"{CROP}x{CROP}",
            "frontend_precision": self.precision,
            "frontend_runtime": "tensorrt",
            "frontend_engine": None if info is None else info.short_hash,
            "trt_version": None if info is None else info.trt_version,
        }

    def setup(self) -> None:
        from cslr_cost.harness.engines import TrtModule, ensure_engine

        one = (1, 3, CROP, CROP)
        profiles = [{"input": (one, one, one)},
                    {"input": ((2, 3, CROP, CROP), (self.opt_batch, 3, CROP, CROP),
                               (self.max_batch, 3, CROP, CROP))}]
        self.engine_info = ensure_engine(self.onnx_path, self.engine_dir, self.precision, profiles)
        self.module = TrtModule(self.engine_info.path, device=self.device)

    def teardown(self) -> None:
        self.module = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _infer(self, crops: list[np.ndarray]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("setup() must run before the front end is measured")
        # FP32 engines refuse odd batches from 3 up ("Cask: Failed to update runtime
        # arguments"), as HRNet's did (pose_trt.py): split off the last crop.
        if len(crops) > 1 and len(crops) % 2 == 1:
            return torch.cat([self._infer(crops[:-1]), self._infer(crops[-1:])])
        x = torch.from_numpy(np.stack(crops)).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2)[:, [2, 1, 0]].float() / 127.5 - 1.0
        return self.module(input=x.contiguous())["features"].float()

    def run(self, crop: Any) -> torch.Tensor:
        return self._infer([crop])[0]

    def run_batch(self, crops: Any) -> list[torch.Tensor]:
        crops = list(crops)
        out: list[torch.Tensor] = []
        for i in range(0, len(crops), self.max_batch):
            out += list(self._infer(crops[i:i + self.max_batch]))
        return out


class TrtRgbHeadStage(Stage):
    """The clip-level temporal part as a TensorRT engine: (1, T, ...) -> (1, T', 1296) logits."""

    name = "head"
    input_key = "sequence"
    output_key = "gloss_logits"
    granularity = "clip"
    uses_cuda = True

    def __init__(self, onnx_path: Path, engine_dir: Path, model: str, feature_shape: tuple[int, ...],
                 precision: str = "fp32", device: str = "cuda",
                 min_frames: int = 24, opt_frames: int = 140, max_frames: int = 320) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision fp32 or fp16, got {precision!r}")
        self.onnx_path, self.engine_dir = Path(onnx_path), Path(engine_dir)
        self.model, self.precision, self.device = model, precision, device
        self.feature_shape = tuple(feature_shape)
        self.frames = (min_frames, opt_frames, max_frames)
        self.module = None
        self.engine_info = None

    @property
    def vocab_size(self) -> int:
        return VOCAB

    def config(self) -> dict[str, Any]:
        info = self.engine_info
        return {
            "head_model": HEAD_LABEL[self.model],
            "head_precision": self.precision,
            "head_runtime": "tensorrt",
            "head_engine": None if info is None else info.short_hash,
            "trt_version": None if info is None else info.trt_version,
        }

    def setup(self) -> None:
        from cslr_cost.harness.engines import TrtModule, ensure_engine

        lo, opt, hi = self.frames
        profiles = [{"features": tuple((1, n, *self.feature_shape) for n in (lo, opt, hi))}]
        self.engine_info = ensure_engine(self.onnx_path, self.engine_dir, self.precision, profiles)
        self.module = TrtModule(self.engine_info.path, device=self.device)

    def teardown(self) -> None:
        self.module = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self, sequence: Any) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("setup() must run before run(); engine loading is not timed")
        x = sequence if torch.is_tensor(sequence) else torch.as_tensor(sequence)
        return self.module(features=x.to(self.device, torch.float32).contiguous())["logits"].float()[None]

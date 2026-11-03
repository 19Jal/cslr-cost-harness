"""pose.py — RTMPose whole-body keypoint extraction (stage 4).

The stage the field leaves out. Keypoint CSLR papers report the cost of the
recognition head on stored keypoints; the pose estimator that produced them runs
on every frame of a deployed pipeline and is usually its largest cost. Measuring this
stage properly is the point of the harness, so it gets the most careful
adapter.

Per-frame, CUDA.

--------------------------------------------------------------------------
WHY NOT `inference_topdown`
--------------------------------------------------------------------------
mmpose's one-call API re-does the whole front end: it loads the image, computes
the box centre/scale, warps the crop, normalises, runs the model and decodes.
Calling it here would charge `crop`'s work to `pose` and measure the warp
twice -- once in `crop`, once inside this stage. The per-stage table would then
double-count preprocessing and overstate the extractor, and so bias every
per-stage share. So this adapter takes the already
warped `Crop` and runs only:

    normalise (GPU) -> backbone+neck -> SimCC head -> argmax decode -> unwarp

which is what an ONNX/TensorRT deployment of RTMPose would execute.

--------------------------------------------------------------------------
THE COLOUR CONVERSION AND NORMALISATION ARE CHARGED HERE, ON DEVICE
--------------------------------------------------------------------------
RTMPose's `PoseDataPreprocessor` carries `bgr_to_rgb=True` and mean/std, and
runs on the GPU after the upload. Reproduced explicitly rather than called
through mmengine's dict API, so that what crosses PCIe is visibly **uint8**:
uploading normalised float32 instead would quadruple the transfer for no
benefit, and on a Jetson -- where "PCIe" is shared memory bandwidth -- that is
not a rounding error.

--------------------------------------------------------------------------
KEYPOINTS COME BACK IN CROP SPACE AND MUST BE UNWARPED
--------------------------------------------------------------------------
SimCC decodes to the 192x256 canvas. `postproc` expects source-frame pixels,
because that is what MSKA's released HRNet keypoints are and what its
normalisation divides by. The mapping is mmpose's own, from
`TopdownPoseEstimator.add_pred_to_datasample`:

    keypoints = keypoints / input_size * input_scale
                + input_center - 0.5 * input_scale

Skipping it does not raise: the keypoints stay finite and roughly plausible,
`postproc` divides them by the frame size anyway, and the head produces glosses
at a WER that is worse for reasons unrelated to anything being studied. This is
the same failure class as the normalisation trap in `postproc.py`, and it is
why both transforms are written out with their source cited rather than
inherited from a convenience wrapper.

--------------------------------------------------------------------------
MODEL SCALE AND PRECISION ARE MEASURED AXES
--------------------------------------------------------------------------
t / s / m are all fetched, because the pose sweep varies front-end capacity against
WER: the interesting result is where the accuracy knee sits relative to the
head's own cost. `precision` moves fp32/fp16 here; INT8 is a TensorRT concern
and comes from the ONNX/TensorRT path, not from a `.half()` call.

--------------------------------------------------------------------------
TWO RUNTIMES, ONE ADAPTER
--------------------------------------------------------------------------
Everything above -- the uint8 upload, the on-device normalisation, the unwarp
-- lives in `_RtmposeBase`, shared by this PyTorch stage and the TensorRT
stage in `pose_trt.py`. Only the model call and the decode differ between
them, so a runtime comparison measures the runtime, not two independently
written preprocessing paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cslr_cost.harness.stages.base import Stage

#: RTMPose's input canvas, (h, w). Defined here rather than in `pose_trt`
#: because both stages need it and `pose_trt` already imports from this
#: module — putting it the other way round is a circular import.
RTMPOSE_INPUT_HW = (256, 192)

#: Input canvas per scale, (h, w). t/s/m share 256x192; `l` is DWPose-l at
#: 384x288 -- a second cost regime, since the crop is the same size at every
#: ladder rung and 256x192 alone would cover only one.
INPUT_HW = {"t": (256, 192), "s": (256, 192), "m": (256, 192), "l": (384, 288),
            # HRNet-w48 DARK, the front end behind MSKA's released keypoints
            # (TensorRT only: `pose_trt.TrtHrnetStage`).
            "hrnet-w48": (384, 288)}


def input_hw_for(scale: str) -> tuple[int, int]:
    """(h, w) of the model input for this scale."""
    if scale not in INPUT_HW:
        raise ValueError(f"unknown pose model {scale!r}; known: {sorted(INPUT_HW)}")
    return INPUT_HW[scale]


def crop_size_for(scale: str) -> tuple[int, int]:
    """(w, h), the order `TopdownCropStage` and mmpose's `input_size` use."""
    h, w = input_hw_for(scale)
    return (w, h)

#: Whole-body layout size. MSKA's head consumes 133; see postproc.py on the
#: open question of whether index i means the same joint in both layouts.
WHOLEBODY_KEYPOINTS = 133

#: Checkpoints, keyed by the scale axis the pose sweep varies.
RTMPOSE_SCALES = ("t", "s", "m", "l")


class _RtmposeBase(Stage):
    """What every RTMPose runtime shares: the boundary, preprocessing, unwarp."""

    name = "pose"
    input_key = "crops"
    output_key = "keypoints_raw"
    granularity = "frame"
    uses_cuda = True

    #: What this runtime can actually execute. Subclasses widen it; the base is
    #: the torch path, where INT8 does not exist -- it is an exported QDQ graph
    #: compiled by TensorRT, not a torch dtype.
    SUPPORTED_PRECISIONS = ("fp32", "fp16")

    def __init__(self, scale: str = "t", device: str = "cuda", precision: str = "fp32") -> None:
        if precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(
                f"{type(self).__name__} supports {self.SUPPORTED_PRECISIONS}, got "
                f"{precision!r}. INT8 is an exported QDQ graph compiled by "
                "TensorRT (export/quantize_rtmpose_int8.py) — on the torch path "
                "there is no calibration set and no int8 kernel to run."
            )
        self.scale = scale
        self.device = device
        self.precision = precision
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._bgr_to_rgb = True

    # -- the timed unit ------------------------------------------------------

    @torch.no_grad()
    def run(self, crop: Any) -> np.ndarray:
        """One `Crop` -> (133, 3) keypoints as (x, y, score) in frame pixels."""
        return self._infer([crop])[0]

    @torch.no_grad()
    def run_batch(self, crops: Any) -> list[np.ndarray]:
        """P7's batch regime: one forward for the whole batch.

        This is where batch throughput actually comes from. `decode` is serial
        and `crop`'s warp is per-image, so `pose` is the only front-end stage
        whose batch number differs from its streaming number -- which is worth
        stating plainly, because "batch 32 throughput" quoted for a pipeline
        usually credits the whole pipeline for one stage's GPU occupancy.
        """
        return self._infer(list(crops))

    supports_batching = True

    # -- shared implementation -----------------------------------------------

    def _infer(self, crops: list[Any]) -> list[np.ndarray]:
        raise NotImplementedError

    def _preprocess(self, crops: list[Any], dtype: torch.dtype) -> torch.Tensor:
        """uint8 HWC crops -> a normalised NCHW batch on device.

        Deliberately uploaded as uint8 and converted on the GPU: sending
        float32 would quadruple the transfer for no benefit, and on a Jetson
        that is shared memory bandwidth rather than a rounding error.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("setup() must run before the extractor is measured")
        imgs = []
        for crop in crops:
            img = crop.img
            if img.ndim != 3:
                raise ValueError(f"expected one crop (H, W, 3), got {img.shape}")
            imgs.append(np.ascontiguousarray(img))

        x = torch.from_numpy(np.stack(imgs)).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2)
        if self._bgr_to_rgb:
            x = x[:, [2, 1, 0], ...]

        return (x.to(dtype) - self._mean.to(dtype)) / self._std.to(dtype)

    def _to_frame_coords(self, kpts: Any, scores: Any, crop: Any) -> np.ndarray:
        """Canvas (K, 2) + (K,) -> (K, 3) keypoints in source-frame pixels."""
        kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        # crop space -> source frame pixels
        # (TopdownPoseEstimator.add_pred_to_datasample)
        input_size = np.asarray(crop.input_size, dtype=np.float32)
        kpts = kpts / input_size * crop.scale + crop.center - 0.5 * crop.scale

        if kpts.shape[0] != WHOLEBODY_KEYPOINTS:
            raise ValueError(
                f"extractor returned {kpts.shape[0]} keypoints, expected "
                f"{WHOLEBODY_KEYPOINTS}. MSKA's DSTA indexes groups out of the "
                "full whole-body layout, so a reduced set indexes out of bounds "
                "and surfaces as CUDNN_STATUS_INTERNAL_ERROR far downstream."
            )

        return np.concatenate([kpts, scores[:, None]], axis=1).astype(np.float32)


class RtmposeStage(_RtmposeBase):
    """RTMPose whole-body extractor, PyTorch inference path only."""

    def __init__(
        self,
        config_path: Path,
        checkpoint: Path,
        scale: str = "t",
        device: str = "cuda",
        precision: str = "fp32",
    ) -> None:
        super().__init__(scale=scale, device=device, precision=precision)
        self.config_path = Path(config_path)
        self.checkpoint = Path(checkpoint)
        self.model = None

    def config(self) -> dict[str, Any]:
        return {
            "frontend_model": f"rtmpose-{self.scale}",
            # Which weights, not which family — see harness/schema.py. The
            # mmpose config stem names the training set (`coco-wholebody` vs
            # `coco-ubody-wholebody`), which the checkpoint filename does not.
            "frontend_checkpoint": Path(self.config_path).stem,
            "frontend_input": "{}x{}".format(*input_hw_for(self.scale)),
            "frontend_precision": self.precision,
            "frontend_runtime": "pytorch",
        }

    # -- lifecycle -----------------------------------------------------------

    def setup(self) -> None:
        from mmpose.apis import init_model

        model = init_model(str(self.config_path), str(self.checkpoint), device=self.device)
        model.eval()
        if self.precision == "fp16":
            model.half()

        dp = model.data_preprocessor
        # Buffers are (3, 1, 1) in the order the config declares, and the
        # channel flip happens *before* normalising -- so mean/std are in RGB
        # order when bgr_to_rgb is set. Read from the model rather than
        # hardcoded: a different checkpoint may well normalise differently, and
        # a wrong mean does not raise, it just degrades keypoints.
        self._mean = dp.mean.view(1, 3, 1, 1).to(self.device)
        self._std = dp.std.view(1, 3, 1, 1).to(self.device)
        self._bgr_to_rgb = bool(getattr(dp, "_channel_conversion", True))

        self.model = model

    def teardown(self) -> None:
        self.model = None
        self._mean = None
        self._std = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _infer(self, crops: list[Any]) -> list[np.ndarray]:
        if self.model is None:
            raise RuntimeError("setup() must run before the extractor is measured")
        if not crops:
            return []

        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        x = self._preprocess(crops, dtype)
        feats = self.model.extract_feat(x)
        simcc_x, simcc_y = self.model.head.forward(feats)

        # SimCC's argmax decode is numpy-side in mmpose; float16 logits would
        # be decoded at half precision for no benefit, so cast back first.
        instances = self.model.head.decode((simcc_x.float(), simcc_y.float()))
        return [
            self._to_frame_coords(inst.keypoints, inst.keypoint_scores, crop)
            for inst, crop in zip(instances, crops)
        ]

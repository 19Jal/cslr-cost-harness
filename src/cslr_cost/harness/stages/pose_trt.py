"""pose_trt.py — RTMPose from its ONNX export, run as a TensorRT engine (stage 4).

The deployment form of `pose.py`'s extractor, and the one the cost tables are
built from. Same stage contract, same `Crop` in and `(133, 3)` frame-pixel
keypoints out, same uint8 upload and on-GPU normalisation, same unwarp. What
runs between preprocessing and decoding is an engine compiled on this device
from `rtmpose-{t,s,m}.onnx` (backbone + neck + SimCC head, emitting
the two SimCC logit tensors).

--------------------------------------------------------------------------
THE SIMCC DECODE IS PORTED, AND HELD TO MMPOSE BY A TEST
--------------------------------------------------------------------------
The export stops at logits, so the argmax decode is ours. mmpose's
`SimCCLabel.decode` is not called: the point of the ONNX path is to run where
OpenMMLab is not installed. The port
is fifteen lines, and `tests/harness/test_pose_trt.py` holds it to mmpose's own
output on the dev box, where both exist -- "verify the adapter
reproduces the SDK output while its Python path is still available".

The settings it hardcodes are identical for t/s/m (checked against all three
configs, and re-checked by that test): `simcc_split_ratio = 2.0`,
`use_dark = False`, no visibility decode. DARK refinement would decode
differently without raising, so this port offers no DARK option at all rather
than an approximate one.

--------------------------------------------------------------------------
TWO OPTIMISATION PROFILES, ONE PER REGIME
--------------------------------------------------------------------------
Streaming and isolation run batch 1; P7's batch regime runs batch N. TensorRT
picks kernels for a profile's `opt` shape, so one engine carries a static
batch-1 profile and a batch profile tuned at `opt_batch`, and each call is
routed by its batch size (`engines.TrtModule`). A single batch-32-tuned profile
would make every streaming row a measurement of kernels nobody deploys for
streaming.

--------------------------------------------------------------------------
PRECISION
--------------------------------------------------------------------------
`fp16` is a builder flag here, not a `.half()`: TensorRT chooses per-layer
precision and keeps the engine's I/O in float32. So the upload is normalised to
float32 in both arms, unlike the PyTorch stage's fp16 arm, and the difference
is inside the engine where it belongs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cslr_cost.harness.stages.pose import RTMPOSE_INPUT_HW, _RtmposeBase, input_hw_for

#: RTMPose's network input, (H, W). The ONNX files declare [batch, 3, 256, 192].

#: `PoseDataPreprocessor` mean / std, RGB order (applied after bgr_to_rgb).
#: `pose.py` reads these off the model; with no model object here they are the
#: configs' values, pinned, and test-checked against every scale's config.
RTMPOSE_MEAN = (123.675, 116.28, 103.53)
RTMPOSE_STD = (58.395, 57.12, 57.375)

#: `SimCCLabel(simcc_split_ratio=2.0)`: the SimCC canvas is 2x the input.
SIMCC_SPLIT_RATIO = 2.0

#: Batch profile defaults: tuned at the ladder's `--batch`, capped at the top
#: of the batch-size sweep.
DEFAULT_OPT_BATCH = 32
DEFAULT_MAX_BATCH = 64


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(N, K, Wx), (N, K, Wy) SimCC logits -> (N, K, 2) coords, (N, K) scores.

    mmpose `get_simcc_maximum(apply_softmax=False)` followed by
    `keypoints /= simcc_split_ratio` (`SimCCLabel.decode`, use_dark=False).
    Coordinates are on the 192x256 input canvas; the score is the smaller of
    the two axes' peak logits, and a non-positive score sets the location to
    -1 before the split-ratio division, exactly as mmpose does.
    """
    if simcc_x.ndim != 3 or simcc_y.ndim != 3 or simcc_x.shape[:2] != simcc_y.shape[:2]:
        raise ValueError(f"expected (N, K, W) pairs, got {simcc_x.shape} and {simcc_y.shape}")
    n, k, _ = simcc_x.shape
    x = simcc_x.reshape(n * k, -1)
    y = simcc_y.reshape(n * k, -1)

    locs = np.stack((np.argmax(x, axis=1), np.argmax(y, axis=1)), axis=-1).astype(np.float32)
    vals = np.minimum(np.amax(x, axis=1), np.amax(y, axis=1))
    locs[vals <= 0.0] = -1
    locs /= SIMCC_SPLIT_RATIO
    return locs.reshape(n, k, 2), vals.reshape(n, k)


class TrtRtmposeStage(_RtmposeBase):
    """RTMPose whole-body extractor as a TensorRT engine."""

    #: TensorRT compiles the QDQ export, so this runtime has the int8 arm.
    SUPPORTED_PRECISIONS = ("fp32", "fp16", "int8")

    #: RTMPose's head normalises with ScaleNorm, x / ||x||: Abs -> Pow(2) ->
    #: ReduceSum -> Pow(0.5) over the feature vector. In FP16 the sum of squares
    #: can pass 65504. It did for DWPose-l at 384x288 (2026-10-02): keypoints
    #: bit-identical on most frames but 17% of frames with a joint >50 px off,
    #: confidence median 0.84 -> 0.70, -6.7 AP on COCO, 21.97 -> 42.74 WER.
    #: t/s/m FP16 measured lossless (<=0.001 AP), so their engines -- and every
    #: row measured with them -- are left exactly as they were: pinned for `l`.
    FP32_NORM_LAYERS = ("/mlp/mlp.0/", "/gau/ln/")
    PINNED_SCALES = ("l",)

    def __init__(
        self,
        onnx_path: Path,
        engine_dir: Path,
        scale: str = "t",
        device: str = "cuda",
        precision: str = "fp32",
        opt_batch: int = DEFAULT_OPT_BATCH,
        max_batch: int = DEFAULT_MAX_BATCH,
        checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__(scale=scale, device=device, precision=precision)
        #: Which weights this ONNX was exported from. See `config()`.
        self.checkpoint = checkpoint
        if not 2 <= opt_batch <= max_batch:
            raise ValueError(f"need 2 <= opt_batch <= max_batch, got {opt_batch}, {max_batch}")
        self.onnx_path = Path(onnx_path)
        self.engine_dir = Path(engine_dir)
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        self.module = None
        self.engine_info = None

    def config(self) -> dict[str, Any]:
        info = self.engine_info
        h, w = input_hw_for(self.scale)
        return {
            "frontend_model": f"rtmpose-{self.scale}",
            # A model NAME is not an identity: `rtmpose-m` ships as a
            # COCO-WholeBody checkpoint and as a COCO+UBody one, scoring 58.2
            # and 0.631 respectively. Our exports are named `rtmpose-m.onnx`,
            # which records neither -- so the stem is NOT used as the answer.
            # Callers that know the weights pass `checkpoint=`; otherwise the
            # row carries the ONNX content hash, which is unambiguous even
            # though it is not human-readable. Never a bare family name.
            "frontend_checkpoint": self._checkpoint_id(),
            "frontend_input": f"{h}x{w}",
            "frontend_precision": self.precision,
            "frontend_runtime": "tensorrt",
            "frontend_engine": None if info is None else info.short_hash,
            "trt_version": None if info is None else info.trt_version,
        }

    def _checkpoint_id(self) -> str:
        """An identity for the weights, never a bare family name.

        `--checkpoint` when the caller knows it; otherwise the ONNX file's own
        content hash. `rtmpose-m.onnx` could be either published checkpoint and
        a row saying "rtmpose-m" cannot be resolved in month four.
        """
        if self.checkpoint:
            return self.checkpoint
        info = self.engine_info
        if info is not None:
            return f"{self.onnx_path.stem}@{info.onnx_sha256[:12]}"
        return f"{self.onnx_path.stem}@unresolved"

    # -- lifecycle -----------------------------------------------------------

    def setup(self) -> None:
        from cslr_cost.harness.engines import TrtModule, ensure_engine

        h, w = input_hw_for(self.scale)
        one = (1, 3, h, w)
        profiles = [
            {"input": (one, one, one)},
            {"input": ((2, 3, h, w), (self.opt_batch, 3, h, w), (self.max_batch, 3, h, w))},
        ]
        pins = self.FP32_NORM_LAYERS if self.scale in self.PINNED_SCALES else ()
        self.engine_info = ensure_engine(self.onnx_path, self.engine_dir, self.precision, profiles,
                                         fp32_layers=pins)
        self.module = TrtModule(self.engine_info.path, device=self.device)

        outputs = set(self.module.output_names)
        if outputs != {"simcc_x", "simcc_y"}:
            raise ValueError(
                f"{self.onnx_path.name} emits {sorted(outputs)}; expected SimCC's "
                "simcc_x / simcc_y. A graph that already decodes would be decoded twice."
            )

        self._mean = torch.tensor(RTMPOSE_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(RTMPOSE_STD, device=self.device).view(1, 3, 1, 1)
        self._bgr_to_rgb = True

    def teardown(self) -> None:
        self.module = None
        self._mean = None
        self._std = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- the timed unit ------------------------------------------------------

    def _infer(self, crops: list[Any]) -> list[np.ndarray]:
        if self.module is None:
            raise RuntimeError("setup() must run before the extractor is measured")
        if not crops:
            return []

        # Some engines refuse an ODD batch above 1. Measured on this box: a
        # `rtmpose-t` fp32 engine executes batches 1,2,4,6,...,32 and fails
        # every one of 3,5,7,...,31 with
        #
        #     IExecutionContext::enqueueV3: Error Code 1: Cask (Failed to
        #     update runtime arguments.)
        #
        # It is a property of the *tactic*, not of the model: the same ONNX at
        # fp16, and `rtmpose-s` at both precisions, take odd batches happily,
        # while `rtmpose-m` fp32 refuses them exactly as `-t` does. TensorRT
        # selects tactics by timing them, so which side of this a rebuild lands
        # on is not stable — an engine that works today can stop working after
        # a re-export, with no source change.
        #
        # Earlier runs never met it because every measured clip was 128 frames at
        # batch 32: four full batches, never a partial one. Extraction reads
        # whole clips of arbitrary length, so the last batch is partial and odd
        # about half the time.
        #
        # Splitting an odd batch into (even, 1) uses only widths that are known
        # good — 1 has its own static profile — at the cost of one extra call on
        # the final batch alone. Padding the batch instead would spend a real
        # forward pass on a duplicate frame and quietly inflate the stage's
        # measured cost.
        if len(crops) > 1 and len(crops) % 2 == 1:
            return self._infer(crops[:-1]) + self._infer(crops[-1:])

        x = self._preprocess(crops, torch.float32)
        out = self.module(input=x)

        # The decode is host-side, as it is in mmpose; the device-to-host copy
        # is part of this stage's cost in both implementations.
        simcc_x = out["simcc_x"].float().cpu().numpy()
        simcc_y = out["simcc_y"].float().cpu().numpy()
        kpts, scores = decode_simcc(simcc_x, simcc_y)
        return [self._to_frame_coords(k, s, c) for k, s, c in zip(kpts, scores, crops)]



# ---------------------------------------------------------------------------
# HRNet-w48 DARK, 384x288: the front end behind MSKA's released keypoints
# ---------------------------------------------------------------------------

#: mmpose's `flip_indices` for COCO-WholeBody (the dataset metainfo): joint i of
#: a mirrored image is joint FLIP_INDICES[i] of the original. Pinned here, since
#: the Orin has no mmpose, and test-checked against mmpose's own list.
WHOLEBODY_FLIP_INDICES = (
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19,
    39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 49, 48, 47, 46,
    45, 44, 43, 42, 41, 40, 50, 51, 52, 53, 58, 57, 56, 55, 54, 68, 67, 66, 65, 70, 69,
    62, 61, 60, 59, 64, 63, 77, 76, 75, 74, 73, 72, 71, 82, 81, 80, 79, 78, 87, 86, 85,
    84, 83, 90, 89, 88,
    *range(112, 133), *range(91, 112),
)

#: `MSRAHeatmap(input_size=(288, 384), heatmap_size=(72, 96), unbiased=True)`:
#: heatmap -> input canvas is x4, and DARK blurs with an 11x11 kernel.
HRNET_HEATMAP_STRIDE = 4.0
DARK_BLUR_KERNEL = 11


def flip_back_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """Heatmaps of a mirrored input -> heatmaps aligned with the original.

    mmpose `flip_heatmaps(flip_mode="heatmap", shift_heatmap=True)`: flip along
    W, swap each joint with its mirror twin, then shift one pixel right.
    """
    out = heatmaps.flip(-1)[:, list(WHOLEBODY_FLIP_INDICES)]
    out[..., 1:] = out[..., :-1].clone()
    return out


def _gaussian_blur(heatmaps: np.ndarray, kernel: int = DARK_BLUR_KERNEL) -> np.ndarray:
    """mmpose `gaussian_blur`: zero-pad by the kernel radius, blur, crop, and
    rescale each heatmap back to its original peak. (K, H, W), in place."""
    import cv2

    border = (kernel - 1) // 2
    k, h, w = heatmaps.shape
    padded = np.zeros((h + 2 * border, w + 2 * border), dtype=np.float32)
    for j in range(k):
        peak = np.max(heatmaps[j])
        padded[:] = 0.0
        padded[border:-border, border:-border] = heatmaps[j]
        blurred = cv2.GaussianBlur(padded, (kernel, kernel), 0)[border:-border, border:-border]
        heatmaps[j] = blurred * (peak / np.max(blurred))
    return heatmaps


def decode_heatmaps_dark(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, K, H, W) heatmaps -> (N, K, 2) input-canvas coords, (N, K) scores.

    mmpose `MSRAHeatmap.decode` with `unbiased=True`, per instance:
    `get_heatmap_maximum` (argmax; a non-positive peak sets the location to
    -1), `refine_keypoints_dark` (blur, log, one Newton step from the first and
    second derivatives at the peak, skipped within two pixels of the border or
    on a singular Hessian), then x4 to the input canvas. mmpose loops over
    joints in Python; this computes the same step for all joints at once, so
    the baseline is not charged for an interpreter loop. Held to mmpose's
    output by `tests/harness/test_pose_hrnet.py`.
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"expected (N, K, H, W) heatmaps, got {heatmaps.shape}")
    n, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(n * k, -1)
    idx = np.argmax(flat, axis=1)
    scores = flat[np.arange(n * k), idx]
    ys, xs = np.unravel_index(idx, (h, w))
    locs = np.stack((xs, ys), axis=-1).astype(np.float32)
    locs[scores <= 0.0] = -1

    for i in range(n):
        lg = _gaussian_blur(heatmaps[i].copy())
        np.maximum(lg, 1e-10, lg)
        np.log(lg, lg)
        sl = locs[i * k:(i + 1) * k]
        x = sl[:, 0].astype(int)
        y = sl[:, 1].astype(int)
        ok = (x > 1) & (x < w - 2) & (y > 1) & (y < h - 2)
        j, xo, yo = np.nonzero(ok)[0], x[ok], y[ok]
        at = lambda dy, dx: lg[j, yo + dy, xo + dx]  # noqa: E731
        dx = 0.5 * (at(0, 1) - at(0, -1))
        dy = 0.5 * (at(1, 0) - at(-1, 0))
        dxx = 0.25 * (at(0, 2) - 2 * at(0, 0) + at(0, -2))
        dxy = 0.25 * (at(1, 1) - at(-1, 1) - at(1, -1) + at(-1, -1))
        dyy = 0.25 * (at(2, 0) - 2 * at(0, 0) + at(-2, 0))
        det = dxx * dyy - dxy ** 2
        nz = det != 0
        # offset = -H^-1 d, with H^-1 = [[dyy, -dxy], [-dxy, dxx]] / det
        ox = -(dyy[nz] * dx[nz] - dxy[nz] * dy[nz]) / det[nz]
        oy = -(-dxy[nz] * dx[nz] + dxx[nz] * dy[nz]) / det[nz]
        sl[j[nz], 0] += ox
        sl[j[nz], 1] += oy

    return (locs * HRNET_HEATMAP_STRIDE).reshape(n, k, 2), scores.reshape(n, k)


class TrtHrnetStage(_RtmposeBase):
    """HRNet-w48 DARK (384x288) as a TensorRT engine, with mmpose's flip test.

    The front end of MSKA as published. The same stage contract, crop,
    upload, normalisation (mmpose's `PoseDataPreprocessor` values are the
    same as RTMPose's) and unwarp as `TrtRtmposeStage`. What differs is what
    mmpose ran after the network:

      * **flip test:** the crop and its mirror go through the engine in one
        batch of 2N; the mirrored heatmaps are flipped back and the two are
        averaged on the GPU (`HeatmapHead.predict`), so the network cost is
        twice a single pass, as it was for MSKA's authors;
      * **DARK decode** on the host, as mmpose decodes (`decode_heatmaps_dark`).

    `flip_test=False` is the single-pass variant, for pricing the flip alone;
    it no longer reproduces the released keypoints (0.58 px off).
    """

    SUPPORTED_PRECISIONS = ("fp32", "fp16")
    SCALE = "hrnet-w48"

    def __init__(
        self,
        onnx_path: Path,
        engine_dir: Path,
        device: str = "cuda",
        precision: str = "fp32",
        opt_batch: int = DEFAULT_OPT_BATCH,
        max_batch: int = DEFAULT_MAX_BATCH,
        flip_test: bool = True,
        checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__(scale=self.SCALE, device=device, precision=precision)
        if not 2 <= opt_batch <= max_batch:
            raise ValueError(f"need 2 <= opt_batch <= max_batch, got {opt_batch}, {max_batch}")
        self.onnx_path = Path(onnx_path)
        self.engine_dir = Path(engine_dir)
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        self.flip_test = flip_test
        self.checkpoint = checkpoint
        self.module = None
        self.engine_info = None

    def config(self) -> dict[str, Any]:
        info = self.engine_info
        h, w = input_hw_for(self.scale)
        ck = self.checkpoint or (f"{self.onnx_path.stem}@{info.onnx_sha256[:12]}"
                                 if info is not None else f"{self.onnx_path.stem}@unresolved")
        return {
            "frontend_model": "hrnet-w48-dark" + ("-flip" if self.flip_test else ""),
            "frontend_checkpoint": ck,
            "frontend_input": f"{h}x{w}",
            "frontend_precision": self.precision,
            "frontend_runtime": "tensorrt",
            "frontend_engine": None if info is None else info.short_hash,
            "trt_version": None if info is None else info.trt_version,
        }

    def setup(self) -> None:
        from cslr_cost.harness.engines import TrtModule, ensure_engine

        h, w = input_hw_for(self.scale)
        m = 2 if self.flip_test else 1          # engine batch per crop
        one = (m, 3, h, w)
        profiles = [
            {"input": (one, one, one)},
            {"input": ((2 * m, 3, h, w), (m * self.opt_batch, 3, h, w), (m * self.max_batch, 3, h, w))},
        ]
        self.engine_info = ensure_engine(self.onnx_path, self.engine_dir, self.precision, profiles)
        self.module = TrtModule(self.engine_info.path, device=self.device)
        if set(self.module.output_names) != {"heatmaps"}:
            raise ValueError(f"{self.onnx_path.name} emits {sorted(self.module.output_names)}; "
                             "expected HRNet's single heatmaps tensor")
        self._mean = torch.tensor(RTMPOSE_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(RTMPOSE_STD, device=self.device).view(1, 3, 1, 1)
        self._bgr_to_rgb = True

    def teardown(self) -> None:
        self.module = None
        self._mean = None
        self._std = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _infer(self, crops: list[Any]) -> list[np.ndarray]:
        if self.module is None:
            raise RuntimeError("setup() must run before the extractor is measured")
        if not crops:
            return []
        # The same tactic property `TrtRtmposeStage._infer` documents: the
        # fp32 engine on the dev box refuses every odd crop count from 3 up
        # (engine batch 2N = 6, 10, 14, ... with flip; 2026-10-18, it failed
        # 241 of 540 dev clips in an extraction). Split as there.
        if len(crops) > 1 and len(crops) % 2 == 1:
            return self._infer(crops[:-1]) + self._infer(crops[-1:])
        x = self._preprocess(crops, torch.float32)
        n = x.shape[0]
        if self.flip_test:
            hm = self.module(input=torch.cat([x, x.flip(-1)]))["heatmaps"].float()
            hm = (hm[:n] + flip_back_heatmaps(hm[n:])) * 0.5
        else:
            hm = self.module(input=x)["heatmaps"].float()
        kpts, scores = decode_heatmaps_dark(hm.cpu().numpy())
        return [self._to_frame_coords(k, s, c) for k, s, c in zip(kpts, scores, crops)]


#: The pose model keys the harness knows: RTMPose/DWPose scales, and HRNet.
HRNET_ONNX = "hrnet-w48-dark-384x288.onnx"


def make_trt_pose_stage(onnx_dir: Path, scale: str, precision: str, engine_dir: Path,
                        opt_batch: int, max_batch: int, weights: str = "dw-ucoco"):
    """The TensorRT pose stage for a `--pose-scale` value."""
    if scale == TrtHrnetStage.SCALE:
        if weights != "dw-ucoco":
            raise ValueError("hrnet-w48 has one checkpoint; --weights does not apply")
        return TrtHrnetStage(Path(onnx_dir) / HRNET_ONNX, engine_dir, precision=precision,
                             opt_batch=opt_batch, max_batch=max_batch)
    return TrtRtmposeStage(onnx_for_scale(Path(onnx_dir), scale, precision, weights), engine_dir,
                           scale=scale, precision=precision, opt_batch=opt_batch, max_batch=max_batch)

def onnx_for_scale(onnx_dir: Path, scale: str, precision: str = "fp32",
                   weights: str = "dw-ucoco") -> Path:
    """The ONNX an engine of this precision is built from.

    fp32 and fp16 compile the same graph: `rtmpose-{scale}.onnx`, as the export's
    export names them. int8 compiles the QDQ export beside it,
    `rtmpose-{scale}.int8.qdq.onnx` (`export/quantize_rtmpose_int8.py`) --
    `engines.py` refuses a graph with no QDQ nodes, so asking for int8 from the
    FP32 file fails loudly rather than quantizing on nothing.
    """
    stem = f"rtmpose-{scale}" if weights == "dw-ucoco" else f"rtmpose-{scale}.{weights}"
    if precision == "int8":
        return Path(onnx_dir) / f"{stem}.int8.qdq.onnx"
    return Path(onnx_dir) / f"{stem}.onnx"


__all__: Optional[list[str]] = [
    "DEFAULT_MAX_BATCH",
    "HRNET_ONNX",
    "TrtHrnetStage",
    "WHOLEBODY_FLIP_INDICES",
    "decode_heatmaps_dark",
    "flip_back_heatmaps",
    "make_trt_pose_stage",
    "DEFAULT_OPT_BATCH",
    "RTMPOSE_INPUT_HW",
    "RTMPOSE_MEAN",
    "RTMPOSE_STD",
    "SIMCC_SPLIT_RATIO",
    "TrtRtmposeStage",
    "decode_simcc",
    "onnx_for_scale",
]

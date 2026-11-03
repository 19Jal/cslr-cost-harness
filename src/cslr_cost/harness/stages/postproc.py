"""postproc.py — keypoint normalisation and layout mapping (stage 5).

Turns the extractor's raw keypoints into what the head expects. Per-frame.

--------------------------------------------------------------------------
THE NORMALISATION IS MSKA'S, EXACTLY
--------------------------------------------------------------------------
From `MSKA/datasets.py:102-106`:

    kp[:, 0] /= w                        # x by frame width
    kp[:, 1]  = h - kp[:, 1]             # flip y: origin to bottom-left
    kp[:, 1] /= h                        # y by frame height
    kp[:, :2] = (kp[:, :2] - 0.5) / 0.5  # rescale to [-1, 1]

The confidence channel is left untouched.

This is the likeliest silent failure in the whole pipeline. Uni-Sign normalises
hands and face to root joints and leaves the body unnormalised; if we inherit
that convention instead, **the head still runs and still produces plausible
glosses, at meaningfully worse WER, with no error anywhere**. Nothing crashes. Nothing warns. The number is just quietly wrong.

--------------------------------------------------------------------------
`w` AND `h` ARE THE FRAME'S OWN DIMENSIONS
--------------------------------------------------------------------------
MSKA hardcodes 210x260 for PHOENIX-2014 because its released keypoints were
extracted from native-resolution frames. That constant is **not** transferable:
the divisor has to be the dimensions the keypoints were actually measured in.

It matters here specifically because of the resolution ladder. Keypoints
extracted from an upscaled 720p clip are in 720p coordinates; dividing them by
210x260 puts them far outside [-1, 1] and the head sees nonsense. The ladder is
cost-only, so no *accuracy* number is at risk — compute depends on tensor
dimensions, not values — but an adapter that silently mis-normalises is a trap
waiting for the first native-resolution run.

--------------------------------------------------------------------------
THE 133 -> HEAD LAYOUT MAPPING IS STILL OPEN
--------------------------------------------------------------------------
MSKA's head consumes **V=133** and selects overlapping groups internally (left
27, right 27, face 26, body 79 — see the DSTA config). It does *not* want a
reduced set.

RTMPose whole-body also emits 133, but whether index i means the same joint in
both layouts is unverified. Until it is, `index_map` is identity and this
module is correct only for keypoints already in MSKA's layout — which is the
case for its released HRNet keypoints, and is exactly the reference the
comparison of front ends is made against.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from cslr_cost.harness.stages.base import Stage

#: PHOENIX-2014 native frame size, as MSKA hardcodes it. A default, not a rule.
PHOENIX_2014_WH = (210, 260)
CSL_DAILY_WH = (512, 512)

#: What MSKA's head consumes.
HEAD_KEYPOINTS = 133


class KeypointPostprocStage(Stage):
    """Normalise raw keypoints into the head's input convention."""

    name = "postproc"
    input_key = "keypoints_raw"
    output_key = "keypoints"
    granularity = "frame"
    uses_cuda = False

    def __init__(
        self,
        frame_wh: tuple[int, int] = PHOENIX_2014_WH,
        index_map: Optional[Sequence[int]] = None,
        expect_keypoints: int = HEAD_KEYPOINTS,
    ) -> None:
        self.frame_w, self.frame_h = frame_wh
        self.index_map = np.asarray(index_map, dtype=np.int64) if index_map is not None else None
        self.expect_keypoints = expect_keypoints

    def config(self) -> dict[str, Any]:
        return {
            "notes": (
                f"normalised against {self.frame_w}x{self.frame_h}; "
                f"layout map: {'identity' if self.index_map is None else 'explicit'}"
            )
        }

    def run(self, keypoints_raw: Any) -> np.ndarray:
        """One frame's raw keypoints -> normalised keypoints.

        Accepts (K, 3) as (x, y, confidence) in pixel coordinates.
        """
        kp = keypoints_raw
        if torch.is_tensor(kp):
            kp = kp.detach().cpu().numpy()
        kp = np.asarray(kp, dtype=np.float32)

        if kp.ndim != 2 or kp.shape[1] < 2:
            raise ValueError(
                f"expected one frame of keypoints shaped (K, 2 or 3), got {kp.shape}. "
                "This stage is per-frame; pass a clip's worth and the "
                "normalisation silently applies to the wrong axis."
            )

        if self.index_map is not None:
            kp = kp[self.index_map]

        if kp.shape[0] != self.expect_keypoints:
            raise ValueError(
                f"got {kp.shape[0]} keypoints, head expects {self.expect_keypoints}. "
                "MSKA's DSTA indexes groups out of the full 133-point layout, so a "
                "reduced set indexes out of bounds — which surfaces as "
                "CUDNN_STATUS_INTERNAL_ERROR, pointing nowhere near the cause."
            )

        out = kp.copy()
        out[:, 0] /= self.frame_w
        out[:, 1] = self.frame_h - out[:, 1]
        out[:, 1] /= self.frame_h
        out[:, :2] = (out[:, :2] - 0.5) / 0.5
        return out


def normalise_clip(
    keypoints: Any,
    frame_wh: tuple[int, int] = PHOENIX_2014_WH,
    index_map: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """A whole clip's raw keypoints -> normalised, in one vectorised pass.

    `KeypointPostprocStage.run` is the *timed* unit and stays per-frame, because
    that is how the deployed pipeline runs it. Evaluation is not timed and reads
    whole clips, so it would otherwise loop the stage T times purely to satisfy
    the measurement's shape contract.

    Two implementations of MSKA's normalisation is exactly the silent failure
    `postproc`'s docstring warns about, so `tests/harness/test_postproc.py`
    asserts this function frame-by-frame equals the stage. Change one, the test
    fails on the other.

    Takes (T, K, 2 or 3) in pixel coordinates, returns (T, K, C).
    """
    kp = keypoints
    if torch.is_tensor(kp):
        kp = kp.detach().cpu().numpy()
    kp = np.asarray(kp, dtype=np.float32)

    if kp.ndim != 3 or kp.shape[2] < 2:
        raise ValueError(
            f"expected a clip shaped (T, K, 2 or 3), got {kp.shape}. "
            "For a single frame use KeypointPostprocStage.run."
        )

    if index_map is not None:
        kp = kp[:, np.asarray(index_map, dtype=np.int64)]

    frame_w, frame_h = frame_wh
    out = kp.copy()
    out[:, :, 0] /= frame_w
    out[:, :, 1] = frame_h - out[:, :, 1]
    out[:, :, 1] /= frame_h
    out[:, :, :2] = (out[:, :, :2] - 0.5) / 0.5
    return out


def assemble_sequence(units: list[np.ndarray]) -> torch.Tensor:
    """Per-frame keypoints -> the clip tensor the head consumes.

    (T, K, C) -> (1, C, T, K): batch, channels (x, y, confidence), time, joints.
    This is the pipeline's assembly boundary — where per-frame processing ends
    and the per-clip head begins.
    """
    if not units:
        raise ValueError("cannot assemble an empty sequence")
    stacked = np.stack(units, axis=0)              # (T, K, C)
    tensor = torch.from_numpy(stacked).permute(2, 0, 1)   # (C, T, K)
    return tensor.unsqueeze(0).contiguous()        # (1, C, T, K)

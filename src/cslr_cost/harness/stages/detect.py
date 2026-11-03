"""detect.py — person detection (stage 2).

Finds the signer's bounding box so `crop` can warp it into the pose model's
input. Per-frame.

--------------------------------------------------------------------------
ON PHOENIX THIS STAGE IS DEGENERATE, AND THAT IS A FINDING, NOT A GAP
--------------------------------------------------------------------------
PHOENIX-2014 ships 210x260 frames that are *already* a tight signer view. Run a
detector on them and it returns approximately the whole frame. So there are two
honest things to measure, and they answer different questions:

- **`WholeFrameDetectStage`** — the box is the frame. Near-zero cost, and the
  correct choice for any run whose output feeds a WER number, because it is
  what MSKA's released keypoints effectively assumed. Reports
  `detector: "whole-frame"` so no reader mistakes it for a detector result.

- **`RtmdetPersonStage`** — a real detector. This is the number a *deployed*
  system pays, where the camera sees a room rather than a pre-cropped signer,
  and it belongs in the per-stage budget for exactly that reason.

Quoting the first as though it were the second would understate the front end,
which is the error a whole-pipeline account is meant to avoid. Quoting the second on
PHOENIX and calling it representative would overstate it. Both get measured;
each row says which.

--------------------------------------------------------------------------
THE DETECTOR'S OWN PREPROCESSING BELONGS TO THE DETECTOR
--------------------------------------------------------------------------
`inference_detector` resizes to 320x320, pads and normalises before the
network runs. That cost is charged here rather than to `decode` or `crop`: it
exists only because a detector is present, and it disappears entirely in the
whole-frame configuration. Attributing it anywhere else would make the
detector look cheaper than deploying one actually is.

--------------------------------------------------------------------------
A MISSED DETECTION MUST NOT LOOK LIKE A SUCCESSFUL ONE
--------------------------------------------------------------------------
When nothing is detected the stage falls back to the whole frame, because the
pipeline has to produce a frame's keypoints either way. But it **counts** the
fallbacks. Without that counter a detector that fires on 60% of frames and
silently whole-frames the rest reports the same box quality as one that works,
and the pose numbers downstream carry an error with no trace of where it
entered.

--------------------------------------------------------------------------
OUTPUT IS `(frame, box)`, NOT JUST `box`
--------------------------------------------------------------------------
`crop` needs both, and passing the box alone would force `crop` to hold a
reference to the frame it was not given -- which breaks isolation replay, since
the fixture store keys strictly on boundary artifacts. The `boxes` boundary is
defined as the pair in `stages/base.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from cslr_cost.harness.stages.base import Stage

#: Score below which a detection is treated as no detection.
DEFAULT_SCORE_THR = 0.3

#: COCO's person class. The nano person checkpoint has `num_classes=1`, so its
#: only label is 0 either way -- kept explicit so a swap to a general-purpose
#: 80-class detector does not silently start cropping to furniture.
PERSON_LABEL = 0


class WholeFrameDetectStage(Stage):
    """The box is the whole frame. Correct for pre-cropped corpora."""

    name = "detect"
    input_key = "frames"
    output_key = "boxes"
    granularity = "frame"
    uses_cuda = False

    def config(self) -> dict[str, Any]:
        # No runtime: there is no model here, and naming one would imply a
        # detector ran.
        return {"detector_policy": "whole-frame"}

    def run(self, frame: Any) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        return frame, np.array([0.0, 0.0, w, h], dtype=np.float32)


class RtmdetPersonStage(Stage):
    """RTMDet person detector (mmdet), one frame per call."""

    name = "detect"
    input_key = "frames"
    output_key = "boxes"
    granularity = "frame"
    uses_cuda = True

    def __init__(
        self,
        config_path: Path,
        checkpoint: Path,
        device: str = "cuda",
        score_thr: float = DEFAULT_SCORE_THR,
        model_name: str = "rtmdet-nano",
        precision: str = "fp32",
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.score_thr = score_thr
        self.model_name = model_name
        self.precision = precision
        self.model = None
        #: Frames on which nothing scored above threshold and the whole frame
        #: was substituted. Reported, never swallowed.
        self.fallback_frames = 0
        self.frames_seen = 0

    def config(self) -> dict[str, Any]:
        return {
            "detector_policy": f"{self.model_name}/{self.precision}@{self.score_thr}",
            "detect_runtime": "pytorch",
        }

    def setup(self) -> None:
        from mmdet.apis import init_detector

        self.model = init_detector(
            str(self.config_path), str(self.checkpoint), device=self.device
        )

    def teardown(self) -> None:
        self.model = None

    @property
    def fallback_rate(self) -> Optional[float]:
        if not self.frames_seen:
            return None
        return self.fallback_frames / self.frames_seen

    def run(self, frame: Any) -> tuple[np.ndarray, np.ndarray]:
        from mmdet.apis import inference_detector

        if self.model is None:
            raise RuntimeError("setup() must run before the detector is measured")

        self.frames_seen += 1
        result = inference_detector(self.model, frame)
        pred = result.pred_instances

        keep = (pred.scores > self.score_thr) & (pred.labels == PERSON_LABEL)
        if not bool(keep.any()):
            self.fallback_frames += 1
            h, w = frame.shape[:2]
            return frame, np.array([0.0, 0.0, w, h], dtype=np.float32)

        boxes = pred.bboxes[keep]
        scores = pred.scores[keep]
        best = boxes[int(scores.argmax())]
        return frame, best.detach().cpu().numpy().astype(np.float32)

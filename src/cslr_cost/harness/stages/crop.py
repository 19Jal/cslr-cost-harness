"""crop.py — bbox to model input (stage 3).

Takes the detector's box together with the frame it came from, and produces the
fixed-size region the pose extractor consumes. Per-frame, CPU.

--------------------------------------------------------------------------
THIS IS AN AFFINE WARP, NOT A CROP-AND-RESIZE
--------------------------------------------------------------------------
The stage is called `crop` because that is what the pipeline diagram calls it,
but the operation top-down pose estimators actually perform is an **affine
warp**: the box is padded (x1.25), reshaped to the model's aspect ratio, and
mapped onto a 192x256 canvas in one `warpAffine`. Slicing the array and calling
`cv2.resize` is not the same transform — it distorts aspect ratio and drops the
padding — and a model fed the wrong geometry does not fail. It returns
keypoints that are simply worse, which is unattributable later.

So the geometry here is mmpose's own: `bbox_xyxy2cs` and `get_warp_matrix`
are copied verbatim from `mmpose/structures/bbox/transforms.py` (v1.3.2, the
version the checkpoints were evaluated with), restricted to the arguments this
stage uses — no rotation, no shift. The padding constant, the aspect-ratio fix
and the warp all come from the config the checkpoint was trained and evaluated
with. A reimplementation that drifts from it costs accuracy silently, which is
the worst kind of cost — hence copied rather than rewritten, and held to
mmpose's own functions by `tests/harness/test_crop.py` wherever mmpose is
installed.

They are copied rather than imported because the Orin has no mmpose, and
installing OpenMMLab on a Jetson to get two dozen lines of NumPy is exactly the
dependency the TensorRT path exists to avoid. Before the copy, this was the one
front-end stage that still could not run on the device.

--------------------------------------------------------------------------
WHY `center` AND `scale` TRAVEL WITH THE CROP
--------------------------------------------------------------------------
The pose model returns keypoints in **crop** coordinates. They have to be
mapped back to the source frame before `postproc` normalises them, and the
inverse warp needs the center/scale that produced the crop.

Lose them and nothing crashes: `postproc` divides crop-space coordinates by the
frame's width and height, producing values that are still finite, still roughly
in range, and wrong for every frame. The head runs, glosses come out, WER is
worse for a reason no experiment is varying. That is why they
are carried in the boundary artifact instead of being recomputed downstream.

--------------------------------------------------------------------------
THE PADDING CONSTANT IS OPEN ON PRE-CROPPED CORPORA
--------------------------------------------------------------------------
1.25 is mmpose's default and what the RTMPose checkpoints were trained and
evaluated with, so it stays the default here. But PHOENIX frames are *already*
a tight 210x260 signer view, and padding a full frame pushes the warp canvas
outside the image, so keypoints predicted in the padded band map outside the
frame. Measured against MSKA's released HRNet keypoints on
`dev/01April_2010_Thursday_heute_default-1`, normalised out-of-range fractions
come out as:

    MSKA HRNet (the reference)            0.014%
    rtmpose-t, padding 1.25               4.95%
    rtmpose-t, padding 1.00               6.83%   (x perfect, y worse)

Neither padding reproduces the reference, and 1.0 is not simply better: it
fixes x (the canvas then matches the frame width exactly) while making y worse,
because the aspect-ratio fix grows 260 -> 280 regardless and that band sits
outside the image. A SimCC/heatmap argmax is always inside its own canvas, so
the *only* source of out-of-frame keypoints is canvas geometry that exceeds the
frame.

**This changes no cost number** -- the crop is 192x256 whichever padding is
used, so cost measurements are unaffected. It is recorded here because it
is an accuracy risk of exactly the kind `postproc.py` describes: the head runs,
glosses come out, and roughly 5% of keypoints sit outside the [-1, 1] range it
was trained on. Resolving it needs the same comparison run properly across clips.

--------------------------------------------------------------------------
NO COLOUR CONVERSION AND NO NORMALISATION HERE
--------------------------------------------------------------------------
Both belong to `pose`, on the GPU, inside RTMPose's `PoseDataPreprocessor`
(`bgr_to_rgb=True`, mean/std). Doing either here would move real milliseconds
out of the stage that a deployment would charge them to, and would also
quadruple the bytes crossing the PCIe boundary by uploading float32 instead of
uint8. See `decode.py` for the full argument; the short version is that stage
boundaries are a measurement decision, so they are made once and written down.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from cslr_cost.harness.stages.base import Stage

#: RTMPose whole-body input, as (w, h). The config's `codec.input_size`.
RTMPOSE_INPUT_SIZE = (192, 256)

#: mmpose's `GetBBoxCenterScale` default, and what the RTMPose configs use.
BBOX_PADDING = 1.25


class Crop:
    """A warped model input plus the geometry needed to invert it.

    `img` is uint8 BGR (h, w, 3) — unconverted and unnormalised on purpose.
    """

    __slots__ = ("img", "center", "scale", "input_size", "source_wh")

    def __init__(
        self,
        img: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        input_size: tuple[int, int],
        source_wh: tuple[int, int],
    ) -> None:
        self.img = img
        self.center = center
        self.scale = scale
        self.input_size = input_size
        #: The frame this came from. `postproc` needs it to normalise, and it
        #: must be the frame's real size, not a hardcoded corpus constant --
        #: see the resolution-ladder warning in postproc.py.
        self.source_wh = source_wh

    def __repr__(self) -> str:
        return (
            f"<Crop {self.img.shape} center={self.center.round(1).tolist()} "
            f"scale={self.scale.round(1).tolist()}>"
        )


def _bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(x1, y1, x2, y2) boxes, shape (n, 4) -> center (n, 2) and padded scale (n, 2).

    `mmpose.structures.bbox.bbox_xyxy2cs`, verbatim for 2-D input.
    """
    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5
    return center, scale


def _get_warp_matrix(
    center: np.ndarray, scale: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
    """2x3 affine mapping the box onto the canvas.

    `mmpose.structures.bbox.get_warp_matrix` with `rot=0`, `shift=(0, 0)`,
    `inv=False`, `fix_aspect_ratio=True` — the only arguments this stage passes.
    With no rotation, `_rotate_point` is the identity and is dropped.
    """
    import cv2

    src_w = scale[0]
    dst_w, dst_h = output_size[:2]

    src_dir = np.array([src_w * -0.5, 0.0])
    dst_dir = np.array([dst_w * -0.5, 0.0])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a - b` rotated 90 degrees anticlockwise about `b`. mmpose, verbatim."""
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Reshape a box to the model's aspect ratio by growing, never cropping.

    Copied from `mmpose.datasets.transforms.TopdownAffine._fix_aspect_ratio`
    (a private static method, so it is reproduced rather than imported). It
    expands the short side; shrinking the long one instead would cut hands out
    of frame, which for sign language is not a small error.
    """
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


class TopdownCropStage(Stage):
    """Warp the detected box into the pose model's input canvas."""

    name = "crop"
    input_key = "boxes"
    output_key = "crops"
    granularity = "frame"
    uses_cuda = False

    def __init__(
        self,
        input_size: tuple[int, int] = RTMPOSE_INPUT_SIZE,
        padding: float = BBOX_PADDING,
    ) -> None:
        self.input_size = tuple(input_size)
        self.padding = padding

    def config(self) -> dict[str, Any]:
        w, h = self.input_size
        return {"notes": f"crop {w}x{h} pad {self.padding}"}

    def run(self, boxes: Any) -> Crop:
        """`(frame, box_xyxy)` -> a warped `Crop`."""
        import cv2

        frame, box = boxes
        if frame.ndim != 3:
            raise ValueError(
                f"expected one frame (H, W, 3), got {frame.shape}. This stage "
                "is per-frame; a stacked clip would warp the wrong axes."
            )

        bbox = np.asarray(box, dtype=np.float32).reshape(1, 4)
        center, scale = _bbox_xyxy2cs(bbox, padding=self.padding)

        w, h = self.input_size
        scale = _fix_aspect_ratio(scale, aspect_ratio=w / h)

        warp_mat = _get_warp_matrix(center[0], scale[0], output_size=(w, h))
        img = cv2.warpAffine(frame, warp_mat, (w, h), flags=cv2.INTER_LINEAR)

        src_h, src_w = frame.shape[:2]
        return Crop(
            img=img,
            center=center[0],
            scale=scale[0],
            input_size=self.input_size,
            source_wh=(src_w, src_h),
        )

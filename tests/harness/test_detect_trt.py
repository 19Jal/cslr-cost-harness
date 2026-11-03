"""The TensorRT RTMDet stage: its letterbox, its decode, and its boxes.

Layered by what each part needs:

  * the letterbox geometry against mmcv's own rescale arithmetic -- pure numpy.
  * the decode against mmdet's `predict_by_feat`, fed the same feature maps --
    needs mmdet, not TensorRT.
  * the whole stage against `RtmdetPersonStage` on real frames -- needs
    TensorRT, the export and the checkpoint.

The middle layer is the one that matters: a decode that is subtly wrong still
returns plausible boxes, and plausible boxes crop plausible-looking signers.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cslr_cost.harness.stages.detect_trt import (  # noqa: E402
    INPUT_SIZE,
    PAD_VALUE,
    RTMDET_MEAN,
    RTMDET_STD,
    STRIDES,
    TrtRtmdetPersonStage,
    decode_rtmdet,
    grid_priors,
    letterbox,
)

SHARED = Path(os.environ.get("CSLR_DATA", "data"))
RTMDET_DIR = SHARED / "rtmdet"
CONFIG = RTMDET_DIR / "rtmdet_nano_320-8xb32_coco-person.py"
CHECKPOINT = RTMDET_DIR / "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
ONNX = Path("models/onnx/rtmdet-nano-person-320.onnx")
ENGINE_DIR = Path("models/engines/test")
LADDER_360P = Path("data/res_ladder/out/360p")
PHOENIX = Path(
    "data/phoenix2014/phoenix2014-release/phoenix-2014-multisigner"
    "/features/fullFrame-210x260px/dev"
)


# ---------------------------------------------------------------------------
# letterbox
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h, w, expect", [
    (260, 210, (258, 320)),      # PHOENIX native: width and height round apart
    (320, 320, (320, 320)),
    (180, 640, (320, 90)),
])
def test_letterbox_matches_mmcv_rescale(h, w, expect):
    pytest.importorskip("cv2")
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    canvas, (sx, sy) = letterbox(frame)
    assert canvas.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    nw, nh = expect
    assert (round(sx * w), round(sy * h)) == (nw, nh)
    assert canvas[:nh, :nw].shape[:2] == (nh, nw)
    if nh < INPUT_SIZE:
        assert (canvas[nh:, :] == PAD_VALUE).all()
    if nw < INPUT_SIZE:
        assert (canvas[:, nw:] == PAD_VALUE).all()


def test_scale_factors_are_per_axis_not_shared():
    """210x260 rescales to 258x320: 1.2286 across, 1.2308 down."""
    pytest.importorskip("cv2")
    _, (sx, sy) = letterbox(np.zeros((260, 210, 3), dtype=np.uint8))
    assert sx != sy
    assert sx == pytest.approx(258 / 210, abs=1e-6)
    assert sy == pytest.approx(320 / 260, abs=1e-6)


def test_priors_are_stride_grids_at_offset_zero():
    p = grid_priors(3, 8, torch.device("cpu"))
    assert p.shape == (9, 2)
    np.testing.assert_array_equal(p[:3].numpy(), [[0, 0], [8, 0], [16, 0]])
    np.testing.assert_array_equal(p[3].numpy(), [0, 8])


def test_normalisation_constants_match_the_config():
    # OpenMMLab is absent on the Orin by design; the constants are checked
    # against its configs wherever it is installed.
    pytest.importorskip("mmengine")
    from mmengine.config import Config

    if not CONFIG.exists():
        pytest.skip("rtmdet config not available")
    pytest.importorskip("mmdet")
    dp = Config.fromfile(str(CONFIG)).model.data_preprocessor
    assert tuple(dp.mean) == RTMDET_MEAN
    assert tuple(dp.std) == RTMDET_STD
    assert dp.bgr_to_rgb is False, "RTMDet eats BGR; flipping moves every box"


# ---------------------------------------------------------------------------
# decode, against mmdet's own, on the same feature maps
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="rtmdet checkpoint not available")
def test_decode_reproduces_mmdet_predict_by_feat():
    pytest.importorskip("mmdet")
    from mmdet.apis import init_detector

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model = init_detector(str(CONFIG), str(CHECKPOINT), device="cuda").eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device="cuda")
    with torch.no_grad():
        cls_scores, bbox_preds = model.bbox_head(model.extract_feat(x))
        ref = model.bbox_head.predict_by_feat(
            list(cls_scores), list(bbox_preds),
            batch_img_metas=[{"img_shape": (INPUT_SIZE, INPUT_SIZE),
                              "scale_factor": (1.0, 1.0),
                              "ori_shape": (INPUT_SIZE, INPUT_SIZE)}],
            rescale=False,
        )[0]

    boxes, scores = decode_rtmdet(list(cls_scores), list(bbox_preds))
    assert len(boxes) == len(ref.bboxes), "different number of detections survived NMS"
    if len(boxes):
        order, ref_order = scores.argsort(descending=True), ref.scores.argsort(descending=True)
        np.testing.assert_allclose(
            boxes[order].cpu().numpy(), ref.bboxes[ref_order].cpu().numpy(), atol=1e-3)
        np.testing.assert_allclose(
            scores[order].cpu().numpy(), ref.scores[ref_order].cpu().numpy(), atol=1e-5)


def test_levels_and_strides_agree_with_the_head():
    assert STRIDES == (8, 16, 32)
    assert [INPUT_SIZE // s for s in STRIDES] == [40, 20, 10]


# ---------------------------------------------------------------------------
# the whole stage, against the PyTorch adapter, on real frames
# ---------------------------------------------------------------------------

def _frames(n: int = 6) -> list[np.ndarray]:
    pytest.importorskip("cv2")
    from cslr_cost.harness.stages.decode import open_video

    clip = next(iter(sorted(LADDER_360P.glob("*.mp4"))), None)
    if clip is None and PHOENIX.is_dir():
        seq = sorted(p for p in PHOENIX.iterdir() if p.is_dir())[0]
        clip = seq / "1" if (seq / "1").is_dir() else seq
    if clip is None:
        pytest.skip("no clip to detect in")
    src = open_video(clip)
    out = [src.read() for _ in range(n)]
    src.close()
    return out


@pytest.mark.parametrize("precision, px_tol", [("fp32", 1.0), ("fp16", 2.0)])
def test_engine_boxes_match_the_pytorch_detector(precision, px_tol):
    from cslr_cost.harness.engines import trt_version
    from cslr_cost.harness.stages.detect import RtmdetPersonStage

    if trt_version() is None or not ONNX.exists() or not CHECKPOINT.exists():
        pytest.skip("TensorRT, the RTMDet export or the checkpoint is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    frames = _frames()
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = RtmdetPersonStage(CONFIG, CHECKPOINT)
        ref.setup()
        expected = [ref.run(f)[1] for f in frames]
        ref_fallbacks = ref.fallback_frames
        ref.teardown()
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved

    stage = TrtRtmdetPersonStage(ONNX, ENGINE_DIR, precision=precision)
    stage.setup()
    try:
        got = [stage.run(f)[1] for f in frames]
        assert stage.config()["detect_engine"] is not None
        assert stage.config()["detect_runtime"] == "tensorrt"
        assert stage.frames_seen == len(frames)
        # A detector that silently whole-frames half its input must not read as
        # agreement: the fallback counts have to match too.
        assert stage.fallback_frames == ref_fallbacks
    finally:
        stage.teardown()

    for g, e in zip(got, expected):
        assert g.shape == e.shape == (4,)
        assert np.abs(g - e).max() <= px_tol, f"box {g} vs mmdet {e}"

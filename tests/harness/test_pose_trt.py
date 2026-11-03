"""The TensorRT RTMPose stage: its decode against mmpose's, its engine against PyTorch.

Three layers, each skipped only for what it genuinely needs:

  * `decode_simcc` against mmpose's `SimCCLabel.decode` on synthetic logits
    -- needs mmpose, not TensorRT or checkpoints.
  * the pinned mean/std/codec settings against every scale's config
    -- needs mmpose's bundled configs.
  * the engine against `RtmposeStage` on real PHOENIX crops, fp32 and fp16
    -- needs TensorRT, the ONNX exports, checkpoints and frames.

The last is "verify the adapter reproduces the SDK output",
done while the Python path is still available to compare against.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cslr_cost.harness.stages.pose_trt import (  # noqa: E402
    RTMPOSE_MEAN,
    RTMPOSE_STD,
    SIMCC_SPLIT_RATIO,
    TrtRtmposeStage,
    decode_simcc,
    onnx_for_scale,
)

SHARED = Path(os.environ.get("CSLR_DATA", "data"))
ONNX_DIR = SHARED / "onnx"
RTMPOSE_DIR = SHARED / "rtmpose"
PHOENIX = Path(
    "data/phoenix2014/phoenix2014-release/phoenix-2014-multisigner"
    "/features/fullFrame-210x260px/dev"
)
LADDER_360P = Path("data/res_ladder/out/360p")
ENGINE_DIR = Path("models/engines/test")


def _mmpose_configs() -> Path:
    mmpose = pytest.importorskip("mmpose")
    return (
        Path(os.path.dirname(mmpose.__file__))
        / ".mim/configs/wholebody_2d_keypoint/rtmpose/ubody"
    )


def _config(scale: str) -> Path:
    return _mmpose_configs() / f"rtmpose-{scale}_8xb64-270e_coco-ubody-wholebody-256x192.py"


# ---------------------------------------------------------------------------
# decode, no TensorRT needed
# ---------------------------------------------------------------------------

def test_decode_matches_mmpose_simcc_label():
    pytest.importorskip("mmpose")
    from mmpose.codecs import SimCCLabel

    codec = SimCCLabel(input_size=(192, 256), sigma=(4.9, 5.66),
                       simcc_split_ratio=2.0, normalize=False, use_dark=False)
    rng = np.random.default_rng(0)
    n, k = 5, 133
    sx = rng.normal(0.5, 1.0, (n, k, 384)).astype(np.float32)
    sy = rng.normal(0.5, 1.0, (n, k, 512)).astype(np.float32)
    sx[0, :10] -= 5.0   # some keypoints with non-positive peaks, which mmpose sets to -1

    kpts, scores = decode_simcc(sx, sy)
    for i in range(n):
        ref_k, ref_s = codec.decode(sx[i], sy[i])     # per instance, as the head calls it
        np.testing.assert_array_equal(kpts[i], ref_k[0])
        np.testing.assert_array_equal(scores[i], ref_s[0])


def test_non_positive_peaks_decode_to_the_mmpose_sentinel():
    sx = np.full((1, 2, 384), -1.0, dtype=np.float32)
    sy = np.full((1, 2, 512), -1.0, dtype=np.float32)
    sx[0, 1, 7], sy[0, 1, 9] = 3.0, 2.0
    kpts, scores = decode_simcc(sx, sy)
    np.testing.assert_array_equal(kpts[0, 0], [-1 / SIMCC_SPLIT_RATIO] * 2)
    np.testing.assert_array_equal(kpts[0, 1], [7 / SIMCC_SPLIT_RATIO, 9 / SIMCC_SPLIT_RATIO])
    assert scores[0, 1] == 2.0     # the smaller of the two axes' peaks


def test_decode_refuses_mismatched_shapes():
    with pytest.raises(ValueError):
        decode_simcc(np.zeros((1, 133, 384)), np.zeros((2, 133, 512)))


@pytest.mark.parametrize("scale", ["t", "s", "m"])
def test_pinned_constants_match_every_scale_config(scale):
    # OpenMMLab is absent on the Orin by design; the constants are checked
    # against its configs wherever it is installed.
    pytest.importorskip("mmengine")
    from mmengine.config import Config

    cfg_path = _config(scale)
    if not cfg_path.exists():
        pytest.skip(f"{cfg_path.name} not bundled")
    cfg = Config.fromfile(str(cfg_path))
    dp = cfg.model.data_preprocessor
    assert tuple(dp.mean) == RTMPOSE_MEAN
    assert tuple(dp.std) == RTMPOSE_STD
    assert dp.bgr_to_rgb is True
    codec = cfg.codec
    assert codec.type == "SimCCLabel"
    assert codec.simcc_split_ratio == SIMCC_SPLIT_RATIO
    assert codec.use_dark is False, "DARK refinement would need a different decode"
    assert not codec.get("decode_visibility", False)


def test_int8_is_accepted_on_tensorrt_and_refused_on_torch():
    """INT8 exists on one runtime only, and the split is the point.

    TensorRT compiles the QDQ export; the torch path has no
    calibration set and no int8 kernel, so asking for it there is a mistake
    worth naming rather than a silent fp32 run."""
    from cslr_cost.harness.stages.pose import RtmposeStage

    TrtRtmposeStage(onnx_for_scale(ONNX_DIR, "t"), ENGINE_DIR, precision="int8")

    with pytest.raises(ValueError, match="no int8 kernel|QDQ"):
        RtmposeStage(Path("unused.py"), Path("unused.pth"), precision="int8")


def test_an_onnx_without_qdq_nodes_cannot_become_an_int8_engine():
    """TensorRT accepts BuilderFlag.INT8 on a graph with no QDQ nodes and
    quietly falls back to implicit quantization calibrated on nothing. That
    engine runs, and its rows would be labelled int8."""
    from cslr_cost.harness.engines import EngineBuildError, build_engine

    if not _have_engine_inputs("t"):
        pytest.skip("TensorRT, ONNX export or checkpoint unavailable")

    h, w = 256, 192
    one = (1, 3, h, w)
    with pytest.raises(EngineBuildError, match="QuantizeLinear"):
        build_engine(
            onnx_for_scale(ONNX_DIR, "t"),
            ENGINE_DIR / "should-not-exist.engine",
            "int8",
            [{"input": (one, one, one)}],
        )


def test_running_before_setup_is_an_error_not_a_measurement():
    stage = TrtRtmposeStage(onnx_for_scale(ONNX_DIR, "t"), ENGINE_DIR)
    with pytest.raises(RuntimeError, match="setup"):
        stage.run(object())


def test_config_names_the_runtime_before_and_after_setup():
    stage = TrtRtmposeStage(onnx_for_scale(ONNX_DIR, "s"), ENGINE_DIR, scale="s", precision="fp16")
    cfg = stage.config()
    assert cfg["frontend_runtime"] == "tensorrt"
    assert cfg["frontend_model"] == "rtmpose-s"
    assert cfg["frontend_precision"] == "fp16"
    assert cfg["frontend_engine"] is None      # no engine yet: must not claim one


# ---------------------------------------------------------------------------
# engine vs PyTorch, on real crops
# ---------------------------------------------------------------------------

def _have_engine_inputs(scale: str) -> bool:
    from cslr_cost.harness.engines import trt_version

    if trt_version() is None:
        return False
    return (
        torch.cuda.is_available()
        and onnx_for_scale(ONNX_DIR, scale).exists()
        and (RTMPOSE_DIR / f"rtmpose-{scale}_wholebody_256x192.pth").exists()
    )


def _real_clip() -> Optional[Path]:
    """A 360p ladder clip, else PHOENIX's native frames, else nothing."""
    clips = sorted(LADDER_360P.glob("*.mp4")) if LADDER_360P.is_dir() else []
    if clips:
        return clips[0]
    if PHOENIX.is_dir():
        seq = sorted(p for p in PHOENIX.iterdir() if p.is_dir())[0]
        return seq / "1" if (seq / "1").is_dir() else seq
    return None


@pytest.fixture(scope="module")
def real_crops():
    pytest.importorskip("cv2")
    from cslr_cost.harness.stages.crop import TopdownCropStage
    from cslr_cost.harness.stages.decode import VideoDecodeStage, open_video
    from cslr_cost.harness.stages.detect import WholeFrameDetectStage

    clip = _real_clip()
    if clip is None:
        pytest.skip("no ladder clip or PHOENIX frames to crop from")
    source = open_video(clip)
    decode, detect, crop = VideoDecodeStage(), WholeFrameDetectStage(), TopdownCropStage()
    for st in (decode, detect, crop):
        st.setup()
    crops = [crop.run(detect.run(decode.run(source))) for _ in range(8)]
    source.close()
    return crops


@pytest.mark.parametrize("precision, px_tol", [("fp32", 0.51), ("fp16", 1.01)])
def test_engine_reproduces_pytorch_keypoints(real_crops, precision, px_tol):
    """Same crops, same adapter, two runtimes.

    The decode is an argmax over a 0.5-px SimCC bin, so an export that is
    numerically faithful moves a keypoint by zero bins almost everywhere and by
    one bin (0.5 px on the canvas, scaled by the crop) at a near-tie. FP16 is
    allowed one more bin. The mean is held much tighter than the max: a
    systematic error -- a wrong mean/std, a missed unwarp -- shifts every
    keypoint, where near-ties move a few.
    """
    from cslr_cost.harness.stages.pose import RtmposeStage

    scale = "t"
    if not _have_engine_inputs(scale):
        pytest.skip("TensorRT, ONNX export or checkpoint unavailable")

    # strict FP32 in the reference, matching engines built with TF32 cleared
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = RtmposeStage(_config(scale), RTMPOSE_DIR / f"rtmpose-{scale}_wholebody_256x192.pth",
                           scale=scale, precision="fp32")
        ref.setup()
        expected = ref.run_batch(real_crops)
        ref.teardown()
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved

    stage = TrtRtmposeStage(onnx_for_scale(ONNX_DIR, scale), ENGINE_DIR,
                            scale=scale, precision=precision, opt_batch=4, max_batch=8)
    stage.setup()
    try:
        single = [stage.run(c) for c in real_crops]         # profile 0
        batched = stage.run_batch(real_crops)                # profile 1
        assert stage.config()["frontend_engine"] is not None
    finally:
        stage.teardown()

    for got in (single, batched):
        errs, in_bin = [], []
        for g, e, c in zip(got, expected, real_crops):
            assert g.shape == e.shape == (133, 3)
            # one canvas pixel, in frame pixels, on the coarser axis
            frame_px = float(np.max(np.asarray(c.scale) / np.asarray(c.input_size)))
            err = np.abs(g[:, :2] - e[:, :2]).max(axis=1)
            errs.append(err / frame_px)
            in_bin.append(err <= px_tol * frame_px + 1e-4)
        errs, in_bin = np.concatenate(errs), np.concatenate(in_bin)
        # near-ties may flip an argmax to a distant second peak, so the max is
        # not the test; a systematic error moves the median and the bulk
        assert np.median(errs) == 0.0, "systematic keypoint offset between runtimes"
        assert in_bin.mean() >= (0.98 if precision == "fp32" else 0.95)

        scores_g = np.concatenate([g[:, 2] for g in got])
        scores_e = np.concatenate([e[:, 2] for e in expected])
        assert np.median(np.abs(scores_g - scores_e)) <= (0.02 if precision == "fp16" else 2e-3)


# ---------------------------------------------------------------------------
# odd batches — a tactic constraint, not a model one
# ---------------------------------------------------------------------------

def _trt_stage(scale: str = "t", precision: str = "fp32"):
    stage = TrtRtmposeStage(
        onnx_for_scale(ONNX_DIR, scale), ENGINE_DIR,
        scale=scale, precision=precision, opt_batch=4, max_batch=8,
    )
    stage.setup()
    return stage


def test_every_batch_width_runs_including_odd_ones(real_crops):
    """Some engines refuse an odd batch above 1.

    Measured on this box: a `rtmpose-t` fp32 engine executes batches
    1, 2, 4, 6, ... 32 and fails every one of 3, 5, 7, ... 31 with
    `Cask (Failed to update runtime arguments.)`. The same ONNX at fp16, and
    `rtmpose-s` at both precisions, take odd batches happily, while
    `rtmpose-m` fp32 refuses them exactly as `-t` does — so it is a property of
    the selected tactic, not of the model. TensorRT picks tactics by timing
    them, so a rebuild can move an engine onto the bad side with no source
    change, which makes this something to handle rather than avoid.

    Earlier runs never met it: every measured clip was 128 frames at batch 32, so
    the final batch was never partial. Extraction reads whole clips of
    arbitrary length, so it is partial — and odd about half the time.
    """
    if not _have_engine_inputs("t"):
        pytest.skip("TensorRT, ONNX export or checkpoint unavailable")

    stage = _trt_stage()
    try:
        for width in range(1, len(real_crops) + 1):
            assert len(stage.run_batch(real_crops[:width])) == width, width
    finally:
        stage.teardown()


def test_splitting_an_odd_batch_does_not_change_the_keypoints(real_crops):
    """The split is (even, 1). It has to be exact, or it trades a crash for a
    silently different number — which is worse."""
    if not _have_engine_inputs("t"):
        pytest.skip("TensorRT, ONNX export or checkpoint unavailable")

    odd = real_crops[:7]
    stage = _trt_stage()
    try:
        batched = stage.run_batch(odd)
        even_then_one = stage.run_batch(odd[:6]) + [stage.run(odd[6])]
    finally:
        stage.teardown()

    for i, (a, b) in enumerate(zip(batched, even_then_one)):
        np.testing.assert_array_equal(a, b, err_msg=f"crop {i}")


def test_int8_compiles_the_qdq_export_and_the_float_arms_share_one_graph(tmp_path):
    """An int8 engine built from the FP32 file would quantize on nothing;
    `engines.py` refuses that, so the path has to point at the QDQ export."""
    assert onnx_for_scale(tmp_path, "m", "int8").name == "rtmpose-m.int8.qdq.onnx"
    assert onnx_for_scale(tmp_path, "m", "fp16") == onnx_for_scale(tmp_path, "m") \
        == tmp_path / "rtmpose-m.onnx"


def test_other_weights_resolve_to_their_own_export(tmp_path):
    assert onnx_for_scale(tmp_path, "m", "fp32", "coco-wb").name == "rtmpose-m.coco-wb.onnx"
    assert onnx_for_scale(tmp_path, "m", "int8", "coco-wb").name == "rtmpose-m.coco-wb.int8.qdq.onnx"
    assert onnx_for_scale(tmp_path, "m", "fp32", "dw-ucoco") == tmp_path / "rtmpose-m.onnx"

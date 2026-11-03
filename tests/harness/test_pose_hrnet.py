"""The TensorRT HRNet-w48 stage: flip and DARK decode against mmpose, and the
engine against MSKA's released keypoints.

Three layers, each skipped only for what it genuinely needs:

  * the pinned flip indices, the flip-back and the DARK decode against mmpose's
    own functions -- needs mmpose, not TensorRT;
  * the engine, fp32, on real PHOENIX frames against mmpose's PyTorch inference
    with flip test -- needs TensorRT, the ONNX export, the checkpoint, frames;
  * the same engine against the keypoints MSKA released -- the point of the
    stage: it must reproduce the front end MSKA was published with.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cslr_cost.harness.stages.pose_trt import (  # noqa: E402
    HRNET_ONNX,
    WHOLEBODY_FLIP_INDICES,
    TrtHrnetStage,
    decode_heatmaps_dark,
    flip_back_heatmaps,
)

SHARED = Path(os.environ.get("CSLR_DATA", "data"))
CHECKPOINT = SHARED / "coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
ONNX = Path("models/onnx") / HRNET_ONNX
FRAMES = SHARED / "phoenix2014/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px"
RELEASED = Path("MSKA/data/Phoenix-2014/phoenix-2014.dev")
ENGINE_DIR = Path("models/engines/test")


def _mmpose_config() -> Path:
    mmpose = pytest.importorskip("mmpose")
    from cslr_cost.export.export_hrnet_onnx import CONFIG_NAME

    return (Path(os.path.dirname(mmpose.__file__))
            / ".mim/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody" / CONFIG_NAME)


# ---------------------------------------------------------------------------
# flip and decode, no TensorRT needed
# ---------------------------------------------------------------------------

def test_flip_indices_match_mmpose():
    pytest.importorskip("mmpose")
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    from mmpose.datasets.datasets.wholebody import CocoWholeBodyDataset

    meta = parse_pose_metainfo(CocoWholeBodyDataset.METAINFO)
    assert list(WHOLEBODY_FLIP_INDICES) == list(meta["flip_indices"])


def test_flip_back_matches_mmpose():
    pytest.importorskip("mmpose")
    from mmpose.models.utils.tta import flip_heatmaps

    hm = torch.randn(3, 133, 96, 72)
    ref = flip_heatmaps(hm.clone(), flip_indices=list(WHOLEBODY_FLIP_INDICES),
                        flip_mode="heatmap", shift_heatmap=True)
    assert torch.equal(flip_back_heatmaps(hm.clone()), ref)


def _heatmaps(rng: np.random.Generator, n: int) -> np.ndarray:
    """Gaussian peaks at sub-pixel positions (sigma 3, as the codec trains),
    with noise; some joints near the border, some with no positive response."""
    h, w = 96, 72
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty((n, 133, h, w), dtype=np.float32)
    for i in range(n):
        for k in range(133):
            if k % 29 == 0:                       # border: DARK must skip it
                cx, cy = rng.choice([0.4, 1.3, w - 1.6]), rng.uniform(0, h - 1)
            else:
                cx, cy = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
            amp = rng.uniform(0.2, 1.0)
            g = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
            out[i, k] = g + rng.normal(0, 0.01, g.shape).astype(np.float32)
        out[i, 7] = -np.abs(out[i, 7]) - 0.01         # no positive response
    return out


def test_dark_decode_matches_mmpose():
    pytest.importorskip("mmpose")
    from mmpose.codecs import MSRAHeatmap

    codec = MSRAHeatmap(input_size=(288, 384), heatmap_size=(72, 96), sigma=3, unbiased=True)
    hm = _heatmaps(np.random.default_rng(0), 3)
    kpts, scores = decode_heatmaps_dark(hm.copy())
    for i in range(hm.shape[0]):
        ref_k, ref_s = codec.decode(hm[i].copy())
        np.testing.assert_allclose(kpts[i], ref_k[0], atol=1e-4)
        np.testing.assert_array_equal(scores[i], ref_s[0])
    assert (kpts[:, 7] == -4).all()                  # mmpose's -1 sentinel, x4


# ---------------------------------------------------------------------------
# the engine, on PHOENIX frames
# ---------------------------------------------------------------------------

def _have_engine_inputs() -> bool:
    from cslr_cost.harness.engines import trt_version

    if trt_version() is None:          # the image's wheel imports as tensorrt_bindings
        return False
    return ONNX.exists() and CHECKPOINT.exists() and FRAMES.exists() and RELEASED.exists()


@pytest.fixture(scope="module")
def phoenix_frames():
    if not _have_engine_inputs():
        pytest.skip("TensorRT, the HRNet export, its checkpoint or PHOENIX frames unavailable")
    cv2 = pytest.importorskip("cv2")
    with RELEASED.open("rb") as fh:
        released = pickle.load(fh)
    key = next(iter(released))
    rec = released[key]
    paths = sorted((FRAMES / rec["name"].strip("/")).glob("*.png"))[::9][:12]
    idx = [sorted((FRAMES / rec["name"].strip("/")).glob("*.png")).index(p) for p in paths]
    frames = [cv2.imread(str(p)) for p in paths]
    ref = np.asarray(rec["keypoint"], dtype=np.float32)[idx]
    return frames, ref


def _engine_keypoints(frames, flip_test=True):
    from cslr_cost.harness.stages.crop import TopdownCropStage
    from cslr_cost.harness.stages.detect import WholeFrameDetectStage
    from cslr_cost.harness.stages.pose import crop_size_for

    detect, crop = WholeFrameDetectStage(), TopdownCropStage(input_size=crop_size_for("hrnet-w48"))
    stage = TrtHrnetStage(ONNX, ENGINE_DIR, precision="fp32", flip_test=flip_test)
    for st in (detect, crop, stage):
        st.setup()
    crops = [crop.run(detect.run(f)) for f in frames]
    single = np.stack([stage.run(c) for c in crops])
    batch = np.stack(stage.run_batch(crops))
    stage.teardown()
    return single, batch


def test_engine_reproduces_mmpose_with_flip_test(phoenix_frames):
    """Same frames: mmpose's PyTorch inference with flip test, and the engine."""
    from mmpose.apis import inference_topdown, init_model

    frames, _ = phoenix_frames
    model = init_model(str(_mmpose_config()), str(CHECKPOINT), device="cuda")
    ref = []
    for f in frames:
        h, w = f.shape[:2]
        s = inference_topdown(model, f, bboxes=np.array([[0, 0, w, h]], dtype=np.float32))[0].pred_instances
        ref.append(np.concatenate([s.keypoints[0], s.keypoint_scores[0][:, None]], axis=1))
    ref = np.stack(ref)
    single, batch = _engine_keypoints(frames)
    for got in (single, batch):
        err = np.linalg.norm(got[..., :2] - ref[..., :2], axis=-1)
        assert np.median(err) < 0.01 and err.max() < 0.5, (np.median(err), err.max())
        assert np.median(np.abs(got[..., 2] - ref[..., 2])) < 1e-3


def test_engine_reproduces_mska_released_keypoints(phoenix_frames):
    """The reason the stage exists: the front end MSKA was published with."""
    frames, released = phoenix_frames
    single, _ = _engine_keypoints(frames)
    err = np.linalg.norm(single[..., :2] - released[..., :2], axis=-1)
    assert np.median(err) < 0.01, np.median(err)
    assert np.median(np.abs(single[..., 2] - released[..., 2])) < 1e-3
    # and without the flip test it does not: the flip is part of the front end
    noflip, _ = _engine_keypoints(frames, flip_test=False)
    assert np.median(np.linalg.norm(noflip[..., :2] - released[..., :2], axis=-1)) > 0.1


def test_every_crop_count_runs_and_odd_counts_split_cleanly(phoenix_frames):
    """Odd crop counts from 3 up are refused by the fp32 engine (as for some
    RTMPose engines); the stage splits them, and the split changes nothing."""
    from cslr_cost.harness.stages.crop import TopdownCropStage
    from cslr_cost.harness.stages.detect import WholeFrameDetectStage
    from cslr_cost.harness.stages.pose import crop_size_for

    frames, _ = phoenix_frames
    detect, crop = WholeFrameDetectStage(), TopdownCropStage(input_size=crop_size_for("hrnet-w48"))
    stage = TrtHrnetStage(ONNX, ENGINE_DIR, precision="fp32")
    for st in (detect, crop, stage):
        st.setup()
    crops = [crop.run(detect.run(f)) for f in frames][:7]
    single = np.stack([stage.run(c) for c in crops])
    for n in range(1, 8):
        got = np.stack(stage.run_batch(crops[:n]))
        np.testing.assert_allclose(got, single[:n], atol=1e-3)
    stage.teardown()

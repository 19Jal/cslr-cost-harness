"""End-to-end check of the pipeline's front half on real models and real video.

    decode -> detect -> crop -> pose -> postproc -> [assemble]

The counterpart to `test_backhalf_integration.py`. Skipped when the RTMPose
checkpoints or PHOENIX frames are absent, so the suite still runs on a machine
that has neither.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mmpose = pytest.importorskip("mmpose")
torch = pytest.importorskip("torch")

from cslr_cost.harness.fixtures import FixtureStore  # noqa: E402
from cslr_cost.harness.stages.base import BOUNDARIES, Pipeline  # noqa: E402
from cslr_cost.harness.stages.crop import TopdownCropStage  # noqa: E402
from cslr_cost.harness.stages.decode import VideoDecodeStage, open_video  # noqa: E402
from cslr_cost.harness.stages.detect import WholeFrameDetectStage  # noqa: E402
from cslr_cost.harness.stages.pose import (  # noqa: E402
    WHOLEBODY_KEYPOINTS,
    RtmposeStage,
)
from cslr_cost.harness.stages.postproc import (  # noqa: E402
    KeypointPostprocStage,
    assemble_sequence,
)

RTMPOSE_DIR = Path("data/rtmpose")
CHECKPOINT = RTMPOSE_DIR / "rtmpose-t_wholebody_256x192.pth"
MMPOSE_CONFIGS = (
    Path(os.path.dirname(mmpose.__file__))
    / ".mim/configs/wholebody_2d_keypoint/rtmpose/ubody"
)
CONFIG = MMPOSE_CONFIGS / "rtmpose-t_8xb64-270e_coco-ubody-wholebody-256x192.py"

_PHOENIX_DEV = (
    "phoenix2014/phoenix2014-release/phoenix-2014-multisigner"
    "/features/fullFrame-210x260px/dev"
)
# The corpus lives in the repo's (symlinked, gitignored) dataset/, mounted at
# /dataset in the container. The shared-data tree is checked as well, but on its
# own it silently skipped this whole file on the dev box: the corpus is not there.
PHOENIX = next(
    (p for p in (Path("/dataset") / _PHOENIX_DEV,
                 Path(os.environ.get("CSLR_DATA", "data")) / _PHOENIX_DEV)
     if p.is_dir()),
    Path("/dataset") / _PHOENIX_DEV,
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT.exists() and CONFIG.exists() and PHOENIX.is_dir()),
    reason="RTMPose checkpoint or PHOENIX frames not available",
)

T = 8


def _clip_dir() -> Path:
    seq = sorted(p for p in PHOENIX.iterdir() if p.is_dir())[0]
    return seq / "1" if (seq / "1").is_dir() else seq


@pytest.fixture(scope="module")
def pipeline_and_source():
    source = open_video(_clip_dir())
    pipe = Pipeline(
        [
            VideoDecodeStage(),
            WholeFrameDetectStage(),
            TopdownCropStage(),
            RtmposeStage(CONFIG, CHECKPOINT, scale="t",
                         device="cuda" if torch.cuda.is_available() else "cpu"),
            KeypointPostprocStage(frame_wh=source.wh),
        ],
        assemble=assemble_sequence,
    )
    pipe.setup()
    yield pipe, source
    pipe.teardown()


def test_the_front_half_is_connected():
    """A disconnected pipeline still runs and still produces plausible-looking
    numbers, so connectivity is checked at construction rather than discovered
    in a results table."""
    pipe = Pipeline(
        [VideoDecodeStage(), WholeFrameDetectStage(), TopdownCropStage()],
    )
    assert [s.name for s in pipe] == ["decode", "detect", "crop"]
    assert pipe.input_key == "video" and pipe.output_key == "crops"


def test_every_boundary_artifact_is_captured(pipeline_and_source):
    pipe, source = pipeline_and_source
    source.position = 0
    store = FixtureStore.capture(pipe, source, T)
    for key in ("video", "frames", "boxes", "crops", "keypoints_raw", "keypoints"):
        assert key in store, f"boundary '{key}' was not captured"
        assert key in BOUNDARIES
    assert len(store["frames"]) == T


def test_extractor_emits_the_layout_the_head_consumes(pipeline_and_source):
    """MSKA's DSTA indexes overlapping groups out of the full 133-point layout,
    so a reduced set indexes out of bounds and surfaces far downstream as
    CUDNN_STATUS_INTERNAL_ERROR."""
    pipe, source = pipeline_and_source
    source.position = 0
    store = FixtureStore.capture(pipe, source, T)
    kp = store["keypoints_raw"][0]
    assert kp.shape == (WHOLEBODY_KEYPOINTS, 3)
    assert kp.dtype == np.float32


def test_keypoints_come_back_in_frame_coordinates(pipeline_and_source):
    """SimCC decodes to the 192x256 canvas. Skipping the unwarp does not raise
    -- the values stay finite and roughly plausible -- so it is asserted."""
    pipe, source = pipeline_and_source
    source.position = 0
    store = FixtureStore.capture(pipe, source, T)
    kp = np.stack(store["keypoints_raw"])
    w, h = source.wh

    # The warp canvas is padded beyond the frame, so some points legitimately
    # land outside it; what must not happen is keypoints still sitting in crop
    # space, which for a 210x260 frame would cluster inside 192x256.
    assert kp[..., 0].max() > w * 0.75, "x looks like crop space, not frame space"
    assert np.median(kp[..., 0]) < w * 1.5
    assert np.median(kp[..., 1]) < h * 1.5
    assert 0.0 <= kp[..., 2].min() and kp[..., 2].max() < 1.5


def test_assembled_sequence_has_the_head_layout(pipeline_and_source):
    pipe, source = pipeline_and_source
    source.position = 0
    seq = pipe.run(source, T)
    assert seq.shape == (1, 3, T, WHOLEBODY_KEYPOINTS)
    assert seq.dtype == torch.float32


def test_isolated_replay_reproduces_the_composed_result(pipeline_and_source):
    """What makes P1's isolation protocol legitimate: replaying a stage
    against its captured input must do the same work it did in the composed
    pass. If it does not, the isolated sum and the composed total disagree for
    reasons that have nothing to do with data movement."""
    pipe, source = pipeline_and_source
    source.position = 0
    store = FixtureStore.capture(pipe, source, T)

    pose = next(s for s in pipe if s.name == "pose")
    for i in (0, 1, T - 1):
        replayed = pose.run(store.input_for(pose, i))
        np.testing.assert_allclose(replayed, store["keypoints_raw"][i], rtol=0, atol=0)


def test_replay_cycles_through_real_frames_not_one_buffer(pipeline_and_source):
    """Hammering a single buffer measures an artificially cache-hot path and
    understates the stage, which then shows up as unexplained residual."""
    pipe, source = pipeline_and_source
    source.position = 0
    store = FixtureStore.capture(pipe, source, T)
    pose = next(s for s in pipe if s.name == "pose")
    assert len(store.units_for(pose)) == T
    assert store.input_for(pose, T) is store.input_for(pose, 0)     # wraps
    assert store.input_for(pose, 1) is not store.input_for(pose, 0)


def test_precision_must_be_a_torch_dtype_not_int8():
    with pytest.raises(ValueError, match="TensorRT"):
        RtmposeStage(CONFIG, CHECKPOINT, precision="int8")


def test_running_before_setup_is_an_error_not_a_measurement():
    stage = RtmposeStage(CONFIG, CHECKPOINT)
    with pytest.raises(RuntimeError, match="setup"):
        stage.run(object())

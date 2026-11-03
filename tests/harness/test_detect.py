"""detect stage: the whole-frame degenerate case, and the fallback counter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cslr_cost.harness.stages.detect import (
    PERSON_LABEL,
    RtmdetPersonStage,
    WholeFrameDetectStage,
)

DET = Path("data/rtmdet")
DET_CONFIG = DET / "rtmdet_nano_320-8xb32_coco-person.py"
DET_WEIGHTS = DET / "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"

FRAME_W, FRAME_H = 210, 260


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


# -- whole frame ------------------------------------------------------------


def test_whole_frame_box_is_the_frame():
    frame, box = WholeFrameDetectStage().run(_frame())
    np.testing.assert_allclose(box, [0, 0, FRAME_W, FRAME_H])


def test_output_carries_the_frame_alongside_the_box():
    """`crop` needs both. Passing the box alone would force crop to hold a
    reference it was never given, which breaks fixture replay — the store keys
    strictly on boundary artifacts."""
    src = _frame()
    frame, box = WholeFrameDetectStage().run(src)
    assert frame is src
    assert box.shape == (4,)


def test_whole_frame_says_so_in_the_row():
    """A degenerate detector reported as a real one would understate the front
    end, which is the error a whole-pipeline account is meant to avoid."""
    assert WholeFrameDetectStage().config()["detector_policy"] == "whole-frame"


def test_stage_declares_its_boundary_and_granularity():
    stage = WholeFrameDetectStage()
    assert (stage.name, stage.input_key, stage.output_key) == ("detect", "frames", "boxes")
    assert stage.granularity == "frame"


# -- real detector ----------------------------------------------------------

rtmdet = pytest.mark.skipif(
    not (DET_CONFIG.exists() and DET_WEIGHTS.exists()),
    reason="RTMDet checkpoint not available",
)


@pytest.fixture(scope="module")
def detector():
    pytest.importorskip("mmdet")
    stage = RtmdetPersonStage(DET_CONFIG, DET_WEIGHTS)
    stage.setup()
    yield stage
    stage.teardown()


@rtmdet
def test_detector_returns_a_box_inside_the_frame(detector):
    frame = _frame()
    _, box = detector.run(frame)
    assert box.shape == (4,) and box.dtype == np.float32
    assert box[0] < box[2] and box[1] < box[3]


@rtmdet
def test_a_missed_detection_is_counted_not_swallowed(detector):
    """A detector that fires on 60% of frames and whole-frames the rest would
    otherwise report the same box quality as one that works, and the pose
    numbers downstream would carry an error with no trace of where it entered."""
    detector.fallback_frames = 0
    detector.frames_seen = 0
    original = detector.score_thr
    detector.score_thr = 1.1          # nothing can pass
    try:
        _, box = detector.run(_frame())
    finally:
        detector.score_thr = original
    np.testing.assert_allclose(box, [0, 0, FRAME_W, FRAME_H])
    assert detector.fallback_frames == 1
    assert detector.fallback_rate == 1.0


@rtmdet
def test_fallback_rate_is_none_before_any_frame():
    stage = RtmdetPersonStage(DET_CONFIG, DET_WEIGHTS)
    assert stage.fallback_rate is None


@rtmdet
def test_running_before_setup_is_an_error_not_a_measurement():
    """Model loading must never land inside a timed window."""
    stage = RtmdetPersonStage(DET_CONFIG, DET_WEIGHTS)
    with pytest.raises(RuntimeError, match="setup"):
        stage.run(_frame())


def test_person_label_is_explicit():
    """The nano person checkpoint has one class, so this is 0 either way — kept
    named so a swap to a general-purpose 80-class detector does not silently
    start cropping to furniture."""
    assert PERSON_LABEL == 0

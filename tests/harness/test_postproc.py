"""Tests for keypoint normalisation — the pipeline's likeliest silent failure.

If this convention is wrong, the head still runs and still produces plausible
glosses, at meaningfully worse WER, with no error anywhere. So the checks here
are against MSKA's arithmetic specifically, not against "something reasonable".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cslr_cost.harness.stages.postproc import (
    HEAD_KEYPOINTS,
    PHOENIX_2014_WH,
    KeypointPostprocStage,
    assemble_sequence,
    normalise_clip,
)

W, H = PHOENIX_2014_WH


def frame(k: int = HEAD_KEYPOINTS) -> np.ndarray:
    rng = np.random.default_rng(0)
    kp = np.empty((k, 3), dtype=np.float32)
    kp[:, 0] = rng.uniform(0, W, k)
    kp[:, 1] = rng.uniform(0, H, k)
    kp[:, 2] = rng.uniform(0, 1, k)
    return kp


def test_matches_mska_arithmetic_exactly():
    """Reproduces datasets.py:102-106 on the same input."""
    kp = frame()
    got = KeypointPostprocStage().run(kp)

    ref = kp.copy()
    ref[:, 0] /= W
    ref[:, 1] = H - ref[:, 1]
    ref[:, 1] /= H
    ref[:, :2] = (ref[:, :2] - 0.5) / 0.5

    np.testing.assert_allclose(got, ref, rtol=1e-6)


def test_output_lands_in_minus_one_to_one():
    got = KeypointPostprocStage().run(frame())
    assert got[:, :2].min() >= -1.001 and got[:, :2].max() <= 1.001


def test_y_is_flipped_not_merely_scaled():
    """`h - y` moves the origin to bottom-left. Scaling without the flip is the
    easy mistake, and it produces a vertically mirrored skeleton that still
    decodes to plausible glosses."""
    kp = np.zeros((HEAD_KEYPOINTS, 3), dtype=np.float32)
    kp[:, 1] = 0.0                       # top of the frame in pixel coords
    got = KeypointPostprocStage().run(kp)
    assert got[0, 1] == pytest.approx(1.0), "y=0 (top) must map to +1 (bottom-up)"

    kp[:, 1] = H
    got = KeypointPostprocStage().run(kp)
    assert got[0, 1] == pytest.approx(-1.0), "y=h must map to -1"


def test_confidence_channel_is_untouched():
    kp = frame()
    got = KeypointPostprocStage().run(kp)
    np.testing.assert_allclose(got[:, 2], kp[:, 2], rtol=1e-6)


def test_normalises_against_the_frame_it_was_measured_in():
    """The divisor is the frame's own size, not a corpus constant. Keypoints
    from an upscaled 720p clip divided by 210x260 land far outside [-1, 1]."""
    kp = frame()
    kp[:, 0] *= 4
    kp[:, 1] *= 4

    wrong = KeypointPostprocStage().run(kp)
    assert wrong[:, :2].max() > 2.0, "mis-normalisation should be visible"

    right = KeypointPostprocStage(frame_wh=(W * 4, H * 4)).run(kp)
    assert right[:, :2].min() >= -1.001 and right[:, :2].max() <= 1.001


def test_reduced_keypoint_set_is_refused_with_the_real_reason():
    """MSKA indexes groups out of the full 133 layout, so fewer points index
    out of bounds — surfacing as CUDNN_STATUS_INTERNAL_ERROR."""
    with pytest.raises(ValueError, match="CUDNN_STATUS_INTERNAL_ERROR"):
        KeypointPostprocStage().run(frame(79))


def test_a_whole_clip_passed_to_a_per_frame_stage_is_refused():
    clip = np.zeros((30, HEAD_KEYPOINTS, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="per-frame"):
        KeypointPostprocStage().run(clip)


def test_index_map_reorders_before_normalising():
    kp = frame()
    perm = np.arange(HEAD_KEYPOINTS)[::-1].copy()
    got = KeypointPostprocStage(index_map=perm).run(kp)
    ref = KeypointPostprocStage().run(kp[perm])
    np.testing.assert_allclose(got, ref, rtol=1e-6)


def test_assembly_produces_the_head_input_layout():
    """(T, K, C) -> (1, C, T, K): batch, channels, time, joints."""
    units = [KeypointPostprocStage().run(frame()) for _ in range(12)]
    seq = assemble_sequence(units)
    assert seq.shape == (1, 3, 12, HEAD_KEYPOINTS)
    assert seq.dtype == torch.float32


def test_assembly_preserves_frame_order():
    units = []
    for t in range(5):
        kp = np.zeros((HEAD_KEYPOINTS, 3), dtype=np.float32)
        kp[:, 2] = t                      # tag each frame in the confidence channel
        units.append(kp)
    seq = assemble_sequence(units)
    assert [float(seq[0, 2, t, 0]) for t in range(5)] == [0, 1, 2, 3, 4]


def test_empty_sequence_is_refused():
    with pytest.raises(ValueError, match="empty"):
        assemble_sequence([])


# ---------------------------------------------------------------------------
# normalise_clip — the vectorised twin, pinned to the stage
# ---------------------------------------------------------------------------

def clip(t: int = 7, k: int = HEAD_KEYPOINTS) -> np.ndarray:
    rng = np.random.default_rng(1)
    kp = np.empty((t, k, 3), dtype=np.float32)
    kp[:, :, 0] = rng.uniform(0, W, (t, k))
    kp[:, :, 1] = rng.uniform(0, H, (t, k))
    kp[:, :, 2] = rng.uniform(0, 1, (t, k))
    return kp


def test_normalise_clip_equals_the_stage_frame_by_frame():
    """The pin. Two implementations of MSKA's normalisation is the silent
    failure this module's docstring is about; this is what keeps them one."""
    seq = clip()
    stage = KeypointPostprocStage()

    got = normalise_clip(seq)
    for t in range(seq.shape[0]):
        np.testing.assert_array_equal(got[t], stage.run(seq[t]))


def test_normalise_clip_handles_the_released_keypoint_domain():
    """MSKA's released HRNet keypoints are unclamped: coordinates fall far
    outside the frame (x down to -2540 on a 210px frame) and confidences exceed
    1.0. The normalisation is affine, so it must carry them through rather than
    clip them — the head was trained on exactly those values."""
    seq = clip(t=2)
    seq[0, 0] = (-2540.1157, -2025.8203, 1.0776738)

    got = normalise_clip(seq)

    assert got[0, 0, 0] < -20          # far outside [-1, 1], and that is correct
    assert got[0, 0, 2] == pytest.approx(1.0776738)


def test_normalise_clip_refuses_a_single_frame():
    """(K, 3) is the stage's shape, not this function's. Passing one frame here
    would normalise across joints instead of across the clip."""
    with pytest.raises(ValueError, match="clip shaped"):
        normalise_clip(frame())


def test_normalise_clip_accepts_a_torch_tensor():
    """Released keypoints arrive as torch tensors, straight out of the pickle."""
    seq = clip(t=3)
    np.testing.assert_array_equal(normalise_clip(torch.from_numpy(seq)), normalise_clip(seq))


def test_normalise_clip_applies_the_index_map_before_normalising():
    seq = clip(t=2, k=HEAD_KEYPOINTS)
    order = np.roll(np.arange(HEAD_KEYPOINTS), 3)

    mapped = normalise_clip(seq, index_map=order)
    plain = normalise_clip(seq)

    np.testing.assert_array_equal(mapped, plain[:, order])

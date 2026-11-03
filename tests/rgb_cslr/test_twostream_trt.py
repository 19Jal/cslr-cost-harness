"""TwoStream-SLR in the harness: the preprocessing that would fail silently.

  * The 79 joints must be TwoStream's (sorted(use_keypoints): face_others_1_3, hand,
    mouth_half, pose), or the keypoint stream reads the wrong heatmap channels.
  * Keypoints from a 1080-line frame must be rescaled to the 210x260 frame TwoStream's
    heatmaps are drawn on (raw_size 260x210), or every Gaussian lands off-target.
  * The heatmap must equal TwoStream's gen_gaussian_hmap_op at raw size, then resized.
  * The clip is trimmed to a multiple of 4 frames, as its evaluation selects them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cslr_cost.harness.stages import twostream_trt as ts


def test_the_79_joints_are_twostreams():
    assert len(ts.JOINTS_79) == 79 and len(set(ts.JOINTS_79)) == 79
    assert ts.JOINTS_79[:3] == [23, 26, 29]            # face_others_1_3 first (sorted order)
    assert ts.JOINTS_79[16:58] == list(range(91, 133))  # both hands
    assert ts.JOINTS_79[-11:] == list(range(11))         # pose last


def _their_heatmap(coords, raw=(260, 210), sigma=8.0, out=112):
    """TwoStream's gen_gaussian_hmap_op (openpose-style grid), then a bilinear resize."""
    H, W = raw
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cy, cx = coords[..., 1][..., None, None], coords[..., 0][..., None, None]
    hm = torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))
    hm = torch.nn.functional.interpolate(hm, size=(out, out), mode="bilinear", align_corners=False)
    return (hm - 0.5) / 0.5


def test_heatmaps_match_twostreams_rendering():
    torch.manual_seed(0)
    kp = torch.rand(2, 79, 2) * torch.tensor([210.0, 260.0])
    assert torch.allclose(ts.render_heatmaps(kp), _their_heatmap(kp), atol=1e-6)


def test_keypoints_are_rescaled_from_the_measured_frame_to_210x260():
    post = ts.TwoStreamPostprocStage(frame_wh=(872, 1080), device="cpu")
    kp = np.zeros((133, 3), np.float32)
    kp[:, 0], kp[:, 1] = 872 / 2, 1080 / 2                 # frame centre at 1080p
    _, hm = post.run((kp, torch.zeros(3, 224, 224)))
    centre = _their_heatmap(torch.tensor([[[105.0, 130.0]] * 79]))[0]
    assert torch.allclose(hm, centre, atol=1e-5)


@pytest.mark.parametrize("t,kept", [(4, 4), (7, 4), (128, 128), (131, 128)])
def test_clip_is_trimmed_to_a_multiple_of_4(t, kept):
    units = [(torch.zeros(3, 2, 2), torch.zeros(79, 2, 2))] * t
    rgb, hm = ts.assemble_twostream(units)
    assert rgb.shape == (1, 3, kept, 2, 2) and hm.shape == (1, 79, kept, 2, 2)


def test_keypoint_stream_alone_assembles_heatmaps_only():
    """twostream-kp: postproc passes no RGB, and the sequence is (None, heatmaps)."""
    post = ts.TwoStreamPostprocStage(frame_wh=(210, 260), device="cpu")
    kp = np.zeros((133, 3), np.float32)
    units = [post.run((kp, None)) for _ in range(131)]
    rgb, hm = ts.assemble_twostream(units)
    assert rgb is None and hm.shape == (1, 79, 128, 112, 112)


def test_keypoint_stream_head_names_itself_and_needs_no_rgb_profile():
    head = ts.TrtTwoStreamHeadStage("x.onnx", "engines", keypoint_only=True)
    assert head.config()["head_model"] == "twostream-kp"
    assert ts.TrtTwoStreamHeadStage("x.onnx", "engines").config()["head_model"] == "twostream-slr"

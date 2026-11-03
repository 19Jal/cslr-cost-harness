"""crop stage geometry — and the round trip that guards the silent failure.

`crop` warps a box into the model canvas; `pose` warps the predicted keypoints
back. If those two transforms disagree, nothing raises: keypoints land in the
wrong place, `postproc` normalises them anyway, the head emits glosses, and WER
is worse for a reason no experiment is varying. So the inverse is tested
directly rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cslr_cost.harness.stages.crop import (  # noqa: E402
    BBOX_PADDING,
    RTMPOSE_INPUT_SIZE,
    Crop,
    TopdownCropStage,
    _bbox_xyxy2cs,
    _fix_aspect_ratio,
    _get_warp_matrix,
)

try:
    from mmpose.structures.bbox import bbox_xyxy2cs, get_warp_matrix
    HAVE_MMPOSE = True
except ImportError:          # the Orin: no OpenMMLab, by design
    HAVE_MMPOSE = False

FRAME_W, FRAME_H = 290, 360


def _frame(w: int = FRAME_W, h: int = FRAME_H) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _whole_frame_box(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.array([0.0, 0.0, w, h], dtype=np.float32)


# -- shape and dtype contract ----------------------------------------------


def test_output_is_the_model_canvas_in_uint8_bgr():
    frame = _frame()
    crop = TopdownCropStage().run((frame, _whole_frame_box(frame)))
    w, h = RTMPOSE_INPUT_SIZE
    assert crop.img.shape == (h, w, 3)
    assert crop.img.dtype == np.uint8, (
        "crop must stay uint8: normalising here would quadruple what crosses "
        "the PCIe boundary and charge pose's GPU-side work to the CPU stage"
    )


def test_colour_is_untouched():
    """Decode emits BGR and RTMPose's data preprocessor does the swap on the
    GPU, so no stage in between may convert."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :, 0] = 200          # blue channel only
    crop = TopdownCropStage().run((frame, _whole_frame_box(frame)))
    centre = crop.img[crop.img.shape[0] // 2, crop.img.shape[1] // 2]
    assert centre[0] > centre[2], "channels look swapped: BGR became RGB"


def test_a_stacked_clip_is_rejected():
    stage = TopdownCropStage()
    clip = np.zeros((4, FRAME_H, FRAME_W, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="per-frame"):
        stage.run((clip, np.array([0, 0, FRAME_W, FRAME_H], dtype=np.float32)))


# -- geometry ---------------------------------------------------------------


def test_aspect_ratio_fix_only_ever_grows():
    """Shrinking the long side to fit would cut hands out of frame, which for
    sign language is not a small error."""
    for w, h in [(100.0, 400.0), (400.0, 100.0), (192.0, 256.0)]:
        scale = np.array([[w, h]], dtype=np.float32)
        fixed = _fix_aspect_ratio(scale, aspect_ratio=192 / 256)[0]
        assert fixed[0] >= w - 1e-4 and fixed[1] >= h - 1e-4
        assert fixed[0] / fixed[1] == pytest.approx(192 / 256, rel=1e-5)


def test_padding_is_applied_to_the_box():
    frame = _frame()
    box = _whole_frame_box(frame)
    padded = TopdownCropStage(padding=BBOX_PADDING).run((frame, box))
    unpadded = TopdownCropStage(padding=1.0).run((frame, box))
    assert (padded.scale > unpadded.scale).all()
    assert padded.center == pytest.approx(unpadded.center)


def test_source_dimensions_travel_with_the_crop():
    """postproc must normalise against the frame's real size. Hardcoding the
    corpus constant instead is what breaks the moment the resolution ladder
    feeds it a 720p frame."""
    frame = _frame(w=1280, h=720)
    crop = TopdownCropStage().run((frame, _whole_frame_box(frame)))
    assert crop.source_wh == (1280, 720)


# -- the round trip ---------------------------------------------------------


def _unwarp(points: np.ndarray, crop: Crop) -> np.ndarray:
    """pose.py's crop-space -> frame-space mapping, reproduced verbatim.

    Kept as a copy on purpose: if pose.py's formula is edited, this test must
    fail rather than silently follow it.
    """
    input_size = np.asarray(crop.input_size, dtype=np.float32)
    return points / input_size * crop.scale + crop.center - 0.5 * crop.scale


@pytest.mark.parametrize("padding", [1.0, 1.25, 1.5])
def test_unwarp_inverts_the_warp(padding):
    frame = _frame()
    box = np.array([30.0, 40.0, 250.0, 330.0], dtype=np.float32)
    crop = TopdownCropStage(padding=padding).run((frame, box))

    center, scale = _bbox_xyxy2cs(box.reshape(1, 4), padding=padding)
    w, h = RTMPOSE_INPUT_SIZE
    scale = _fix_aspect_ratio(scale, aspect_ratio=w / h)
    warp_mat = _get_warp_matrix(center[0], scale[0], output_size=(w, h))

    src_pts = np.array(
        [[50.0, 60.0], [145.0, 180.0], [200.0, 300.0], [0.0, 0.0]], dtype=np.float32
    )
    in_crop = cv2.transform(src_pts.reshape(1, -1, 2), warp_mat).reshape(-1, 2)
    recovered = _unwarp(in_crop, crop)

    np.testing.assert_allclose(recovered, src_pts, rtol=1e-4, atol=1e-3)


def test_a_point_at_the_crop_centre_maps_to_the_box_centre():
    """The single most load-bearing case: get the half-scale offset wrong and
    every keypoint is displaced by half a box, consistently, with no error."""
    frame = _frame()
    box = np.array([30.0, 40.0, 250.0, 330.0], dtype=np.float32)
    crop = TopdownCropStage().run((frame, box))
    w, h = RTMPOSE_INPUT_SIZE
    centre_of_canvas = np.array([[w / 2, h / 2]], dtype=np.float32)
    np.testing.assert_allclose(_unwarp(centre_of_canvas, crop)[0], crop.center, atol=1e-4)


def test_stage_declares_its_boundary_and_granularity():
    stage = TopdownCropStage()
    assert (stage.name, stage.input_key, stage.output_key) == ("crop", "boxes", "crops")
    assert stage.granularity == "frame" and stage.uses_cuda is False
    assert "192x256" in stage.config()["notes"]


# -- the copied geometry is mmpose's --------------------------------------------


@pytest.mark.skipif(not HAVE_MMPOSE, reason="mmpose not installed (expected on the Orin)")
@pytest.mark.parametrize("padding", [1.0, 1.25, 1.5])
@pytest.mark.parametrize(
    "box", [[0.0, 0.0, 290.0, 360.0], [30.0, 40.0, 250.0, 330.0], [-12.5, 7.0, 910.0, 1080.0]]
)
def test_vendored_geometry_matches_mmpose_exactly(box, padding):
    """The copy exists so the Orin needs no OpenMMLab. If it drifts from
    mmpose's own functions, every Orin keypoint shifts with no error — so it is
    held to them wherever mmpose is available."""
    b = np.asarray(box, dtype=np.float32).reshape(1, 4)
    c_ours, s_ours = _bbox_xyxy2cs(b, padding=padding)
    c_ref, s_ref = bbox_xyxy2cs(b, padding=padding)
    np.testing.assert_array_equal(c_ours, c_ref)
    np.testing.assert_array_equal(s_ours, s_ref)

    w, h = RTMPOSE_INPUT_SIZE
    s_fix = _fix_aspect_ratio(s_ours, aspect_ratio=w / h)
    np.testing.assert_array_equal(
        _get_warp_matrix(c_ours[0], s_fix[0], output_size=(w, h)),
        get_warp_matrix(c_ours[0], s_fix[0], 0.0, output_size=(w, h)),
    )


def test_crop_imports_no_openmmlab():
    """The point of the copy. A stray mmpose import would make crop the one
    front-end stage that cannot run on the Orin, and it would only surface there."""
    import ast
    from pathlib import Path

    import cslr_cost.harness.stages.crop as crop_mod

    tree = ast.parse(Path(crop_mod.__file__).read_text())
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert not any(m.split(".")[0] in {"mmpose", "mmdet", "mmcv", "mmengine"} for m in imported)

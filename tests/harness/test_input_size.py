"""Regime B: a second input size, threaded from one table.

The crop is the same size at every ladder rung, so 256x192 alone would be the only
cost regime. DWPose-l at 384x288 is a second one, and the crop,
the engine profile and the row's `frontend_input` must all agree on it.
"""

import pytest

from cslr_cost.harness.stages.pose import RTMPOSE_SCALES, crop_size_for, input_hw_for


@pytest.mark.parametrize("scale", ["t", "s", "m"])
def test_the_256x192_family(scale):
    assert input_hw_for(scale) == (256, 192)
    assert crop_size_for(scale) == (192, 256)


def test_dwpose_l_is_its_own_cost_regime():
    assert "l" in RTMPOSE_SCALES
    assert input_hw_for("l") == (384, 288)
    assert crop_size_for("l") == (288, 384)


def test_an_unknown_scale_is_refused():
    with pytest.raises(ValueError):
        input_hw_for("xl")


def test_the_row_records_the_input_it_ran_at(tmp_path):
    from cslr_cost.harness.stages.pose_trt import TrtRtmposeStage

    cfg = TrtRtmposeStage(tmp_path / "rtmpose-l.onnx", tmp_path, scale="l").config()
    assert cfg["frontend_input"] == "384x288" and cfg["frontend_model"] == "rtmpose-l"


def test_only_dwpose_l_pins_its_norms_to_fp32(tmp_path, monkeypatch):
    """ScaleNorm overflowed FP16 for DWPose-l (-6.7 AP, +20.8 WER) and not for
    t/s/m (<=0.001 AP). Pinning only `l` leaves every t/s/m engine -- and every
    row measured with one -- exactly as it was."""
    from cslr_cost.harness import engines
    from cslr_cost.harness.stages import pose_trt

    seen = {}

    def fake_ensure(onnx, cache, precision, profiles, fp32_layers=()):
        seen[onnx.name] = tuple(fp32_layers)
        raise RuntimeError("stop before building")

    monkeypatch.setattr(engines, "ensure_engine", fake_ensure)
    for scale in ("m", "l"):
        stage = pose_trt.TrtRtmposeStage(tmp_path / f"rtmpose-{scale}.onnx", tmp_path, scale=scale, precision="fp16")
        try:
            stage.setup()
        except RuntimeError:
            pass
    assert seen["rtmpose-m.onnx"] == ()
    assert set(seen["rtmpose-l.onnx"]) == {"/mlp/mlp.0/", "/gau/ln/"}

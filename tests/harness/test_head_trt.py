"""The TensorRT MSKA head: it must be the four-head ensemble, and match PyTorch.

The engine test compares `TrtMskaHeadStage` with `MskaHeadStage` on the same
keypoint sequence, at a length other than the one the export was traced at,
and then through `ctc` -- because agreement on log-scores is the means, and the
same glosses are what the export must actually preserve.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cslr_cost.harness.stages.head_trt import (  # noqa: E402
    ENSEMBLE_OUTPUTS,
    TrtMskaHeadStage,
)

SHARED = Path(os.environ.get("CSLR_DATA", "data"))
ENSEMBLE_ONNX = Path("models/onnx/mska-phoenix2014-ensemble.onnx")
FUSE_ONNX = SHARED / "onnx/mska-phoenix2014.onnx"
MSKA_ROOT = Path("MSKA")
CONFIG = MSKA_ROOT / "configs/phoenix-2014_s2g.yaml"
WEIGHTS = MSKA_ROOT / "pretrained_models/best.pth"
ENGINE_DIR = Path("models/engines/test")


def test_ensemble_order_is_recognition_forwards():
    # left + right + body + fuse (recognition.py). Addition commutes exactly
    # only in exact arithmetic; matching the order keeps fp32 parity bit-close.
    assert ENSEMBLE_OUTPUTS == ("logits_left", "logits_right", "logits_body", "logits_fuse")


def test_int8_is_refused():
    with pytest.raises(ValueError, match="calibration"):
        TrtMskaHeadStage(ENSEMBLE_ONNX, ENGINE_DIR, precision="int8")


def test_opt_shape_must_sit_inside_the_profile():
    with pytest.raises(ValueError):
        TrtMskaHeadStage(ENSEMBLE_ONNX, ENGINE_DIR, min_frames=64, opt_frames=32)


def test_model_label_is_not_claimed_before_the_graph_is_inspected():
    cfg = TrtMskaHeadStage(ENSEMBLE_ONNX, ENGINE_DIR).config()
    assert cfg["head_model"] is None
    assert cfg["head_runtime"] == "tensorrt"


def _have(*paths: Path) -> bool:
    from cslr_cost.harness.engines import trt_version

    if trt_version() is None:
        return False
    return torch.cuda.is_available() and all(p.exists() for p in paths)


def _sequence(t: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    xy = torch.rand(1, 2, t, 133, generator=g) * 2 - 1
    conf = torch.rand(1, 1, t, 133, generator=g) * 0.7 + 0.3
    return torch.cat([xy, conf], dim=1)


@pytest.fixture(scope="module")
def reference():
    if not _have(ENSEMBLE_ONNX, WEIGHTS, CONFIG):
        pytest.skip("TensorRT, ensemble export or MSKA weights unavailable")
    from cslr_cost.harness.stages.head import MskaHeadStage

    # Strict FP32 in the reference: engines are built with TF32 cleared, and
    # torch's default TF32 convolutions alone move log-probs by ~8e-3.
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    st = MskaHeadStage(MSKA_ROOT, CONFIG, WEIGHTS)
    st.setup()
    yield st
    st.teardown()
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


#: FP32 is held tight because TF32 is cleared on both sides, so the only
#: difference left is kernel order. The FP16 figure is measured, not chosen:
#: on four real PHOENIX clips the engine's live-class log-probabilities differ
#: from PyTorch by 0.09-0.10 while every decoded gloss is identical, and on
#: synthetic input by 0.074. 0.15 leaves headroom above what FP16 actually
#: does without admitting a run that has started deciding differently -- which
#: is what the gloss assertion below catches regardless.
@pytest.mark.parametrize("precision, tol", [("fp32", 1e-3), ("fp16", 0.15)])
def test_engine_reproduces_the_pytorch_ensemble(reference, precision, tol):
    from cslr_cost.harness.stages.ctc import CtcDecodeStage

    stage = TrtMskaHeadStage(ENSEMBLE_ONNX, ENGINE_DIR, precision=precision)
    stage.setup()
    try:
        assert stage.config()["head_model"] == "mska-s2g"
        assert stage.vocab_size == reference.vocab_size

        seq = _sequence(100)                   # not the traced 64, not the opt 128
        want = reference.run(seq.cuda()).float()
        got = stage.run(seq)
        assert got.shape == want.shape
        assert torch.isfinite(got).all(), "FP16 ensemble underflowed to -inf"

        # compare as distributions: the ensemble's log of a sum of 4 softmaxes
        # is not normalised, and the decoder renormalises anyway
        delta = (got.log_softmax(-1) - want.log_softmax(-1)).abs()
        top = want.softmax(-1) > 1e-4           # where the beam search actually looks
        assert delta[top].max().item() <= tol

        ctc = CtcDecodeStage(vocab_size=stage.vocab_size, beam_width=5)
        ctc.setup()
        assert ctc.run(got) == ctc.run(want)
    finally:
        stage.teardown()


def test_fuse_only_export_is_labelled_as_such():
    if not _have(FUSE_ONNX):
        pytest.skip("TensorRT or fuse-only export unavailable")
    stage = TrtMskaHeadStage(FUSE_ONNX, ENGINE_DIR, precision="fp32")
    stage.setup()
    try:
        assert stage.config()["head_model"] == "mska-s2g-fuse"
        out = stage.run(_sequence(64))
        assert out.shape[-1] == stage.vocab_size
    finally:
        stage.teardown()

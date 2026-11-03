"""End-to-end check of the pipeline's back half on real MSKA weights.

    postproc -> [assemble] -> head -> ctc

This is the first point where real model code, real weights and the harness's
stage interface meet. It is skipped when the weights are absent, so the suite
still runs on a machine without the 657 MB checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cslr_cost.harness.stages.base import Pipeline
from cslr_cost.harness.stages.ctc import CtcDecodeStage
from cslr_cost.harness.stages.head import MskaHeadStage
from cslr_cost.harness.stages.postproc import (
    HEAD_KEYPOINTS,
    PHOENIX_2014_WH,
    KeypointPostprocStage,
    assemble_sequence,
)

MSKA_ROOT = Path("/workspace/MSKA")
CONFIG = MSKA_ROOT / "configs/phoenix-2014_s2g.yaml"
WEIGHTS = MSKA_ROOT / "pretrained_models/best.pth"

pytestmark = pytest.mark.skipif(
    not (WEIGHTS.exists() and CONFIG.exists()),
    reason="MSKA weights/config not available",
)

W, H = PHOENIX_2014_WH
T = 60


@pytest.fixture(scope="module")
def head() -> MskaHeadStage:
    st = MskaHeadStage(MSKA_ROOT, CONFIG, WEIGHTS)
    st.setup()
    yield st
    st.teardown()


@pytest.fixture(scope="module")
def ctc(head) -> CtcDecodeStage:
    st = CtcDecodeStage(vocab_size=head.vocab_size, beam_width=5)
    st.setup()
    return st


def raw_frames(n: int = T) -> list[np.ndarray]:
    """Plausible pixel-space keypoints for one clip."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        kp = np.empty((HEAD_KEYPOINTS, 3), dtype=np.float32)
        kp[:, 0] = rng.uniform(0, W, HEAD_KEYPOINTS)
        kp[:, 1] = rng.uniform(0, H, HEAD_KEYPOINTS)
        kp[:, 2] = rng.uniform(0.3, 1.0, HEAD_KEYPOINTS)
        out.append(kp)
    return out


def test_back_half_runs_end_to_end(head, ctc):
    post = KeypointPostprocStage()
    units = [post.run(f) for f in raw_frames()]
    seq = assemble_sequence(units)

    assert seq.shape == (1, 3, T, HEAD_KEYPOINTS)

    logits = head.run(seq)
    assert logits.shape == (1, T // 4, head.vocab_size)
    assert torch.isfinite(logits).all()

    glosses = ctc.run(logits)
    assert len(glosses) == 1
    assert all(0 <= g < head.vocab_size for g in glosses[0])


def test_stages_connect_as_a_pipeline(head, ctc):
    """The head consumes the assembly boundary, so postproc and head are not
    directly adjacent — the Pipeline's connectivity check must reflect that."""
    post = KeypointPostprocStage()
    assert post.output_key == "keypoints"
    assert head.input_key == "sequence"
    assert head.output_key == ctc.input_key == "gloss_logits"

    pipe = Pipeline([head, ctc])
    assert [s.name for s in pipe] == ["head", "ctc"]
    assert all(s.granularity == "clip" for s in pipe)


def test_decoded_length_never_exceeds_the_time_axis(head, ctc):
    """CTC cannot emit more symbols than it has timesteps. A longer output
    means the blank index or the collapse is wrong."""
    units = [KeypointPostprocStage().run(f) for f in raw_frames()]
    logits = head.run(assemble_sequence(units))
    glosses = ctc.run(logits)
    assert len(glosses[0]) <= logits.shape[1]


def test_beam_width_is_a_measurable_axis(head):
    """The beam sweep covers {1, 5, 10, 20}; each must at least run and be
    recorded in the row's config."""
    units = [KeypointPostprocStage().run(f) for f in raw_frames()]
    logits = head.run(assemble_sequence(units))

    for beam in (1, 5, 10, 20):
        st = CtcDecodeStage(vocab_size=head.vocab_size, beam_width=beam)
        st.setup()
        out = st.run(logits)
        assert st.config()["beam_width"] == beam
        assert isinstance(out[0], list)

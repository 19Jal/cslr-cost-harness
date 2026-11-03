"""Parity between our `ctcdecode` decoder and MSKA's TensorFlow one.

The 21.24 WER anchor was produced with `tf.nn.ctc_beam_search_decoder`. We
replaced it with `ctcdecode` (TensorFlow on Jetson/Android for one beam search
is not reasonable, and the beam sweep is ours to run). If the replacement
decodes differently, every later WER change is unattributable between "the
decoder changed" and whatever was actually being measured.

`tests/fixtures/ctc_reference.json` holds TF's output on fixed logits, captured
while a TF install still existed, so this check survives without one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cslr_cost.harness.stages.ctc import CtcDecodeStage

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REF_JSON = FIXTURES / "ctc_reference.json"
REF_NPZ = FIXTURES / "ctc_reference_logits.npz"

pytestmark = pytest.mark.skipif(
    not (REF_JSON.exists() and REF_NPZ.exists()),
    reason="TF reference fixture missing",
)


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REF_JSON.read_text())


@pytest.fixture(scope="module")
def logits() -> dict:
    with np.load(REF_NPZ) as z:
        return {k: z[k] for k in z.files}


def _cases(ref: dict):
    for case, per_beam in ref["expected"].items():
        for beam, expected in per_beam.items():
            yield case, int(beam), expected


def test_fixture_describes_the_contract(reference):
    """The fixture must record blank placement and vocabulary, or a future
    reader cannot tell whether a mismatch is a real regression."""
    assert reference["blank_index"] == 0
    assert reference["vocab"] > 0
    assert reference["beams"]


@pytest.mark.parametrize(
    "case,beam",
    [(c, b) for c, b, _ in _cases(json.loads(REF_JSON.read_text()))]
    if REF_JSON.exists() else [],
)
def test_matches_tensorflow(case, beam, reference, logits):
    expected = reference["expected"][case][str(beam)]

    stage = CtcDecodeStage(vocab_size=reference["vocab"], beam_width=beam)
    stage.setup()
    got = stage.run(torch.from_numpy(logits[case]))

    assert got == expected, (
        f"{case} at beam {beam} decodes differently from the TF reference.\n"
        f"  tf  : {expected[0][:20]}{' ...' if len(expected[0]) > 20 else ''}\n"
        f"  ours: {got[0][:20]}{' ...' if len(got[0]) > 20 else ''}\n"
        "A decoder that disagrees with the one the 21.24 anchor was measured "
        "with makes every later WER change unattributable."
    )


def test_second_collapse_is_reproduced():
    """MSKA applies groupby AFTER CTC's own collapse, merging repeats that CTC
    deliberately preserved. That is arguably a bug, but it is what the anchor
    was measured with."""
    assert CtcDecodeStage._collapse([17, 42, 42, 301]) == [17, 42, 301]
    assert CtcDecodeStage._collapse([1, 1, 2, 2, 1]) == [1, 2, 1]


def test_collapse_can_be_disabled_for_investigation(reference, logits):
    """Keeping the un-collapsed sequence available matters for the deletion
    analysis: merged repeats look like deletions in the decomposition."""
    x = torch.from_numpy(logits["structured_T40"])
    raw = CtcDecodeStage(vocab_size=reference["vocab"], beam_width=5,
                         collapse_repeats=False)
    raw.setup()
    assert raw.run(x)[0] == [17, 42, 42, 301]

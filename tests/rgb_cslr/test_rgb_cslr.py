"""The RGB baselines: what would be wrong without anyone noticing.

    replaces, or the engine measures a different network.
  * Padding features at assembly must equal padding frames (VAC pads
    frames; the harness pads per-frame features), or WER through the harness
    pipeline drifts from the reproduced one.
  * The vocabulary is rebuilt, not loaded from the repositories' pickles; it must
    be the 1295-gloss one their checkpoints index.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cslr_cost.harness.stages import rgb_trt
from cslr_cost.rgb_cslr import data, models


@pytest.mark.parametrize("t", [1, 7, 8, 9, 150])
def test_feature_padding_equals_frame_padding(t):
    frames = np.arange(t, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    padded = data.pad(frames)
    assert len(padded) == math.ceil(t / 4) * 4 + 12            # their video_length
    got = rgb_trt.assemble_features([torch.from_numpy(f) for f in frames])
    assert got.shape[0] == 1
    assert torch.equal(got[0], torch.from_numpy(padded))


def test_conv1d_output_length_is_their_update_lgt():
    # K5 P2 K5 P2: (((n - 4) // 2) - 4) // 2
    for n in (24, 140, 228, 320):
        assert models.TemporalConv.out_len(n) == (((n - 4) // 2) - 4) // 2


@pytest.mark.skipif(not data.PHOENIX.exists(), reason="PHOENIX-2014 not on this machine")
def test_vocabulary_is_the_checkpoints_1295_glosses():
    g = data.gloss_dict()
    assert len(g) + 1 == models.NUM_CLASSES
    assert sorted(g.values()) == list(range(1, 1296))
    assert "__ON__" in g and "__OFF__" in g          # the raw annotation, markers kept


def test_vac_builds_with_the_released_key_layout():
    sd = models.RgbCslr("vac").state_dict()
    assert "classifier.weight" in sd and sd["classifier.weight"].shape == (1024, 1296)


def test_vacs_paper_checkpoint_loads_as_plain_linear_classifiers(tmp_path):
    """VAC's paper checkpoint has `nn.Linear` classifiers (weight (1296, 1024) + bias),
    SMKD's weight-normalised ones (1024, 1296): `load` must pick the right one from the
    keys, or a strict load refuses (and a transposed load would be silently wrong)."""
    ref = models.RgbCslr("vac", linear_classifier=True)
    sd = {k.replace("conv2d.", "conv2d.module."): v for k, v in ref.state_dict().items()}
    assert sd["classifier.weight"].shape == (1296, 1024) and "classifier.bias" in sd
    assert sd["conv1d.fc.weight"].shape == (1296, 1024) and "conv1d.fc.bias" in sd
    path = tmp_path / "vac.pt"
    torch.save({"model_state_dict": sd}, path)
    m = models.load("vac", str(path))
    assert isinstance(m.classifier, torch.nn.Linear)
    x = torch.randn(20, 512)
    torch.testing.assert_close(m.head(x), ref.eval().head(x))


def test_odd_batches_are_split_before_the_engine():
    """FP32 TensorRT refuses odd batches from 3 up (as HRNet's engine did)."""
    seen = []

    class Fake:
        def __call__(self, input):
            seen.append(input.shape[0])
            return {"features": torch.zeros(input.shape[0], 512)}

    st = rgb_trt.TrtRgbFrontendStage("x.onnx", "e", "vac-smkd", device="cpu")
    st.module = Fake()
    crops = [np.zeros((224, 224, 3), np.uint8)] * 7
    out = st.run_batch(crops)
    assert len(out) == 7 and all(b == 1 or b % 2 == 0 for b in seen)

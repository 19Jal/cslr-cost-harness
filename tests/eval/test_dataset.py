"""Frame selection and the keypoint file format.

Frame selection is not a detail: `select_frames` decides which frames the head
is given, and getting it wrong changes the logit length and therefore the
decode. So it is checked against MSKA's own `get_selected_index` on real
lengths rather than against a description of it.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cslr_cost.eval.dataset import (
    CLIP_LEN,
    ClipSample,
    iter_clips,
    load_keypoint_file,
    select_frames,
    selected_keypoints,
)

MSKA_ROOT = Path(__file__).resolve().parents[2] / "MSKA"
RELEASED = MSKA_ROOT / "data/Phoenix-2014/phoenix-2014.dev"


# ---------------------------------------------------------------------------
# select_frames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vlen", [4, 8, 99, 100, 215, 299, 300, 399, 400, 401, 512, 1000])
def test_select_frames_matches_mska(vlen):
    """The authority is MSKA's implementation, imported and run side by side."""
    if not (MSKA_ROOT / "datasets.py").exists():
        pytest.skip("MSKA submodule not checked out")
    sys.path.insert(0, str(MSKA_ROOT))
    try:
        from datasets import S2T_Dataset
    except ImportError as exc:                     # pragma: no cover - env-specific
        pytest.skip(f"MSKA datasets not importable: {exc}")

    # `get_selected_index` reads only tmin/tmax/clip_len, so a bare instance is
    # enough — constructing the real dataset would load a 120 MB pickle per case.
    stand_in = S2T_Dataset.__new__(S2T_Dataset)
    stand_in.tmin, stand_in.tmax, stand_in.clip_len = 1, 1, CLIP_LEN

    ref_index, ref_len = stand_in.get_selected_index(vlen)
    got_index, got_len = select_frames(vlen)

    assert got_len == ref_len
    np.testing.assert_array_equal(got_index, ref_index)


def test_short_clips_keep_every_frame_up_to_a_multiple_of_four():
    index, valid = select_frames(99)
    assert valid == 96                       # 99 -> 96, the backbone strides time by 4
    np.testing.assert_array_equal(index, np.arange(96))


def test_long_clips_are_centre_cropped_not_resampled():
    """A resample would change the signing rate; MSKA takes the middle 400."""
    index, valid = select_frames(500)
    assert valid == CLIP_LEN
    assert index[0] == 50 and index[-1] == 449
    np.testing.assert_array_equal(np.diff(index), np.ones(CLIP_LEN - 1))


def test_selection_never_exceeds_the_heads_positional_buffer():
    """MSKA's positional encoding is a fixed buffer of 400 — see stages/head.py."""
    for vlen in (401, 1000, 5000):
        _index, valid = select_frames(vlen)
        assert valid <= CLIP_LEN


# ---------------------------------------------------------------------------
# the file format
# ---------------------------------------------------------------------------

def write_pickle(path: Path, obj) -> Path:
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return path


def test_a_non_dict_file_is_refused(tmp_path):
    p = write_pickle(tmp_path / "bad", ["not", "a", "dict"])
    with pytest.raises(ValueError, match="not a MSKA keypoint file"):
        load_keypoint_file(p)


def test_missing_fields_name_themselves_and_the_format(tmp_path):
    p = write_pickle(tmp_path / "bad", {"k": {"name": "k", "keypoint": None}})
    with pytest.raises(ValueError, match="gloss"):
        load_keypoint_file(p)


def test_num_frames_disagreeing_with_the_tensor_is_refused():
    """MSKA selects on num_frames. A file where the two disagree would have the
    head scored on a different span than the record claims."""
    sample = ClipSample(
        key="k", name="k", gloss="A B", num_frames=10,
        keypoint=np.zeros((8, 133, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="must agree"):
        selected_keypoints(sample)


def test_clips_are_yielded_in_file_order(tmp_path):
    """MSKA evaluates in dict order; the comparison is per-clip, so order only
    has to be stable — but a reordering here would silently misalign a
    hypothesis with its reference if either side were ever zipped."""
    raw = {
        f"c{i}": {
            "name": f"c{i}", "gloss": "A", "num_frames": 4,
            "keypoint": torch.zeros(4, 133, 3),
        }
        for i in range(5)
    }
    p = write_pickle(tmp_path / "kp", raw)
    assert [c.key for c in iter_clips(p)] == [f"c{i}" for i in range(5)]


def test_limit_stops_early(tmp_path):
    raw = {
        f"c{i}": {
            "name": f"c{i}", "gloss": "A", "num_frames": 4,
            "keypoint": torch.zeros(4, 133, 3),
        }
        for i in range(5)
    }
    p = write_pickle(tmp_path / "kp", raw)
    assert len(list(iter_clips(p, limit=2))) == 2


# ---------------------------------------------------------------------------
# against the real released file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RELEASED.exists(), reason="released keypoints not present")
def test_the_released_file_has_the_documented_shape():
    """Pins the interface a keypoint extraction has to write."""
    sample = next(iter_clips(RELEASED, limit=1))
    assert sample.keypoint.ndim == 3
    assert sample.n_keypoints == 133
    assert sample.keypoint.shape[2] == 3
    assert sample.keypoint.dtype == np.float32
    assert sample.keypoint.shape[0] == sample.num_frames

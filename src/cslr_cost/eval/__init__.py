"""Accuracy evaluation: keypoints in, WER with del/sub/ins out.

The harness measures cost. This package measures accuracy, and it exists
because a front-end change needs *both* ends ("pose accuracy degradation **and**
downstream WER") and nothing in the repository could produce a WER before it.

The two are deliberately separate packages. A results row from `harness` is a
cost measurement bound by `harness/schema.py`'s contract; a row from here is an
accuracy measurement on native-resolution input. Mixing them would put WER
fields on ladder rows, which is the one thing `ladder/wer_guard.py` exists to
prevent.
"""

from cslr_cost.eval.dataset import (
    ClipSample,
    load_keypoint_file,
    select_frames,
)
from cslr_cost.eval.wer import HYPOTHESIS_STREAMS, score_split

__all__ = [
    "ClipSample",
    "HYPOTHESIS_STREAMS",
    "load_keypoint_file",
    "select_frames",
    "score_split",
]

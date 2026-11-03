"""dataset.py — MSKA's keypoint files, read the way MSKA reads them.

--------------------------------------------------------------------------
THE FILE FORMAT IS THE INTERFACE
--------------------------------------------------------------------------
`data/Phoenix-2014/phoenix-2014.{train,dev,test}` is a **pickled dict**:

    {key: {"name": str, "gloss": str, "num_frames": int, "keypoint": Tensor}}

where `keypoint` is float32 **(T, 133, 3)** — (x, y, confidence) in **pixel
coordinates at 210x260**, un-normalised. `datasets.py:102-106` normalises at
load time, so a file that arrives pre-normalised is silently normalised twice
and the head sees coordinates in about [-3, -1].

This is also the interface our own extraction writes, which is why
it is pinned here rather than assumed at the call site.

--------------------------------------------------------------------------
FRAME SELECTION IS NOT OPTIONAL
--------------------------------------------------------------------------
`datasets.py:187-204` selects frames before the head ever sees them, and at
evaluation (`tmin == tmax == 1`) it does two things that change the answer:

  * clips longer than `clip_len` (400) are **centre-cropped**, not resampled;
  * the length is then truncated to a **multiple of 4**, because the backbone
    downsamples time by 4.

Skipping either produces a different logit length and a CTC decode that
disagrees with the 21.24 anchor for reasons unrelated to anything measured.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

#: MSKA's positional encoding is a fixed buffer of this size; see stages/head.py.
CLIP_LEN = 400

#: The backbone's temporal downsample factor.
TIME_STRIDE = 4


@dataclass(frozen=True)
class ClipSample:
    """One clip, as MSKA's dataset yields it."""

    key: str
    name: str
    gloss: str
    num_frames: int
    keypoint: np.ndarray        # (T, V, 3), pixel coordinates

    @property
    def n_keypoints(self) -> int:
        return int(self.keypoint.shape[1])


def select_frames(num_frames: int, clip_len: int = CLIP_LEN) -> tuple[np.ndarray, int]:
    """MSKA's `get_selected_index`, evaluation branch, reproduced exactly.

    Returns (frame indices, valid length). See the module docstring for why
    this is load-bearing rather than a detail.
    """
    if num_frames <= clip_len:
        frame_index = np.arange(num_frames)
        valid_len = num_frames
    else:
        sequence = np.arange(num_frames)
        an = (num_frames - clip_len) // 2
        en = num_frames - clip_len - an
        frame_index = sequence[an:-en]
        valid_len = clip_len

    if (valid_len % TIME_STRIDE) != 0:
        valid_len -= valid_len % TIME_STRIDE
        frame_index = frame_index[:valid_len]

    assert len(frame_index) == valid_len, (frame_index, valid_len)
    return frame_index, valid_len


def load_keypoint_file(path: Path) -> dict:
    """Load a MSKA keypoint pickle. Raises with the format on a bad file."""
    with Path(path).open("rb") as fh:
        raw = pickle.load(fh)

    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{path} is not a MSKA keypoint file: expected a non-empty dict of "
            "clips, got " + type(raw).__name__
        )

    probe = next(iter(raw.values()))
    missing = {"name", "gloss", "num_frames", "keypoint"} - set(probe)
    if missing:
        raise ValueError(
            f"{path}: clip records are missing {sorted(missing)}. The format is "
            '{key: {"name", "gloss", "num_frames", "keypoint"}} — see '
            "cslr_cost/eval/dataset.py."
        )
    return raw


def iter_clips(path: Path, limit: int | None = None) -> Iterator[ClipSample]:
    """Yield clips in file order, which is the order MSKA evaluates them in."""
    raw = load_keypoint_file(path)
    for i, (key, rec) in enumerate(raw.items()):
        if limit is not None and i >= limit:
            return
        kp = rec["keypoint"]
        if torch.is_tensor(kp):
            kp = kp.detach().cpu().numpy()
        yield ClipSample(
            key=str(key),
            name=str(rec["name"]),
            gloss=str(rec["gloss"]),
            num_frames=int(rec["num_frames"]),
            keypoint=np.asarray(kp, dtype=np.float32),
        )


def selected_keypoints(sample: ClipSample, clip_len: int = CLIP_LEN) -> np.ndarray:
    """The clip's keypoints after MSKA's evaluation-time frame selection.

    MSKA selects on the record's `num_frames`, not on the tensor's length. The
    two agree in every released file; if a written file ever disagreed, the
    selection would index a different span than the head was given credit for,
    so the disagreement is raised rather than papered over with a `min`.
    """
    stored = int(sample.keypoint.shape[0])
    if stored != sample.num_frames:
        raise ValueError(
            f"{sample.key}: num_frames={sample.num_frames} but the keypoint "
            f"tensor has {stored} frames. MSKA selects frames on num_frames, so "
            "these must agree."
        )
    index, _valid = select_frames(sample.num_frames, clip_len=clip_len)
    return sample.keypoint[index]

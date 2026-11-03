"""PHOENIX-2014 for the RGB baselines, as VAC_CSLR / CorrNet prepare it.

- frames: the 210x260 PNGs, resized to 256x256 with cv2 Lanczos-4 (their
  `dataset_preprocess.py --process-image`, which writes PNG, so in-memory is
  identical), BGR -> RGB, centre crop 224, x / 127.5 - 1;
- padding (their `collate_fn`, batch 1): 6 copies of the first frame before,
  6 + (ceil(T / 4) * 4 - T) copies of the last after; length ceil(T/4)*4 + 12;
- vocabulary: every gloss of train + dev + test's raw annotation (train row 2390
  dropped, as they do), sorted, index from 1; 0 is blank. 1295 glosses.

Keys follow the hypothesis files: "<split>/<folder without *.png>".
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np

#: under the shared corpus tree (`CSLR_DATA`, default ./data;
#: see `harness.cli._shared_data`)
PHOENIX = Path(os.environ.get("CSLR_DATA") or Path("data")) / "phoenix2014/phoenix2014-release/phoenix-2014-multisigner"
LEFT_PAD = 6


def corpus(split: str, root: Path = PHOENIX) -> list[dict]:
    lines = (root / "annotations/manual" / f"{split}.corpus.csv").read_text().splitlines()[1:]
    if split == "train":
        del lines[2390]
    out = []
    for ln in lines:
        fileid, folder, signer, label = ln.split("|")
        out.append({"id": fileid, "folder": folder, "signer": signer, "label": label,
                    "key": f"{split}/{folder.replace('*.png', '')}"})
    return out


def gloss_dict(root: Path = PHOENIX) -> dict[str, int]:
    glosses = sorted({g for sp in ("dev", "test", "train") for c in corpus(sp, root) for g in c["label"].split()})
    return {g: i + 1 for i, g in enumerate(glosses)}


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """One frame, BGR uint8 of any size -> float32 (3, 224, 224) in [-1, 1]."""
    img = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[16:240, 16:240]
    return (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


def load_clip(clip: dict, split: str, root: Path = PHOENIX) -> np.ndarray:
    d = root / "features/fullFrame-210x260px" / split / clip["folder"].replace("*.png", "")
    paths = sorted(d.glob("*.png"))
    if not paths:
        raise FileNotFoundError(d)
    return np.stack([preprocess(cv2.imread(str(p))) for p in paths])


def pad(frames: np.ndarray) -> np.ndarray:
    t = len(frames)
    right = int(math.ceil(t / 4.0)) * 4 - t + LEFT_PAD
    return np.concatenate([np.repeat(frames[:1], LEFT_PAD, 0), frames, np.repeat(frames[-1:], right, 0)])

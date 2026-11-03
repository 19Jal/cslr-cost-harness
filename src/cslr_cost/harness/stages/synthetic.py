"""synthetic.py — stage stand-ins with known cost.

The harness is validated against these *before* real models attach, and that
ordering is deliberate. With stubs that burn a known number of milliseconds, a
broken timing layer or a leaky isolation fixture is obvious: the numbers do not
match what you asked for. Do the same bring-up with real models and a 20%
discrepancy is ambiguous between "the harness is wrong" and "that is genuine
data-movement overhead" — and the reconciliation gate (stage sums within 5%
of end-to-end) cannot distinguish them either.

Two properties are load-bearing, and both were got wrong on the first attempt:

**Cost must actually be known.** Every artifact is preallocated in `__init__`
and returned as a view, so a call costs `cost_ms` plus negligible overhead. An
`emit` that allocates a fresh array per call adds unbudgeted milliseconds and
the stage no longer costs what it says — which defeats the entire purpose of
having stubs.

**Cost is burned by busy-spinning, not sleeping.** `time.sleep` yields the CPU,
so a sleeping stage consumes no cycles and draws no power: it would satisfy a
timing check while making every energy measurement a lie, which is precisely
the class of error this harness exists to catch.

The default costs sketch a plausible shape — pose dominant among the per-frame
stages, head a minority of the clip total. They are
placeholders with no empirical content and must never reach a results table.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

from cslr_cost.harness.stages.base import Granularity, Stage


def spin_ms(ms: float) -> None:
    """Burn `ms` milliseconds of CPU. Busy-wait, not sleep — see module docstring."""
    if ms <= 0:
        return
    deadline = time.perf_counter_ns() + int(ms * 1_000_000)
    while time.perf_counter_ns() < deadline:
        pass


class SyntheticStage(Stage):
    """A stage that costs a known amount and returns a preallocated artifact.

    `cost_ms` is per *unit*: per frame for stages 1-5, per clip for head and
    ctc.
    """

    def __init__(
        self,
        name: str,
        input_key: str,
        output_key: str,
        cost_ms: float,
        *,
        granularity: Granularity = "frame",
        uses_cuda: bool = False,
        emit: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.name = name
        self.input_key = input_key
        self.output_key = output_key
        self.granularity = granularity
        self.uses_cuda = uses_cuda
        self.cost_ms = cost_ms
        self._emit = emit
        self._n_calls = 0

    def config(self) -> dict[str, Any]:
        return {"notes": f"synthetic, nominal {self.cost_ms} ms/{self.granularity}"}

    @property
    def n_calls(self) -> int:
        """Call count — lets a test assert isolation actually replayed."""
        return self._n_calls

    def run(self, x: Any) -> Any:
        self._n_calls += 1
        spin_ms(self.cost_ms)
        return self._emit(x) if self._emit is not None else x


#: Nominal per-unit costs. Placeholders, not measurements.
DEFAULT_COST_MS = {
    "decode": 1.5,
    "detect": 3.0,
    "crop": 0.4,
    "pose": 8.0,
    "postproc": 0.3,
    "head": 25.0,   # per clip
    "ctc": 6.0,     # per clip
}


def synthetic_pipeline(
    height: int = 260,
    width: int = 210,
    crop_size: int = 256,
    n_keypoints: int = 133,
    head_keypoints: int = 79,
    vocab: int = 1296,
    costs: Optional[dict[str, float]] = None,
) -> list[Stage]:
    """The seven stages, wired end to end, with realistic per-unit artifacts.

    Every buffer is allocated once here and returned by view, so each stage
    costs what it declares. Note the `boxes` boundary carries `(frame, box)`
    rather than the box alone: cropping needs both, and pretending otherwise
    would make isolated crop cheaper than the real thing by hiding the frame it
    has to index.
    """
    c = {**DEFAULT_COST_MS, **(costs or {})}
    rng = np.random.default_rng(0)

    frame_buf = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    box_buf = np.array([0, 0, width, height], dtype=np.float32)
    crop_buf = rng.random((crop_size, crop_size // 4 * 3, 3), dtype=np.float32)
    kp_raw_buf = rng.random((n_keypoints, 3), dtype=np.float32)
    kp_buf = kp_raw_buf[:head_keypoints]

    def decoded(_video: Any) -> np.ndarray:
        return frame_buf

    def detected(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return frame, box_buf

    def cropped(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return crop_buf

    def posed(_crop: np.ndarray) -> np.ndarray:
        return kp_raw_buf

    def postprocessed(_kp_raw: np.ndarray) -> np.ndarray:
        return kp_buf

    def headed(sequence: Any) -> np.ndarray:
        # The head downsamples temporally; logits are per output step, not frame.
        return rng.random((max(1, len(sequence) // 4), vocab), dtype=np.float32)

    def decoded_glosses(logits: np.ndarray) -> list[int]:
        return logits.argmax(axis=-1).tolist()

    return [
        SyntheticStage("decode",   "video",         "frames",        c["decode"],   emit=decoded),
        SyntheticStage("detect",   "frames",        "boxes",         c["detect"],   emit=detected),
        SyntheticStage("crop",     "boxes",         "crops",         c["crop"],     emit=cropped),
        SyntheticStage("pose",     "crops",         "keypoints_raw", c["pose"],     emit=posed),
        SyntheticStage("postproc", "keypoints_raw", "keypoints",     c["postproc"], emit=postprocessed),
        SyntheticStage("head",     "sequence",      "gloss_logits",  c["head"],
                       granularity="clip", emit=headed),
        SyntheticStage("ctc",      "gloss_logits",  "glosses",       c["ctc"],
                       granularity="clip", emit=decoded_glosses),
    ]

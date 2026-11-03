"""base.py — the stage interface.

Every pipeline stage sits behind one interface so it can be run **in isolation**
and **in composition**, which is what the protocol requires:

    decode -> detect -> crop -> pose -> postproc -> head -> ctc

Note stage 1 is *video decode*, not camera capture. Hardware decode
(NVDEC / V4L2 / MediaCodec) versus a silent CPU fallback can differ by 2x and
appears in no model-level profile, so the decode path is a measured axis rather
than a footnote.

--------------------------------------------------------------------------
THE SHAPE OF THE PIPELINE
--------------------------------------------------------------------------
The seven stages are not a flat chain. Stages 1-5 are **strictly per-frame**;
`head` and `ctc` consume the **assembled sequence** and run once per clip:

    for each frame:  decode -> detect -> crop -> pose -> postproc
    assemble the per-frame keypoints into a sequence
    once per clip:   head -> ctc

That asymmetry is not a detail — it decides how energy normalises. A per-clip stage's joules/frame is joules-per-clip divided by the
frame count; treating a whole-clip call as if it were one frame inflates the
head's apparent share by the frame count.

So `Stage.run` processes **one unit**: one frame for frame-granular stages, one
clip for clip-granular ones. `Pipeline.run` owns the frame loop and the
assembly boundary.

`uses_cuda` decides how a stage is timed. CUDA is asynchronous, so a GPU stage
must be timed with CUDA events and an explicit synchronise; timing it with a
wall clock measures kernel *launch*, not execution, and reports implausibly
fast numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Sequence

Granularity = Literal["frame", "clip"]

#: Boundary artifacts, in pipeline order. A stage's `input_key`/`output_key`
#: name what it consumes and produces; the fixture store keys on these, which
#: is what lets a stage be replayed alone without paying upstream cost.
BOUNDARIES = (
    "video",          # source clip handle
    "frames",         # one decoded RGB frame
    "boxes",          # (frame, box) — cropping needs both
    "crops",          # one cropped, resized signer region
    "keypoints_raw",  # extractor output for one frame, native layout
    "keypoints",      # mapped to the head's layout + normalised, one frame
    "sequence",       # the assembled per-clip keypoint sequence
    "gloss_logits",   # head output over the gloss vocabulary
    "glosses",        # decoded gloss sequence
)

#: The boundary where per-frame processing ends and per-clip begins.
ASSEMBLY_BOUNDARY = "sequence"


class Stage(ABC):
    """One pipeline stage, processing one unit per `run` call.

    `setup`, `warmup` and `teardown` sit deliberately outside the timed region:
    model loading, allocator growth and first-call CUDA module loading are
    one-off costs that would otherwise land in the first measurement.
    """

    name: str = ""
    input_key: str = ""
    output_key: str = ""

    #: "frame" for stages 1-5, "clip" for head and ctc.
    granularity: Granularity = "frame"

    #: True if `run` launches CUDA work, and so needs event-based timing.
    uses_cuda: bool = False

    def config(self) -> dict[str, Any]:
        """Flat config merged into the results row (model, precision, path...)."""
        return {}

    # -- lifecycle (never timed) -------------------------------------------

    def setup(self) -> None:
        """Load models, allocate buffers. Once, outside measurement."""

    def warmup(self, sample: Any, iterations: int = 10) -> None:
        """Reach steady state before measuring."""
        for _ in range(iterations):
            self.run(sample)

    def teardown(self) -> None:
        """Release models and buffers."""

    # -- the timed unit of work --------------------------------------------

    @abstractmethod
    def run(self, x: Any) -> Any:
        """Transform one unit of `input_key` into one unit of `output_key`.

        Must not mutate `x`: isolation replays captured fixtures many times, and
        an in-place stage would measure a different input on each pass.
        """

    #: True only if `run_batch` is overridden with a genuinely batched
    #: implementation. Left False, the default `run_batch` loops -- which is
    #: what actually happens on that stage in the batch regime, and reporting
    #: it as a batch speed-up would be fiction.
    supports_batching: bool = False

    def run_batch(self, units: Sequence[Any]) -> list[Any]:
        """Transform `len(units)` units at once (P7's batch regime).

        The default is a sequential loop, deliberately. Not every stage can
        batch: `decode` is inherently serial (each frame depends on the codec's
        state), and `crop`'s `warpAffine` is per-image. For those the batch
        regime genuinely costs the same as the streaming regime, and that
        equality is a *finding* -- it says the batch throughput advantage comes
        from the model stages alone, not from the pipeline.

        Overriding this is what makes a stage's batch number differ from its
        streaming number, so `supports_batching` must be set alongside it.
        """
        return [self.run(u) for u in units]

    def __repr__(self) -> str:
        where = "cuda" if self.uses_cuda else "cpu"
        return f"<{type(self).__name__} {self.name} {self.granularity}/{where}>"


def _default_assemble(units: list[Any]) -> Any:
    """Per-frame outputs -> the clip-level object the head consumes.

    Kept pluggable and dependency-free here; real pipelines pass an assembler
    that stacks into the tensor layout their head expects.
    """
    return units


class Pipeline:
    """The seven stages, composed with a frame loop and an assembly boundary.

    Construction validates that consecutive stages connect (`output_key` of one
    matching `input_key` of the next), because a disconnected pipeline still
    runs — each stage simply receives whatever the previous returned — and still
    produces plausible-looking numbers.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        *,
        assemble: Callable[[list[Any]], Any] = _default_assemble,
        stride: int = 1,
        reconstruct: Optional[str] = None,
    ) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")

        self.frame_stages = [s for s in stages if s.granularity == "frame"]
        self.clip_stages = [s for s in stages if s.granularity == "clip"]

        if self.clip_stages and self.frame_stages:
            order = [s.granularity for s in stages]
            if "frame" in order[order.index("clip"):]:
                raise ValueError(
                    "per-frame stages must all precede per-clip stages: the "
                    "sequence cannot be assembled before its frames exist"
                )

        for group in (self.frame_stages, self.clip_stages):
            for prev, nxt in zip(group, group[1:]):
                if prev.output_key != nxt.input_key:
                    raise ValueError(
                        f"stage '{prev.name}' emits '{prev.output_key}' but "
                        f"'{nxt.name}' expects '{nxt.input_key}' — the pipeline "
                        "is not connected, and a disconnected pipeline still "
                        "runs and still produces plausible-looking numbers."
                    )

        self.stages = list(stages)
        self.assemble = assemble

        # Every frame stage runs on every frame. (The runners keep a stride
        # attribute, fixed at 1, so per-call bookkeeping stays explicit.)
        if stride != 1 or reconstruct is not None:
            raise ValueError("this harness measures every frame: stride must be 1")
        self.stride = stride
        self.reconstruct = reconstruct
        self.every_frame_stages = [s for s in self.frame_stages if s.input_key == BOUNDARIES[0]]
        self.strided_stages = [s for s in self.frame_stages if s.input_key != BOUNDARIES[0]]
        if self.frame_stages[: len(self.every_frame_stages)] != self.every_frame_stages:
            raise ValueError("the stages that read the source must come first")

    # -- properties ---------------------------------------------------------

    @property
    def input_key(self) -> str:
        return self.stages[0].input_key

    @property
    def output_key(self) -> str:
        return self.stages[-1].output_key

    def config(self) -> dict[str, Any]:
        """Row fields contributed by the pipeline itself (none at stride 1)."""
        return {}

    def stride_of(self, stage: Stage) -> int:
        """How many frames apart this stage runs: the stride for a strided
        frame stage, 1 for decode and for the clip stages."""
        return self.stride if stage in self.strided_stages else 1

    def is_kept(self, index: int) -> bool:
        """Whether frame `index` goes through the strided stages."""
        return index % self.stride == 0

    def stages_for_frame(self, index: int) -> list[Stage]:
        """The frame stages that run on frame `index`."""
        return self.frame_stages if self.is_kept(index) else self.every_frame_stages

    def rebuild(self, units: list[Any], n_frames: int) -> list[Any]:
        """Kept-frame units -> full-rate units. The identity at stride 1."""
        return units

    def setup(self) -> None:
        for s in self.stages:
            s.setup()

    def teardown(self) -> None:
        for s in reversed(self.stages):
            s.teardown()

    # -- execution ----------------------------------------------------------

    def run_frame(self, x: Any, record: Optional[dict[str, list]] = None,
                  index: int = 0) -> Any:
        """Push frame `index` through the frame stages that run on it: stages
        1-5, or only decode on a frame the stride skips. The caller keeps the
        result only when `is_kept(index)`."""
        for stage in self.stages_for_frame(index):
            x = stage.run(x)
            if record is not None:
                record.setdefault(stage.output_key, []).append(x)
        return x

    def run(
        self,
        source: Any,
        n_frames: int,
        record: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Run the composed pipeline over a clip of `n_frames`.

        `record` collects boundary artifacts for the fixture store: per-frame
        boundaries accumulate a list (one entry per frame), clip-level
        boundaries store the single object.
        """
        frame_record: Optional[dict[str, list]] = {} if record is not None else None
        if record is not None:
            record.setdefault(self.input_key, []).append(source)

        units: list[Any] = []
        for i in range(n_frames):
            x = self.run_frame(source, record=frame_record, index=i)
            if self.is_kept(i):
                units.append(x)

        if frame_record is not None and record is not None:
            record.update(frame_record)

        x = self.assemble(self.rebuild(units, n_frames))
        if record is not None:
            record[ASSEMBLY_BOUNDARY] = x

        for stage in self.clip_stages:
            x = stage.run(x)
            if record is not None:
                record[stage.output_key] = x
        return x

    def __iter__(self) -> Iterator[Stage]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        return " -> ".join(s.name for s in self.stages)

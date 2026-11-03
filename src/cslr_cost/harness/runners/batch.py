"""batch.py — the unpaced, batch 16-64 regime.

The other half of P7's "report both regimes". Deliberately a **separate loop
structure** from `streaming.py` rather than the same loop with a batch
parameter: P7 warns that building one and retrofitting the other means
rewriting the harness, and the two really are different shapes. Streaming paces
and asks "does it keep up"; batch saturates and asks "how much can this device
get through".

Their difference is the point. The streaming-minus-batch energy gap prices what
real-time costs -- a number nobody has published for CSLR -- and it only exists
because both are measured the same way on the same clip.

--------------------------------------------------------------------------
BATCHING IS PER STAGE, AND MOST STAGES CANNOT DO IT
--------------------------------------------------------------------------
`Stage.run_batch` defaults to a sequential loop, and that default is not a
placeholder. `decode` is inherently serial -- each frame depends on the codec's
state -- and `crop`'s `warpAffine` is per-image. Only `pose` (and a real
detector) issue one kernel for the whole batch.

So a pipeline's batch speed-up comes from one stage, and this runner records
`batched_stages` to say which. Crediting the whole pipeline for one stage's GPU
occupancy is a common way to overstate batch throughput, and it is avoidable
simply by writing down which stages actually batched.

--------------------------------------------------------------------------
LATENCY IN THIS REGIME MEANS SOMETHING ELSE
--------------------------------------------------------------------------
Per-frame latency is not a user-facing quantity under batching: nothing is
available until the whole batch finishes. `latency_ms_median` is therefore
**per batch**, and `ms_per_frame` is the throughput-equivalent -- batch time
divided by batch size. They are not interchangeable, and reporting the second
as though it were a latency is the exact substitution P7 rule 1 forbids.

The schema enforces the other half: a `batch` row carrying `paced_fps` is
rejected, because a paced batch run is neither regime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from cslr_cost.harness.stages.base import Pipeline
from cslr_cost.harness.timing import (
    Durations,
    MemoryProbe,
    MemorySnapshot,
    make_timer,
)

#: P7's batch regime: "batch 16-64, unpaced".
BATCH_SIZES = (16, 32, 64)
DEFAULT_BATCH_SIZE = 32


@dataclass
class BatchResult:
    """One clip pushed through in batches. Unpaced by construction."""

    n_frames: int
    batch_size: int
    batch_durations: Durations
    stage_durations: dict[str, Durations]
    clip_stage_ms: float
    wall_s: float
    memory: MemorySnapshot
    batched_stages: tuple[str, ...]
    output: Any = None
    notes: str = ""
    #: Frames per batch are batch_size * stride: the strided stages
    #: get `batch_size` units, decode reads `stride` times as many frames.
    stride: int = 1

    regime: str = field(default="batch", init=False)
    #: P7 forbids pacing here, and the schema rejects a batch row that carries
    #: it. Kept as an explicit None so the intent is visible at the call site.
    paced_fps: Optional[float] = field(default=None, init=False)

    @property
    def ms_per_frame(self) -> Optional[float]:
        """Throughput-equivalent, *not* a latency. See the module docstring."""
        median = self.batch_durations.median
        if median is None:
            return None
        return median / (self.batch_size * self.stride)

    @property
    def throughput_fps(self) -> Optional[float]:
        per_frame = self.ms_per_frame
        if not per_frame:
            return None
        return 1000.0 / per_frame

    def summary(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "batch_size": self.batch_size,
            "paced_fps": self.paced_fps,
            "n_frames": self.n_frames,
            "latency_ms_median": self.batch_durations.median,   # per batch
            "latency_ms_p95": self.batch_durations.p95,
            "ms_per_frame": None if self.ms_per_frame is None else round(self.ms_per_frame, 4),
            "peak_ram_mb": self.memory.peak_ram_mb,
            "peak_vram_mb": self.memory.peak_vram_mb,
            "peak_device_mem_mb": self.memory.peak_device_mem_mb,
            "device_mem_source": self.memory.device_mem_source,
        }

    def table(self) -> str:
        lines = [
            f"batch  size {self.batch_size}  unpaced  ({self.n_frames} frames)",
            "",
            f"{'stage':<10}{'ms/batch':>10}{'ms/frame':>10}  batched",
            f"{'-'*10}{'-'*10:>10}{'-'*10:>10}  {'-'*7}",
        ]
        for name, dur in self.stage_durations.items():
            med = dur.median
            if med is None:
                continue
            lines.append(
                f"{name:<10}{med:>10.2f}{med / self.batch_size:>10.3f}"
                f"  {'yes' if name in self.batched_stages else 'no'}"
            )
        per_frame = self.ms_per_frame
        lines += [
            "",
            f"  per batch (frame stages)  {self.batch_durations.median:.2f} ms"
            if self.batch_durations.median is not None else "",
            f"  head+ctc (once per clip)  {self.clip_stage_ms:.2f} ms",
            f"  throughput-equivalent     {per_frame:.3f} ms/frame "
            f"({self.throughput_fps:.1f} fps)" if per_frame else "",
            "",
            "  ms/frame is throughput, not latency: nothing is available until",
            "  the batch finishes.",
        ]
        return "\n".join(line for line in lines if line != "")


def run_batched(
    pipeline: Pipeline,
    source: Any,
    n_frames: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    poll_memory: bool = True,
) -> BatchResult:
    """Push one clip through `pipeline` in batches of `batch_size`, unpaced.

    Call after `pipeline.setup()` and a warm-up pass.

    The frame stages receive `batch_size` units per call; the per-clip stages
    still run once, on the assembled sequence, exactly as in the streaming
    regime. That asymmetry is the pipeline's real shape (stages 1-5 per frame,
    head and ctc per clip) and batching does not change it.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    uses_cuda = any(s.uses_cuda for s in pipeline)
    timer = make_timer(uses_cuda)
    stage_timer = make_timer(uses_cuda)
    probe = MemoryProbe(track_vram=uses_cuda).start() if poll_memory else None

    batch_durations = Durations()
    stage_durations: dict[str, Durations] = {s.name: Durations() for s in pipeline.frame_stages}
    batched_stages = tuple(s.name for s in pipeline.frame_stages if s.supports_batching)

    units: list[Any] = []
    start = time.perf_counter()

    # At stride S a chunk spans batch_size * S frames, so the strided stages
    # (pose among them) still receive full batches of `batch_size` -- what a
    # deployment would do -- while decode reads every frame of the chunk.
    stride = pipeline.stride
    remaining = n_frames
    first = 0
    while remaining > 0:
        size = min(batch_size * stride, remaining)
        remaining -= size

        # The first stage's input is the source handle, one entry per frame in
        # the batch. For a stateful decoder that means "give me `size` frames",
        # which is exactly what the sequential default does.
        chunk: Sequence[Any] = [source] * size

        with timer:
            for stage in pipeline.every_frame_stages:
                with stage_timer:
                    chunk = stage.run_batch(chunk)
                stage_durations[stage.name].add(stage_timer.elapsed_ms)
            if stride > 1:
                chunk = [u for j, u in enumerate(chunk) if pipeline.is_kept(first + j)]
            for stage in pipeline.strided_stages:
                with stage_timer:
                    chunk = stage.run_batch(chunk)
                stage_durations[stage.name].add(stage_timer.elapsed_ms)
        batch_durations.add(timer.elapsed_ms)

        units.extend(chunk)
        first += size
        if probe is not None:
            probe.poll()

    rebuild_ms = 0.0
    if stride > 1:
        with timer:
            units = pipeline.rebuild(units, n_frames)
        rebuild_ms = timer.elapsed_ms
    x = pipeline.assemble(units)
    with timer:
        for stage in pipeline.clip_stages:
            x = stage.run(x)
    clip_stage_ms = timer.elapsed_ms + rebuild_ms

    wall_s = time.perf_counter() - start
    if probe is not None:
        probe.poll()
    memory = probe.stop() if probe is not None else MemorySnapshot()

    return BatchResult(
        n_frames=n_frames,
        batch_size=batch_size,
        batch_durations=batch_durations,
        stage_durations=stage_durations,
        clip_stage_ms=clip_stage_ms,
        wall_s=wall_s,
        memory=memory,
        batched_stages=batched_stages,
        output=x,
        stride=stride,
    )


# ---------------------------------------------------------------------------
# The batch regime inside an energy window
# ---------------------------------------------------------------------------

@dataclass
class BatchSweep:
    """P7's batch regime inside P1's energy protocol.

        idle (settle, then measure)  ->  clips at batch N, unpaced, >= 60 s
                                     ->  idle again, and compare

    `run_batched` alone opens no energy window. For the paced streaming loop
    that is the right call -- a window over it is mostly idle draw -- but
    unpaced batch work is the opposite case: the device is busy for the whole
    window, so window joules over clips completed *is* joules per video. That
    is the figure the batch regime reports, and the batch half of the
    streaming-vs-batch energy gap.
    """

    composed: Any                  # StageMeasurement: "end_to_end", clip granularity
    results: list[BatchResult]
    batch_size: int
    n_frames: int
    idle: Any = None               # IdleBaseline
    contention: Any = None         # ContentionReport

    @property
    def last(self) -> BatchResult:
        return self.results[-1]

    @property
    def batch_durations(self) -> Durations:
        """Per-batch latency pooled over every clip in the window, not one clip's."""
        pooled = Durations()
        for r in self.results:
            for ms in r.batch_durations.samples_ms:
                pooled.add(ms)
        return pooled

    @property
    def energy_is_defensible(self) -> bool:
        """`IsolationSweep`'s rule: a readable, stable idle bracket and no tenant."""
        if self.idle is None or self.idle.unavailable_reason is not None:
            return False
        if not self.idle.stable:
            return False
        return self.contention is None or self.contention.clean

    @property
    def contention_verdict(self) -> Optional[str]:
        """The row's `contention` field (see `energy.contention_verdict`)."""
        from cslr_cost.harness.energy import contention_verdict

        return contention_verdict(self.contention, self.idle)


def measure_batch_sweep(
    pipeline: Pipeline,
    source: Any,
    n_frames: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_seconds: Optional[float] = None,
    warmup_clips: int = 1,
    meter: Any = None,
    idle_seconds: float = 10.0,
    idle_settle_seconds: Optional[float] = None,
    check_contention: bool = True,
) -> BatchSweep:
    """Clips at batch N for P1's minimum window, with energy and an idle bracket.

    One iteration is one clip, and the recorded value is its **wall** time.
    Unpaced, wall time is the work -- the opposite of the paced loop, where a
    clip's wall time is set by the pacing and service time has to be summed.
    """
    # Imported here: isolation and paced import this module's neighbours, and
    # batch.py is the one runner that has so far needed nothing from them.
    from cslr_cost.harness.energy.baseline import (
        close_idle_baseline,
        measure_idle_baseline,
    )
    from cslr_cost.harness.energy.contention import ContentionProbe
    from cslr_cost.harness.runners.isolation import (
        DEFAULT_MIN_SECONDS,
        StageMeasurement,
        _start_energy,
        _stop_energy,
    )
    from cslr_cost.harness.runners.paced import _rewind

    window = DEFAULT_MIN_SECONDS if min_seconds is None else min_seconds
    uses_cuda = any(s.uses_cuda for s in pipeline)

    # Through both idle brackets, as in the other sweeps: an idle window is
    # where a second tenant is easiest to see.
    probe = ContentionProbe().start() if check_contention else None
    baseline = None
    if meter is not None and idle_seconds > 0:
        if probe is not None:
            probe.mark("idle-open")
        baseline = measure_idle_baseline(
            meter, seconds=idle_seconds, settle_seconds=idle_settle_seconds
        )
        if probe is not None:
            probe.mark(None)

    for _ in range(warmup_clips):
        _rewind(source)
        run_batched(pipeline, source, n_frames, batch_size=batch_size, poll_memory=False)

    memory_probe = MemoryProbe(track_vram=uses_cuda).start()
    durations = Durations()
    results: list[BatchResult] = []

    if probe is not None:
        probe.mark(f"batch-{batch_size}")
    meter = _start_energy(meter)
    start = time.perf_counter()
    while time.perf_counter() - start < window:
        _rewind(source)
        result = run_batched(pipeline, source, n_frames, batch_size=batch_size, poll_memory=False)
        durations.add(result.wall_s * 1000.0)
        results.append(result)
        memory_probe.poll()
        if probe is not None:
            probe.poll()
    energy, energy_reason = _stop_energy(meter, uses_cuda)
    window_s = time.perf_counter() - start
    if probe is not None:
        probe.mark(None)
    memory = memory_probe.stop()

    if baseline is not None and meter is not None:
        if probe is not None:
            probe.mark("idle-close")
        baseline = close_idle_baseline(baseline, meter)
    report = probe.stop() if probe is not None else None

    notes = [f"batch {batch_size} unpaced: per-clip wall time, which here is the work"]
    if len(results) < 3:
        notes.append(f"only {len(results)} clips in the window — too few for a median")

    composed = StageMeasurement(
        name="end_to_end",
        granularity="clip",
        measurement_mode="composed",
        durations=durations,
        memory=memory,
        window_s=window_s,
        warmup_s=0.0,
        iterations=len(results),
        uses_cuda=uses_cuda,
        notes=" | ".join(notes),
        energy=energy,
        energy_unavailable_reason=energy_reason,
    )
    return BatchSweep(
        composed=composed,
        results=results,
        batch_size=batch_size,
        n_frames=n_frames,
        idle=baseline,
        contention=report,
    )

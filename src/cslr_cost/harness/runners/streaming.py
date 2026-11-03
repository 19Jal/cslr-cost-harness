"""streaming.py — the paced, batch-1 regime.

P7's three rules are non-negotiable, and each is enforced here rather than
left to discipline:

1. **Batch size 1.** No "effective batch", no padding several clips together.
   Quoting a batch-32 throughput figure as streaming fps is a common reporting error.
2. **Pace the frames** -- one every 40 ms at 25 fps, matching PHOENIX-2014 and
   Isharah capture. An unpaced loop measures throughput; a paced loop measures
   whether the pipeline *keeps up*, which is the deployment question. Frames
   that exceed their budget are logged.
3. **No lookahead beyond what you declare.** MSKA's head is bidirectional: it
   consumes the whole clip, so its first-gloss latency *is* the whole clip.
   That gets reported, not quietly replaced with "time to decode after the clip
   ended".

--------------------------------------------------------------------------
THE PACING GAP MUST BE GENUINELY IDLE
--------------------------------------------------------------------------
The obvious way to pace precisely is to sleep until just before the deadline
and then busy-spin, because `time.sleep` has millisecond-scale jitter. That
would be correct for a timing harness and **catastrophic for this one.**

The streaming-minus-batch energy gap is the headline this regime exists to
produce -- what real-time costs, a number nobody has published for CSLR. A spin
loop draws full CPU power during exactly the interval that is supposed to be
idle, which would erase the gap and report streaming as costing the same as
batch. So the whole gap is slept, and the resulting pacing jitter is
**measured and reported** instead of being spun away. `spin_ms` in
`stages/synthetic.py` exists for the opposite reason and the contrast is the
point: stubs must burn power, pacing must not.

--------------------------------------------------------------------------
NO FRAMES ARE DROPPED, AND THAT IS A CHOICE
--------------------------------------------------------------------------
A real-time system behind schedule usually drops frames to catch up. This
runner does not, for a reason that has nothing to do with fidelity: dropping
changes the *input* to the head, and therefore WER. A dropped-frame streaming
run could not be compared against the batch run on accuracy, and the two
regimes would stop being the same experiment.

So every frame is processed, lateness accumulates, and two different ledgers
are kept because they answer different questions:

- `overrun_ms_total` -- summed per-frame excess over budget, measured inside
  the timed region. P7's ledger: how much work exceeded its slot.
- `lag_ms_final` -- how far behind wall-clock schedule the pipeline ended up.

Neither bounds the other in general, and the difference is informative:

    lag  ~=  overrun_total  -  slack recovered by fast frames
                            +  the loop's own untimed per-frame overhead

So a pipeline with occasional spikes recovers between them and ends with
`lag << overrun_total`, while one that never keeps up ends with `lag` slightly
*above* it -- the excess being harness overhead outside the timed window, which
is a useful check on the runner itself. Reading either number alone would call
both cases the same.

Frame dropping is a legitimate *deployment* strategy, but it is a different
pipeline, and its WER cost has to be measured with it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from cslr_cost.harness.stages.base import Pipeline
from cslr_cost.harness.timing import (
    Durations,
    MemoryProbe,
    MemorySnapshot,
    make_timer,
)

#: PHOENIX-2014 and Isharah capture rate. 25 fps => a 40 ms per-frame budget.
DEFAULT_PACED_FPS = 25.0

#: How the head consumes time, declared rather than inferred (P7 rule 3).
LOOKAHEAD_FULL_CLIP = "full-clip"
LOOKAHEAD_CAUSAL = "causal"

_FULL_CLIP_NOTE = (
    "bidirectional head: consumes the whole clip before emitting, so first-gloss "
    "latency is bounded below by the clip duration itself. This is an "
    "architectural bound, not a compute cost -- a faster device does not reduce "
    "it. P7 rule 3."
)


@dataclass
class StreamingResult:
    """One paced clip. Batch size is 1 by construction, never a parameter."""

    n_frames: int
    paced_fps: float
    frame_durations: Durations
    clip_stage_ms: float
    pacing_jitter: Durations
    overrun_frames: int
    overrun_ms_total: float
    lag_ms_final: float
    idle_s: float
    wall_s: float
    memory: MemorySnapshot
    first_gloss_latency_ms: float
    head_lookahead: str
    #: Per-stage service time measured *inside* the paced loop, when requested.
    #: This is what sums to the composed total; paced isolation does not (see
    #: `runners/paced.py`), because the wake-up after each idle slot belongs to
    #: the slot rather than to any stage.
    stage_durations: dict[str, Durations] = field(default_factory=dict)
    output: Any = None
    notes: str = ""

    batch_size: int = field(default=1, init=False)
    regime: str = field(default="streaming_simulated", init=False)

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / self.paced_fps

    @property
    def overrun_rate(self) -> float:
        return self.overrun_frames / self.n_frames if self.n_frames else 0.0

    @property
    def kept_up(self) -> bool:
        """Did every frame fit its slot? The deployment question P7 poses."""
        return self.overrun_frames == 0

    @property
    def duty_cycle(self) -> Optional[float]:
        """Fraction of wall time spent computing rather than waiting.

        The lever behind the streaming-vs-batch energy gap: a pipeline at 20%
        duty cycle spends 80% of a streaming run idle, and whether that idle
        time is cheap is a property of the device, not of the model.
        """
        if self.wall_s <= 0:
            return None
        return max(0.0, (self.wall_s - self.idle_s) / self.wall_s)

    @property
    def first_gloss_latency_note(self) -> Optional[str]:
        return _FULL_CLIP_NOTE if self.head_lookahead == LOOKAHEAD_FULL_CLIP else None

    def summary(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "batch_size": self.batch_size,
            "paced_fps": self.paced_fps,
            "n_frames": self.n_frames,
            "latency_ms_median": self.frame_durations.median,
            "latency_ms_p95": self.frame_durations.p95,
            "overrun_frames": self.overrun_frames,
            "overrun_ms_total": round(self.overrun_ms_total, 3),
            "lag_ms_final": round(self.lag_ms_final, 3),
            "first_gloss_latency_ms": round(self.first_gloss_latency_ms, 2),
            "duty_cycle": None if self.duty_cycle is None else round(self.duty_cycle, 4),
            "peak_ram_mb": self.memory.peak_ram_mb,
            "peak_vram_mb": self.memory.peak_vram_mb,
            "peak_device_mem_mb": self.memory.peak_device_mem_mb,
            "device_mem_source": self.memory.device_mem_source,
        }

    def table(self) -> str:
        budget = self.frame_budget_ms
        med = self.frame_durations.median
        p95 = self.frame_durations.p95
        jit = self.pacing_jitter.median
        lines = [
            f"streaming-simulated  batch 1  paced {self.paced_fps:g} fps "
            f"({budget:.1f} ms/frame budget)",
            "",
            f"  frames              {self.n_frames}",
            f"  per-frame median    {med:.2f} ms" if med is not None else "  per-frame median    n/a",
            f"  per-frame p95       {p95:.2f} ms" if p95 is not None else "  per-frame p95       n/a",
            f"  head+ctc (once)     {self.clip_stage_ms:.2f} ms",
            f"  pacing jitter med   {jit:.3f} ms" if jit is not None else "  pacing jitter med   n/a",
            "",
            f"  overrun frames      {self.overrun_frames}/{self.n_frames} "
            f"({self.overrun_rate:.1%})",
            f"  overrun total       {self.overrun_ms_total:.1f} ms",
            f"  final lag           {self.lag_ms_final:.1f} ms",
            f"  duty cycle          {self.duty_cycle:.1%}" if self.duty_cycle is not None else "",
            "",
            f"  first-gloss latency {self.first_gloss_latency_ms:.1f} ms "
            f"({self.head_lookahead})",
        ]
        if self.first_gloss_latency_note:
            lines.append(f"    ^ {self.first_gloss_latency_note.split(':')[0]}")
        lines.append("")
        lines.append("  KEEPS UP" if self.kept_up else "  DOES NOT KEEP UP at this pacing")
        return "\n".join(line for line in lines if line != "")


def run_streaming(
    pipeline: Pipeline,
    source: Any,
    n_frames: int,
    *,
    paced_fps: float = DEFAULT_PACED_FPS,
    head_lookahead: str = LOOKAHEAD_FULL_CLIP,
    poll_memory: bool = True,
    per_stage_timing: bool = False,
) -> StreamingResult:
    """Feed one clip through the pipeline at `paced_fps`, batch 1.

    Call after `pipeline.setup()` and a warm-up pass: model loading and
    allocator growth inside a paced loop would be logged as overruns that no
    deployed system would ever see.
    """
    if paced_fps <= 0:
        raise ValueError(f"paced_fps must be positive, got {paced_fps}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    budget_s = 1.0 / paced_fps
    budget_ms = budget_s * 1000.0

    uses_cuda = any(s.uses_cuda for s in pipeline)
    timer = make_timer(uses_cuda)
    probe = MemoryProbe(track_vram=uses_cuda).start() if poll_memory else None

    # Timing each stage in place costs nothing measurable: a three-arm
    # experiment put the per-frame CUDA sync at -0.07 ms/frame, because `pose`
    # already pulls keypoints to the CPU every frame and so drains the queue
    # regardless. It is still opt-in, so the headline streaming number is
    # produced by the same code path whether or not anyone is looking at the
    # breakdown.
    stage_timers = (
        {st.name: make_timer(st.uses_cuda) for st in pipeline}
        if per_stage_timing else {}
    )
    stage_durations: dict[str, Durations] = (
        {st.name: Durations() for st in pipeline} if per_stage_timing else {}
    )

    frame_durations = Durations()
    pacing_jitter = Durations()
    overrun_frames = 0
    overrun_ms_total = 0.0
    idle_s = 0.0

    units: list[Any] = []
    start = time.perf_counter()

    for i in range(n_frames):
        deadline = start + i * budget_s
        now = time.perf_counter()
        if now < deadline:
            # Sleep the whole gap. Spinning here would draw full power during
            # the interval that is supposed to be idle and erase the very
            # energy gap this regime exists to measure.
            time.sleep(deadline - now)
            idle_s += time.perf_counter() - now
        pacing_jitter.add((time.perf_counter() - deadline) * 1000.0)

        with timer:
            if per_stage_timing:
                x = source
                for st in pipeline.stages_for_frame(i):
                    with stage_timers[st.name]:
                        x = st.run(x)
                    stage_durations[st.name].add(stage_timers[st.name].elapsed_ms)
            else:
                x = pipeline.run_frame(source, index=i)
            if pipeline.is_kept(i):
                units.append(x)
        elapsed_ms = timer.elapsed_ms
        frame_durations.add(elapsed_ms)

        if elapsed_ms > budget_ms:
            overrun_frames += 1
            overrun_ms_total += elapsed_ms - budget_ms

        if probe is not None:
            probe.poll()

    frames_done = time.perf_counter()
    lag_ms_final = (frames_done - (start + n_frames * budget_s)) * 1000.0

    # --- the assembly boundary and the per-clip stages ---------------------
    # The stride's rebuild runs once the last frame is in, so it is charged with
    # the per-clip work; at stride 1 it is the identity and costs nothing.
    rebuild_ms = 0.0
    if pipeline.stride > 1:
        with timer:
            units = pipeline.rebuild(units, n_frames)
        rebuild_ms = timer.elapsed_ms
    x = pipeline.assemble(units)
    with timer:
        for stage in pipeline.clip_stages:
            if per_stage_timing:
                with stage_timers[stage.name]:
                    x = stage.run(x)
                stage_durations[stage.name].add(stage_timers[stage.name].elapsed_ms)
            else:
                x = stage.run(x)
    clip_stage_ms = timer.elapsed_ms + rebuild_ms

    wall_s = time.perf_counter() - start
    if probe is not None:
        probe.poll()
    memory = probe.stop() if probe is not None else MemorySnapshot()

    # P7 rule 3: for a bidirectional head the first gloss cannot exist until
    # the whole clip has been consumed, so this is the honest number -- the
    # full paced clip duration plus the head and decoder.
    if head_lookahead == LOOKAHEAD_FULL_CLIP:
        first_gloss_latency_ms = (frames_done - start) * 1000.0 + clip_stage_ms
    else:
        first_gloss_latency_ms = (frame_durations.median or 0.0) + clip_stage_ms

    return StreamingResult(
        n_frames=n_frames,
        paced_fps=paced_fps,
        frame_durations=frame_durations,
        clip_stage_ms=clip_stage_ms,
        pacing_jitter=pacing_jitter,
        overrun_frames=overrun_frames,
        overrun_ms_total=overrun_ms_total,
        lag_ms_final=max(0.0, lag_ms_final),
        idle_s=idle_s,
        wall_s=wall_s,
        memory=memory,
        first_gloss_latency_ms=first_gloss_latency_ms,
        head_lookahead=head_lookahead,
        stage_durations=stage_durations,
        output=x,
    )

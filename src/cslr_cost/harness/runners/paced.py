"""paced.py — isolation under streaming pacing.

P1 wants each stage measured alone. P7 wants the streaming regime paced to
25 fps. The reconciliation gate wants per-stage times reconciling within 5%
**in each regime** — so the two have to be combined, and the combination is not
either one with a flag flipped.

--------------------------------------------------------------------------
WHY THIS IS NOT `isolation.py` WITH A SLEEP IN IT
--------------------------------------------------------------------------
Measured on this box, the same stage costs **1.8x more per frame when paced**
than when run back to back — an additive ~4.4 ms/frame, near-constant across
every resolution rung, and not an artifact of timing (a three-arm experiment
put the per-frame CUDA sync at -0.07 ms). Idling ~30 ms in each 40 ms slot lets
clocks wind down, and every frame pays the ramp.

So the unpaced numbers are simply not the streaming numbers, and copying them
across would have understated a streaming deployment by nearly a factor of two
— a common reporting error.

--------------------------------------------------------------------------
WHAT RECONCILES IS SERVICE TIME, NOT WALL TIME
--------------------------------------------------------------------------
At 20-38% duty cycle most of a paced run's wall clock is `sleep`. Reconciling
wall time would converge on the duty cycle and say nothing about the pipeline.

The quantity that means something is **service time**: the milliseconds spent
inside `run()` for one unit. `Durations` here holds exactly that — the sleep is
outside the timer — and it is directly comparable with
`StreamingResult.frame_durations`, which measures the same thing on the
composed side. That composed half already existed; this is the isolated half.

--------------------------------------------------------------------------
WARM-UP MUST ALSO BE PACED
--------------------------------------------------------------------------
The subtle one. Warm up back-to-back and the clocks are boosted at the moment
the measured window opens, so the first frames are artificially fast and the
median is dragged toward the unpaced figure — hiding the very effect the run
exists to measure. Warm-up here runs at the same cadence as the measurement.

--------------------------------------------------------------------------
EACH STAGE IS PACED AT ITS OWN RATE
--------------------------------------------------------------------------
Frame stages get a slot every 1/fps. Clip stages do not: `head` and `ctc` run
once per clip, as a burst after the last frame, so their cadence is one call
every `n_frames / fps` seconds — 5.12 s for a 128-frame clip at 25 fps. Pacing
them at 25 Hz would measure something no deployment does.

The cost is sample count: a 60 s window gives ~1500 samples for a frame stage
and ~11 for a clip stage. That is thin, and it is reported (`iterations`, plus
a note) rather than hidden — the alternative, measuring clip stages unpaced,
would silently drop the ramp they pay after 5 s of near-idle, which is the
largest such gap in the pipeline.

--------------------------------------------------------------------------
AND PACED ISOLATION DOES NOT SUM EITHER. MEASURED, NOT ASSUMED.
--------------------------------------------------------------------------
The first version of this module assumed paced isolation would reconcile the
way unpaced isolation does. It does not, and the gap is not small — a −52%
residual on the first real run.

The reason is visible in the per-stage ratios. Paced/unpaced cost comes out at
44× for `detect` (0.001 → 0.039 ms), 15× for `postproc`, 7× for `crop`, 4× for
`pose`, and 1.1× for `ctc`. A *slowdown* would hit every stage alike; a ratio
that tracks 1/cost is a **fixed additive charge per call**. Each stage,
measured alone at 25 Hz, pays its own wake-up after its own idle gap:

    Σ paced isolated − Σ unpaced  =  16.86 ms/frame     (five ramps)
    composed paced   − Σ unpaced  =   8.73 ms/frame     (one ramp)

**The wake-up belongs to the slot, not to any stage.** P1's additivity
assumption is a property of back-to-back execution, and pacing breaks it. No
amount of care in this runner recovers it, because isolating a stage genuinely
changes the idle it experiences.

So paced isolation answers "what does this stage cost when it alone runs at
25 Hz" — a real question, and the one that shows `pose` paying a 14.4 ms GPU
clock ramp — but it is **not** the quantity that reconciles. That is
`measure_streaming_insitu`, which times each stage *inside* the paced loop,
where exactly one ramp is paid and it lands on whichever stage touches the
device first.

--------------------------------------------------------------------------
IN THIS REGIME, TOTAL ENERGY DOES NOT SUM. OVER-IDLE ENERGY DOES.
--------------------------------------------------------------------------
An isolated paced stage draws idle power for the ~30 ms it is not working, and
that idle draw is inside its joules. Sum seven such stages and you have summed
seven slots' worth of idle for one frame's work, over-counting by roughly the
number of stages.

The **incremental-over-idle** figure subtracts exactly one idle baseline from
each stage, so it does add up, and it is what a per-stage streaming energy
table has to be built from. P2 asks for both numbers; in this regime they play
different roles, and only one of them is additive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from cslr_cost.harness.energy import (
    ContentionProbe,
    ContentionReport,
    EnergyMeter,
    IdleBaseline,
    close_idle_baseline,
    contention_verdict,
    explain_instability,
    measure_idle_baseline,
)
from cslr_cost.harness.fixtures import FixtureStore
from cslr_cost.harness.runners.isolation import (
    DEFAULT_MIN_SECONDS,
    DEFAULT_WARMUP_SECONDS,
    StageMeasurement,
    _start_energy,
    _stop_energy,
)
from cslr_cost.harness.runners.streaming import (
    DEFAULT_PACED_FPS,
    LOOKAHEAD_FULL_CLIP,
    run_streaming,
)
from cslr_cost.harness.stages.base import Pipeline, Stage
from cslr_cost.harness.timing import (
    Durations,
    MemoryProbe,
    MemorySnapshot,
    make_timer,
)

#: Below this many samples a median is not worth much. Clip stages at a 5 s
#: cadence hit it easily, so it is said out loud on the row.
THIN_SAMPLE_COUNT = 30


def stage_period_s(stage: Stage, paced_fps: float, n_frames: int, stride: int = 1) -> float:
    """How often this stage is called in a paced pipeline.

    Frame stages once per frame; clip stages once per clip. Getting this wrong
    is the paced equivalent of the granularity mis-scaling that produced a
    -1537% residual on the unpaced side.
    """
    if paced_fps <= 0:
        raise ValueError(f"paced_fps must be positive, got {paced_fps}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    per_frame = 1.0 / paced_fps
    # A strided frame stage runs on every `stride`-th frame slot.
    return per_frame * stride if stage.granularity == "frame" else per_frame * n_frames


def measure_stage_paced(
    stage: Stage,
    fixtures: FixtureStore,
    *,
    paced_fps: float,
    n_frames: int,
    stride: int = 1,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    max_iterations: Optional[int] = None,
    poll_memory: bool = True,
    meter: Optional[EnergyMeter] = None,
    contention: Optional[ContentionProbe] = None,
) -> StageMeasurement:
    """Run `stage` alone at its paced cadence, recording per-call service time."""
    units = fixtures.units_for(stage)
    if not units:
        raise ValueError(f"no captured units to replay for stage '{stage.name}'")

    period_s = stage_period_s(stage, paced_fps, n_frames, stride)
    budget_ms = period_s * 1000.0
    timer = make_timer(stage.uses_cuda)

    # --- warm-up, at the SAME cadence -------------------------------------
    # Warming up back-to-back would leave the clocks boosted when the window
    # opens, dragging the first samples toward the unpaced figure and hiding
    # the effect this runner exists to measure.
    warm_start = time.perf_counter()
    i = 0
    while time.perf_counter() - warm_start < warmup_seconds:
        deadline = warm_start + i * period_s
        now = time.perf_counter()
        if now < deadline:
            time.sleep(deadline - now)
        stage.run(units[i % len(units)])
        i += 1
    warm_s = time.perf_counter() - warm_start

    # --- measured window ---------------------------------------------------
    durations = Durations()
    probe = MemoryProbe(track_vram=stage.uses_cuda).start() if poll_memory else None
    if contention is not None:
        contention.mark(f"stage:{stage.name}")
    meter = _start_energy(meter)

    overrun = 0
    overrun_ms = 0.0
    idle_s = 0.0

    start = time.perf_counter()
    i = 0
    while True:
        if time.perf_counter() - start >= min_seconds:
            break
        if max_iterations is not None and i >= max_iterations:
            break

        deadline = start + i * period_s
        now = time.perf_counter()
        if now < deadline:
            # Slept, never spun: a spin would draw full power through the
            # interval that is supposed to be idle, and the whole point of the
            # paced regime is what that idle costs.
            time.sleep(deadline - now)
            idle_s += time.perf_counter() - now

        with timer:
            stage.run(units[i % len(units)])
        elapsed_ms = timer.elapsed_ms
        durations.add(elapsed_ms)

        if elapsed_ms > budget_ms:
            overrun += 1
            overrun_ms += elapsed_ms - budget_ms

        if probe is not None and (i & 0x0F) == 0:
            probe.poll()
        if contention is not None:
            contention.poll()
        i += 1

    energy, energy_reason = _stop_energy(meter, stage.uses_cuda)
    if contention is not None:
        contention.mark(None)
    window_s = time.perf_counter() - start
    memory = probe.stop() if probe is not None else MemorySnapshot()

    notes = [f"paced {paced_fps:g} fps, period {budget_ms:.1f} ms/call"]
    if overrun:
        notes.append(f"{overrun} calls over budget, {overrun_ms:.1f} ms total")
    if i < THIN_SAMPLE_COUNT:
        notes.append(
            f"only {i} samples — a clip stage at a {budget_ms / 1000:.2f}s cadence "
            "cannot produce many in a 60s window; the median is thin and the p95 "
            "should not be quoted"
        )
    if window_s < DEFAULT_MIN_SECONDS:
        notes.append(
            f"window {window_s:.1f}s is below P1's 60s floor — valid for timing, "
            "NOT for energy integration"
        )

    return StageMeasurement(
        name=stage.name,
        granularity=stage.granularity,
        measurement_mode="isolated",
        durations=durations,
        memory=memory,
        window_s=window_s,
        warmup_s=warm_s,
        iterations=i,
        uses_cuda=stage.uses_cuda,
        notes=" | ".join(notes),
        energy=energy,
        energy_unavailable_reason=energy_reason,
    )


def measure_all_stages_paced(
    pipeline: Pipeline,
    fixtures: FixtureStore,
    **kwargs: Any,
) -> dict[str, StageMeasurement]:
    """Every stage alone at its own paced cadence, in pipeline order."""
    out = {}
    for s in pipeline:
        stride = pipeline.stride_of(s)
        m = measure_stage_paced(s, fixtures, stride=stride, **kwargs)
        m.stride = stride
        out[s.name] = m
    return out


@dataclass
class PacedSweep:
    """P1's protocol run inside P7's streaming regime."""

    isolated: dict[str, StageMeasurement]
    composed: StageMeasurement
    paced_fps: float
    n_frames: int
    idle: Optional[IdleBaseline] = None
    contention: Optional[ContentionReport] = None

    @property
    def energy_is_defensible(self) -> bool:
        if self.idle is None or self.idle.unavailable_reason is not None:
            return False
        if not self.idle.stable:
            return False
        return self.contention is None or self.contention.clean

    @property
    def contention_verdict(self) -> Optional[str]:
        """The rows' `contention` field (see `energy.contention_verdict`)."""
        return contention_verdict(self.contention, self.idle)

    def notes(self) -> str:
        parts = [explain_instability(self.idle, self.contention)
                 if self.idle is not None else "no energy meter"]
        if self.contention is not None:
            parts.append(self.contention.note())
        parts.append(
            "paced isolation: TOTAL joules do not sum across stages (each pays a "
            "full slot of idle); over-idle joules do"
        )
        return " | ".join(parts)


def measure_streaming_composed(
    pipeline: Pipeline,
    source: Any,
    n_frames: int,
    *,
    paced_fps: float = DEFAULT_PACED_FPS,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    warmup_clips: int = 1,
    head_lookahead: str = LOOKAHEAD_FULL_CLIP,
    poll_memory: bool = True,
    meter: Optional[EnergyMeter] = None,
    contention: Optional[ContentionProbe] = None,
) -> tuple[StageMeasurement, list]:
    """The composed streaming pass, as something `reconcile` can consume.

    One iteration is one paced clip, and the recorded value is that clip's
    **service time** -- the sum of every frame's time inside `run()`, plus the
    per-clip head and decoder burst. Deliberately *not* wall time: a paced clip
    takes `n_frames / fps` seconds by construction, so reconciling wall time
    would compare the isolated sum against the pacing rate and pass or fail on
    the duty cycle.

    Returns the measurement and the per-clip `StreamingResult`s, which carry
    the overrun ledger and first-gloss latency that no reconciliation needs but
    every results row does.
    """
    durations = Durations()
    results: list = []
    probe = MemoryProbe(track_vram=any(s.uses_cuda for s in pipeline)).start() if poll_memory else None

    for _ in range(warmup_clips):
        _rewind(source)
        run_streaming(pipeline, source, n_frames, paced_fps=paced_fps,
                      head_lookahead=head_lookahead, poll_memory=False)

    if contention is not None:
        contention.mark("composed")
    meter = _start_energy(meter)
    start = time.perf_counter()
    i = 0
    while time.perf_counter() - start < min_seconds:
        _rewind(source)
        result = run_streaming(
            pipeline, source, n_frames, paced_fps=paced_fps,
            head_lookahead=head_lookahead, poll_memory=False,
        )
        durations.add(result.frame_durations.total_ms + result.clip_stage_ms)
        results.append(result)
        if probe is not None:
            probe.poll()
        if contention is not None:
            contention.poll()
        i += 1

    energy, energy_reason = _stop_energy(meter, any(s.uses_cuda for s in pipeline))
    if contention is not None:
        contention.mark(None)
    window_s = time.perf_counter() - start
    memory = probe.stop() if probe is not None else MemorySnapshot()

    notes = ["service time per clip, not wall time (a paced clip's wall time is "
             "set by the pacing, not by the work)"]
    if i < 3:
        notes.append(
            f"only {i} clips in the window — a paced clip takes "
            f"{n_frames / paced_fps:.1f}s, so a short run cannot produce a median"
        )

    return (
        StageMeasurement(
            name="end_to_end",
            granularity="clip",
            measurement_mode="composed",
            durations=durations,
            memory=memory,
            window_s=window_s,
            warmup_s=0.0,
            iterations=i,
            uses_cuda=any(s.uses_cuda for s in pipeline),
            notes=" | ".join(notes),
            energy=energy,
            energy_unavailable_reason=energy_reason,
        ),
        results,
    )


def _rewind(source: Any) -> None:
    """Restart the clip if the source knows how.

    `rewind()` seeks; assigning `position = 0` only moves a counter and leaves
    the decoder where it was (see `stages/decode.py`). Guarded so a fixture or
    a stub source without one still works.
    """
    rewind = getattr(source, "rewind", None)
    if callable(rewind):
        rewind()


def measure_paced_sweep(
    pipeline: Pipeline,
    fixtures: FixtureStore,
    source: Any,
    n_frames: int,
    *,
    paced_fps: float = DEFAULT_PACED_FPS,
    meter: Optional[EnergyMeter] = None,
    idle_seconds: float = 10.0,
    idle_settle_seconds: Optional[float] = None,
    check_contention: bool = True,
    warmup_clips: int = 1,
    **kwargs: Any,
) -> tuple[PacedSweep, list]:
    """P1's whole protocol inside P7's streaming regime.

        idle (settle, then measure)  ->  each stage alone, at its paced cadence
                                     ->  the composed paced pipeline
                                     ->  idle again, and compare

    The mirror of `measure_isolation_sweep`, and deliberately a separate
    function rather than a `paced=True` flag on it: the loops differ in warm-up
    cadence, in what is timed, and in what the recorded value means.
    """
    # Before the opening bracket: an idle window is where a tenant is easiest
    # to see, because this process is contributing nothing to compare against.
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

    isolated = measure_all_stages_paced(
        pipeline, fixtures, paced_fps=paced_fps, n_frames=n_frames,
        meter=meter, contention=probe, **kwargs
    )
    composed, results = measure_streaming_composed(
        pipeline, source, n_frames, paced_fps=paced_fps,
        min_seconds=kwargs.get("min_seconds", DEFAULT_MIN_SECONDS),
        warmup_clips=warmup_clips, meter=meter, contention=probe,
    )

    # Through both idle brackets — see the note in isolation.py: an idle window
    # is the most sensitive place to detect a tenant, not a dilution.
    if baseline is not None and meter is not None:
        if probe is not None:
            probe.mark("idle-close")
        baseline = close_idle_baseline(baseline, meter)

    report = probe.stop() if probe is not None else None

    return (
        PacedSweep(
            isolated=isolated,
            composed=composed,
            paced_fps=paced_fps,
            n_frames=n_frames,
            idle=baseline,
            contention=report,
        ),
        results,
    )


def measure_streaming_insitu(
    pipeline: Pipeline,
    source: Any,
    n_frames: int,
    *,
    paced_fps: float = DEFAULT_PACED_FPS,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    warmup_clips: int = 1,
    head_lookahead: str = LOOKAHEAD_FULL_CLIP,
    poll_memory: bool = True,
    meter: Optional[EnergyMeter] = None,
    contention: Optional[ContentionProbe] = None,
) -> tuple[dict[str, StageMeasurement], StageMeasurement, list]:
    """Per-stage service time measured *inside* the paced loop.

    This is the streaming regime's reconcilable decomposition, and the reason
    it works where paced isolation does not: the pipeline pays one wake-up per
    slot, and timing stages in place charges it to whichever stage actually
    triggers it — normally `pose`, the first to touch the GPU after the idle.

    Every stage row comes back with `measurement_mode="composed"`, because that
    is what it is: this stage, as it behaves inside the composed pass. The
    schema's two modes carry the distinction without needing a third.

    **Both sides are medians of per-clip totals**, and that is load-bearing.
    Reconciling a *sum* against scaled *medians* compares different quantities:
    a sum concentrates around the mean, so `median_per_frame x n_frames`
    matches it only when the distribution is symmetric. Paced frames are not
    symmetric — p95/median runs about 2.8 here, because the post-idle frames
    are far slower than typical — and the first version of this function got
    +26.85% residual purely from that mismatch, with no physical cause at all.

    Accumulating each stage's per-clip total keeps both sides on the same
    footing, and P5's median-not-mean rule still holds: the median is taken
    *over clips*, which is the unit being repeated.

    Returns `(per_stage, end_to_end, streaming_results)`.
    """
    # Per-clip TOTALS per stage, not per-call samples — see the docstring.
    per_stage_durations: dict[str, Durations] = {s.name: Durations() for s in pipeline}
    clip_totals = Durations()
    results: list = []
    uses_cuda = any(s.uses_cuda for s in pipeline)
    probe = MemoryProbe(track_vram=uses_cuda).start() if poll_memory else None

    for _ in range(warmup_clips):
        _rewind(source)
        run_streaming(pipeline, source, n_frames, paced_fps=paced_fps,
                      head_lookahead=head_lookahead, poll_memory=False)

    if contention is not None:
        contention.mark("insitu")
    meter = _start_energy(meter)
    start = time.perf_counter()
    i = 0
    while time.perf_counter() - start < min_seconds:
        _rewind(source)
        result = run_streaming(
            pipeline, source, n_frames, paced_fps=paced_fps,
            head_lookahead=head_lookahead, poll_memory=False,
            per_stage_timing=True,
        )
        for name, dur in result.stage_durations.items():
            per_stage_durations[name].add(dur.total_ms)
        clip_totals.add(result.frame_durations.total_ms + result.clip_stage_ms)
        results.append(result)
        if probe is not None:
            probe.poll()
        if contention is not None:
            contention.poll()
        i += 1

    energy, energy_reason = _stop_energy(meter, uses_cuda)
    if contention is not None:
        contention.mark(None)
    window_s = time.perf_counter() - start
    memory = probe.stop() if probe is not None else MemorySnapshot()

    def _measurement(name: str, durations: Durations, granularity: str) -> StageMeasurement:
        return StageMeasurement(
            name=name,
            granularity=granularity,
            measurement_mode="composed",
            durations=durations,
            memory=MemorySnapshot(),
            window_s=window_s,
            warmup_s=0.0,
            iterations=len(durations),
            uses_cuda=uses_cuda,
            notes=f"in situ, paced {paced_fps:g} fps — service time inside the "
                  "composed loop, where the per-slot wake-up is charged to the "
                  "stage that actually triggers it",
            # Timing decomposes in situ; energy does not. The seven stages run
            # interleaved inside one paced window and both rails integrate that
            # whole window, so there is no per-stage joule figure to record
            # here — not a missing measurement, an unmeasurable one. Said out
            # loud because the schema's rule is that a null energy always
            # states its reason, and "no reason recorded" is how this looked
            # for a whole grid.
            energy_unavailable_reason=(
                "per-stage energy cannot be separated in situ: the stages run "
                "interleaved inside one paced window and the rails integrate "
                "the window, not the stage. The streaming regime's per-stage "
                "energy comes from paced isolation, whose over-idle joules do "
                "sum; this row carries service time only."
            ),
        )

    # Granularity "clip" for every stage: the accumulated value is already a
    # whole clip's worth, so `per_clip_ms` must not scale it again. Declaring
    # "frame" here would multiply by n_frames a second time — the granularity
    # mis-scaling that produced a -1537% residual on the unpaced side.
    per_stage = {
        s.name: _measurement(s.name, per_stage_durations[s.name], "clip")
        for s in pipeline
    }
    end_to_end = StageMeasurement(
        name="end_to_end",
        granularity="clip",
        measurement_mode="composed",
        durations=clip_totals,
        memory=memory,
        window_s=window_s,
        warmup_s=0.0,
        iterations=i,
        uses_cuda=uses_cuda,
        notes="service time per clip, not wall time",
        energy=energy,
        energy_unavailable_reason=energy_reason,
    )
    return per_stage, end_to_end, results

"""isolation.py — measuring one stage alone.

P1's protocol:

    1. Isolation runs. Run each stage in a tight loop for >=60 seconds, alone,
       at steady state. Integrate power over the window; divide by iterations.
    2. Composed run. Measure the whole pipeline the same way.
    3. Reconciliation. The isolated sum should approach the composed total.
       Report the residual explicitly.

This module is step 1, plus the composed pass for step 2. `reconcile.py` is
step 3.

Three things this gets right that a naive loop does not:

**Warm-up is outside the window.** The first calls pay lazy CUDA module
loading, allocator growth and cold caches. Including them drags the median and
inflates p95 with a tail that is a one-off, not a property of the stage.

**Input comes from the fixture store, cycling.** A stage cannot run alone
without its input, and re-deriving it would silently measure upstream stages
too (see `fixtures`). Cycling through the clip's real frames also avoids
measuring an artificially cache-hot path from hammering one buffer.

**Duration is time-based, not count-based.** A 60 s floor is what makes the
energy integration valid: power monitors sample at 10-100 ms, so a window must
span many samples for the integral to mean anything.

**The energy window is the measured window, and nothing else.** The meter opens
after warm-up and closes after the loop, so model loading, allocator growth and
first-call CUDA module loading are outside it -- the same boundary the timer
uses. Opening it earlier would charge a stage for work no steady-state system
repeats, and would do so unevenly: the stages with the largest models would
absorb the most, which is exactly the ranking the per-stage table is trying to
establish.

On a CUDA stage the meter is closed only after an explicit synchronise. The
timer already syncs per iteration, but relying on that would make the energy
window silently dependent on the timer's internals, and a GPU still draining a
queue when the counter is read attributes its joules to whatever runs next.

When no rail is readable the reading is `None` and
`energy_unavailable_reason` carries the meter's explanation. The schema then
writes a row with null energy *and* a stated reason -- never a silent zero,
which would read as a stage that costs nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from cslr_cost.harness.energy import (
    EnergyMeter,
    EnergyReading,
    EnergyUnavailable,
    ContentionProbe,
    ContentionReport,
    IdleBaseline,
    close_idle_baseline,
    contention_verdict,
    explain_instability,
    measure_idle_baseline,
)
from cslr_cost.harness.fixtures import FixtureStore
from cslr_cost.harness.stages.base import Pipeline, Stage
from cslr_cost.harness.timing import (
    Durations,
    MemoryProbe,
    MemorySnapshot,
    make_timer,
)

#: P1 wants >=60 s at steady state. Overridable for tests and smoke runs, but
#: a real measurement must not go below this — say so in `notes` if it does.
DEFAULT_MIN_SECONDS = 60.0
DEFAULT_WARMUP_SECONDS = 5.0


@dataclass
class StageMeasurement:
    """What one isolated (or composed) measurement produced."""

    name: str
    granularity: str
    measurement_mode: str          # "isolated" | "composed"
    durations: Durations
    memory: MemorySnapshot
    window_s: float
    warmup_s: float
    iterations: int
    uses_cuda: bool = False
    notes: str = ""

    #: None when no rail was readable — paired with a reason, never a zero.
    energy: Optional[EnergyReading] = None
    energy_unavailable_reason: Optional[str] = None

    #: A frame stage that runs on every `stride`-th frame is called
    #: ceil(n_frames / stride) times per clip, not n_frames. 1 for decode, the
    #: clip stages, and everything at stride 1.
    stride: int = 1

    def calls_per_clip(self, n_frames: int) -> int:
        """How many of this measurement's iterations make one clip."""
        if self.granularity != "frame":
            return 1
        return -(-n_frames // self.stride)

    @property
    def median_ms(self) -> Optional[float]:
        return self.durations.median

    @property
    def p95_ms(self) -> Optional[float]:
        return self.durations.p95

    def per_clip_ms(self, n_frames: int) -> Optional[float]:
        """Cost of this stage for one clip, as a **median of clip-sized totals**.

        The per-frame/per-clip asymmetry lives here. Stages 1-5 run once per
        frame; `head` and `ctc` run once per clip. Getting that backwards is
        what makes an isolated sum disagree with the composed run by an order
        of magnitude.

        The subtler half is *which statistic*. This is compared against the
        composed measurement, whose every sample is already a clip's **total**
        — so `median_per_call x n_frames` is not the right partner for it. A
        total concentrates around the mean, and mean and median agree only for
        a symmetric distribution. In the paced regime they do not agree at all
        (p95/median ~2.8), and reconciling the two produced a clean-looking
        +26.85% residual with no physical cause whatsoever.

        So per-call samples are grouped into consecutive clip-sized blocks,
        each block summed, and the median taken over blocks. That is the same
        quantity the composed side reports, and it keeps P5's median-not-mean
        rule: robust to a single hiccup, because the median is over blocks.
        Using the mean here instead would let one outlier move the gate.

        Falls back to `median x n_frames` when there are not two full blocks —
        a clip stage measured at a 5 s cadence rarely has more.
        """
        if self.median_ms is None:
            return None
        if self.granularity != "frame":
            return self.median_ms

        calls = self.calls_per_clip(n_frames)
        samples = self.durations.samples_ms
        n_blocks = len(samples) // calls
        if n_blocks < 2:
            return self.median_ms * calls

        totals = Durations()
        for b in range(n_blocks):
            totals.add(sum(samples[b * calls:(b + 1) * calls]))
        return totals.median

    @property
    def joules_per_iteration(self) -> Optional[float]:
        """P1's 'integrate over the window; divide by iterations'."""
        if self.energy is None:
            return None
        return self.energy.per_unit(self.iterations)

    def joules_per_frame(self, n_frames: int) -> Optional[float]:
        """Energy normalised per *frame*, whatever the stage's granularity.

        The mirror image of `per_clip_ms`, and wrong in the opposite direction
        if confused: one iteration of a frame stage is one frame, but one
        iteration of `head` or `ctc` is a whole clip. Dividing a clip stage's
        joules by iterations alone would inflate the head's apparent per-frame
        cost by `n_frames` — and the head's share of system joules is precisely
        what a whole-pipeline account has to get right.
        """
        per_iteration = self.joules_per_iteration
        if per_iteration is None:
            return None
        calls = self.calls_per_clip(n_frames)
        return per_iteration if calls == n_frames else per_iteration * calls / n_frames

    def joules_per_frame_incremental(self, n_frames: int) -> Optional[float]:
        """The same, over idle (P2). None until an idle baseline exists."""
        if self.energy is None:
            return None
        inc = self.energy.per_unit_incremental(self.iterations)
        if inc is None:
            return None
        calls = self.calls_per_clip(n_frames)
        return inc if calls == n_frames else inc * calls / n_frames

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "mode": self.measurement_mode,
            "granularity": self.granularity,
            "iterations": self.iterations,
            "window_s": round(self.window_s, 2),
            **self.durations.summary(),
            "peak_ram_mb": self.memory.peak_ram_mb,
            "peak_vram_mb": self.memory.peak_vram_mb,
            "peak_device_mem_mb": self.memory.peak_device_mem_mb,
            "device_mem_source": self.memory.device_mem_source,
            "energy_backend": None if self.energy is None else self.energy.backend,
            "joules_per_iteration": self.joules_per_iteration,
            "energy_unavailable_reason": self.energy_unavailable_reason,
        }


def _start_energy(meter: Optional[EnergyMeter]) -> Optional[EnergyMeter]:
    """Open the energy window, or report that there is none to open."""
    if meter is None:
        return None
    meter.start()
    return meter


def _stop_energy(
    meter: Optional[EnergyMeter], uses_cuda: bool
) -> tuple[Optional[EnergyReading], Optional[str]]:
    """Close the window after the device has actually finished.

    Returns `(reading, reason)`, exactly one of which is populated. A meter
    that cannot read raises rather than returning zero, and that refusal is
    turned into a reason the results row carries.
    """
    if meter is None:
        return None, "no energy meter supplied to the runner"

    if uses_cuda:
        # The timer already syncs per iteration, but depending on that would
        # tie the energy window to the timer's internals. A GPU still draining
        # its queue when the counter is read donates its joules to whatever
        # runs next.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    try:
        return meter.stop(), None
    except EnergyUnavailable as exc:
        return None, str(exc)


def measure_stage_isolated(
    stage: Stage,
    fixtures: FixtureStore,
    *,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    max_iterations: Optional[int] = None,
    poll_memory: bool = True,
    meter: Optional[EnergyMeter] = None,
    contention: Optional[ContentionProbe] = None,
) -> StageMeasurement:
    """Run `stage` alone against its captured input for at least `min_seconds`.

    Returns per-iteration timings and, when a `meter` is supplied, an energy
    reading over exactly the same span — so joules/iteration and ms/iteration
    describe the same work.
    """
    units = fixtures.units_for(stage)
    if not units:
        raise ValueError(f"no captured units to replay for stage '{stage.name}'")

    timer = make_timer(stage.uses_cuda)

    # --- warm-up: outside the measured window, deliberately ---------------
    warm_start = time.perf_counter()
    i = 0
    while time.perf_counter() - warm_start < warmup_seconds:
        stage.run(units[i % len(units)])
        i += 1
    warm_s = time.perf_counter() - warm_start

    # --- measured window ---------------------------------------------------
    durations = Durations()
    # Named window for the contention verdict (blocker 15), opened outside the
    # energy window because marking forces an NVML query.
    if contention is not None:
        contention.mark(f"stage:{stage.name}")
    probe = MemoryProbe(track_vram=stage.uses_cuda).start() if poll_memory else None
    meter = _start_energy(meter)

    start = time.perf_counter()
    i = 0
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds:
            break
        if max_iterations is not None and i >= max_iterations:
            break

        with timer:
            stage.run(units[i % len(units)])
        durations.add(timer.elapsed_ms)

        if probe is not None and (i & 0x3F) == 0:  # every 64th iteration
            probe.poll()
        if contention is not None and (i & 0x3F) == 0:
            contention.poll()                       # rate-limited internally
        i += 1

    energy, energy_reason = _stop_energy(meter, stage.uses_cuda)
    window_s = time.perf_counter() - start
    if contention is not None:
        contention.mark(None)
    memory = probe.stop() if probe is not None else MemorySnapshot()

    notes = ""
    if window_s < DEFAULT_MIN_SECONDS:
        notes = (
            f"window {window_s:.1f}s is below P1's 60s floor — valid for timing, "
            "NOT for energy integration (power monitors sample at 10-100ms)"
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
        notes=notes,
        energy=energy,
        energy_unavailable_reason=energy_reason,
    )


def measure_pipeline_composed(
    pipeline: Pipeline,
    source: Any,
    n_frames: int,
    *,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    max_iterations: Optional[int] = None,
    poll_memory: bool = True,
    meter: Optional[EnergyMeter] = None,
    contention: Optional[ContentionProbe] = None,
) -> StageMeasurement:
    """Measure the whole pipeline end to end — P1 step 2.

    One iteration is one full clip, so `median_ms` here is milliseconds per
    clip and compares directly against the sum of the isolated stages' per-clip
    costs.
    """
    uses_cuda = any(s.uses_cuda for s in pipeline)
    timer = make_timer(uses_cuda)

    warm_start = time.perf_counter()
    while time.perf_counter() - warm_start < warmup_seconds:
        pipeline.run(source, n_frames)
    warm_s = time.perf_counter() - warm_start

    durations = Durations()
    if contention is not None:
        contention.mark("composed")
    probe = MemoryProbe(track_vram=uses_cuda).start() if poll_memory else None
    meter = _start_energy(meter)

    start = time.perf_counter()
    i = 0
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds:
            break
        if max_iterations is not None and i >= max_iterations:
            break

        with timer:
            pipeline.run(source, n_frames)
        durations.add(timer.elapsed_ms)

        if probe is not None:
            probe.poll()
        if contention is not None:
            contention.poll()                       # rate-limited internally
        i += 1

    energy, energy_reason = _stop_energy(meter, uses_cuda)
    window_s = time.perf_counter() - start
    if contention is not None:
        contention.mark(None)
    memory = probe.stop() if probe is not None else MemorySnapshot()

    return StageMeasurement(
        name="end_to_end",
        granularity="clip",
        measurement_mode="composed",
        durations=durations,
        memory=memory,
        window_s=window_s,
        warmup_s=warm_s,
        iterations=i,
        uses_cuda=uses_cuda,
        energy=energy,
        energy_unavailable_reason=energy_reason,
    )



#: Why an in-situ stage row carries no energy. P1: a sampled rail cannot
#: attribute energy inside one composed pass, and neither can a counter read
#: around one stage call. Per-stage energy is on the stage's isolated row.
INSITU_ENERGY_REASON = (
    "timed in situ: per-stage energy cannot be attributed inside a composed "
    "pass (P1); this stage's energy is on its isolated row"
)


def measure_batch_insitu(
    pipeline: Pipeline,
    source: Any,
    n_frames: int,
    *,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    max_iterations: Optional[int] = None,
    contention: Optional[ContentionProbe] = None,
) -> tuple[dict[str, StageMeasurement], StageMeasurement]:
    """Per-stage time *inside* the unpaced batch loop -- the batch regime's
    reconcilable decomposition.

    The loop is `measure_pipeline_composed`'s exactly -- same warm-up, same
    source, no rewinds between clips, so decode sees the same frames and the
    same seeks -- except that every stage is timed where it runs. Each stage's
    per-clip total is accumulated, and both sides of the reconciliation are
    medians over clips, as in the streaming regime's `measure_streaming_insitu`.

    Why the batch regime needs it too. On the Orin, FFmpeg decodes H.264 on six
    worker threads that run ahead of the reader. Isolated, decode is called back
    to back and waits on them; composed, they decode the next frames while
    `pose` is on the GPU, and `read()` returns a frame already waiting. Isolation
    serialises that overlap away: at FP16, where pose is short enough for the
    overlap to dominate, the stages timed alone overshot the composed clip by up
    to 8.9% (rtmpose-s, 720p) -- a negative residual, parts costing more than
    the whole. Timing in place charges each stage what it costs *in* the
    pipeline, overlap included.

    What it cannot give is per-stage energy (P1), so every row it produces says
    so, and the isolated rows keep theirs. Returns `(per_stage, end_to_end)`.
    """
    uses_cuda = any(s.uses_cuda for s in pipeline)
    clip_timer = make_timer(uses_cuda)
    stage_timers = {s.name: make_timer(s.uses_cuda) for s in pipeline}

    warm_start = time.perf_counter()
    while time.perf_counter() - warm_start < warmup_seconds:
        pipeline.run(source, n_frames)
    warm_s = time.perf_counter() - warm_start

    per_stage = {s.name: Durations() for s in pipeline}
    clip_totals = Durations()

    if contention is not None:
        contention.mark("insitu")
    start = time.perf_counter()
    i = 0
    while True:
        if time.perf_counter() - start >= min_seconds:
            break
        if max_iterations is not None and i >= max_iterations:
            break

        acc = dict.fromkeys(per_stage, 0.0)
        with clip_timer:
            units: list[Any] = []
            for f in range(n_frames):
                x = source
                for st in pipeline.stages_for_frame(f):
                    t = stage_timers[st.name]
                    with t:
                        x = st.run(x)
                    acc[st.name] += t.elapsed_ms
                if pipeline.is_kept(f):
                    units.append(x)
            # At stride > 1 the rebuild is inside the clip's time and no stage's:
            # it lands in the residual, where unattributed pipeline work belongs.
            x = pipeline.assemble(pipeline.rebuild(units, n_frames))
            for st in pipeline.clip_stages:
                t = stage_timers[st.name]
                with t:
                    x = st.run(x)
                acc[st.name] += t.elapsed_ms
        for name, ms in acc.items():
            per_stage[name].add(ms)
        clip_totals.add(clip_timer.elapsed_ms)

        if contention is not None:
            contention.poll()                       # rate-limited internally
        i += 1

    window_s = time.perf_counter() - start

    def _measurement(name: str, durations: Durations, notes: str) -> StageMeasurement:
        # Granularity "clip": each sample is already a whole clip's total, so
        # `per_clip_ms` must not scale it by n_frames again.
        return StageMeasurement(
            name=name,
            granularity="clip",
            measurement_mode="composed",
            durations=durations,
            memory=MemorySnapshot(),
            window_s=window_s,
            warmup_s=warm_s,
            iterations=len(durations),
            uses_cuda=uses_cuda,
            notes=notes,
            energy_unavailable_reason=INSITU_ENERGY_REASON,
        )

    stage_notes = ("in situ, unpaced batch 1 -- this stage's time inside the composed "
                   "loop, where decode's worker threads can overlap the GPU stages")
    per = {s.name: _measurement(s.name, per_stage[s.name], stage_notes) for s in pipeline}
    end_to_end = _measurement(
        "end_to_end", clip_totals,
        "in situ, unpaced batch 1 -- clip time with every stage timed in place",
    )
    return per, end_to_end

def measure_all_stages_isolated(
    pipeline: Pipeline,
    fixtures: FixtureStore,
    *,
    meter: Optional[EnergyMeter] = None,
    idle_seconds: float = 0.0,
    **kwargs: Any,
) -> dict[str, StageMeasurement]:
    """Every stage, alone, in pipeline order.

    `idle_seconds > 0` measures the idle baseline **immediately before** the
    sweep, at the same clocks and thermal state, and folds it into every
    reading as P2 requires. Taking it once at program start and reusing it
    would understate the increment: a cold machine idles lower than a warm one,
    so every stage's over-idle figure would come out too large -- uniformly,
    and therefore invisibly.

    The same meter serves every stage; each opens and closes its own window in
    turn, so the windows never overlap.
    """
    if meter is not None and idle_seconds > 0:
        meter.measure_idle(idle_seconds)
    out = {}
    for s in pipeline:
        m = measure_stage_isolated(s, fixtures, meter=meter, **kwargs)
        m.stride = pipeline.stride_of(s)
        out[s.name] = m
    return out


@dataclass
class IsolationSweep:
    """P1's three steps as one object: isolated, composed, and the baseline."""

    isolated: dict[str, StageMeasurement]
    composed: StageMeasurement
    idle: Optional[IdleBaseline] = None
    contention: Optional[ContentionReport] = None

    @property
    def energy_is_defensible(self) -> bool:
        """False when the idle bracket drifted, no rail was readable, or the
        machine was shared during the window.

        Over-idle energy is the number P2 actually wants and the one both a
        drifting baseline and a second tenant silently corrupt. The two checks
        are complementary rather than redundant: the bracket samples the
        *boundaries* and catches a tenant who is already there or who stays;
        the contention probe samples *through* the window and catches one who
        arrives and leaves inside it, moving every stage measurement without
        moving either end of the bracket.

        Neither gates the *total* joules, which stand on their own.
        """
        if self.idle is None or self.idle.unavailable_reason is not None:
            return False
        if not self.idle.stable:
            return False
        return self.contention is None or self.contention.clean

    @property
    def contention_verdict(self) -> Optional[str]:
        """The rows' `contention` field: every window quiet, and no unexplained
        rise in the idle bracket (blocker 15)."""
        return contention_verdict(self.contention, self.idle)

    def notes(self) -> str:
        parts = [explain_instability(self.idle, self.contention)
                 if self.idle is not None else "no energy meter"]
        if self.contention is not None:
            parts.append(self.contention.note())
        return " | ".join(parts)


def measure_isolation_sweep(
    pipeline: Pipeline,
    fixtures: FixtureStore,
    source: Any,
    n_frames: int,
    *,
    meter: Optional[EnergyMeter] = None,
    idle_seconds: float = 10.0,
    idle_settle_seconds: Optional[float] = None,
    check_contention: bool = True,
    **kwargs: Any,
) -> IsolationSweep:
    """The whole of P1 in the order it has to happen.

        idle (settle, then measure)  ->  each stage alone  ->  composed
                                     ->  idle again, and compare

    The closing idle reading is what makes the run auditable. It costs one more
    window and it is the only thing standing between "a second tenant was on
    the CPU" and a results table that looks fine.
    """
    # Started BEFORE the opening bracket, so contention is watched across the
    # whole run: settle, idle, every stage, composed, idle again.
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

    isolated = {
        s.name: measure_stage_isolated(
            s, fixtures, meter=meter, contention=probe, **kwargs
        )
        for s in pipeline
    }
    # A strided stage runs ceil(n/S) times per clip; without this its per-frame
    # energy and its share of the isolated sum are charged for every frame.
    for s in pipeline:
        isolated[s.name].stride = pipeline.stride_of(s)
    composed = measure_pipeline_composed(
        pipeline, source, n_frames, meter=meter, contention=probe, **kwargs
    )

    # The probe runs THROUGH both idle brackets, not just the working window.
    # An earlier version stopped it first, reasoning that a deliberate sleep
    # would dilute the foreign-CPU average — which is exactly backwards. During
    # an idle window *this* process is doing nothing, so any CPU or GPU activity
    # is unambiguously someone else's: it is the most sensitive detection window
    # in the whole run, not a dilution.
    #
    # The cost of getting it wrong is visible: a tenant arriving during the
    # closing idle produced "IDLE BASELINE UNSTABLE: 16.6 -> 46.4 W" alongside
    # "machine quiet through the window", two checks contradicting each other
    # with nothing to explain why.
    if baseline is not None and meter is not None:
        if probe is not None:
            probe.mark("idle-close")
        baseline = close_idle_baseline(baseline, meter)

    report = probe.stop() if probe is not None else None

    return IsolationSweep(
        isolated=isolated, composed=composed, idle=baseline, contention=report
    )

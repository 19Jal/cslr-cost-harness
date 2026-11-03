"""sustained.py — warm to plateau, then hold for ten minutes.

    "Warm up until temperature plateaus, then measure for 10 minutes. Log
     thermal_c_end. If sustained is far below peak, the device is throttling —
     that is a headline finding for real-time deployment, not a nuisance to
     smooth away."

Streaming CSLR is a **sustained-load** workload. A benchmark that runs for five
seconds measures a device's boost behaviour, not its steady state, and on a
passively-cooled edge module those differ by a lot. Quoting peak throughput for
something that runs for minutes is the same class of error as quoting batch
throughput as a streaming figure — a common reporting error, and one this harness must not make.

--------------------------------------------------------------------------
THE DEV BOX IS THE WRONG DEVICE TO CARE ABOUT, AND THAT IS THE POINT
--------------------------------------------------------------------------
An RTX 3060 in a desktop case with active cooling will barely throttle. The
Orin Nano is power-capped and cooled by a small fan on a quiet profile, and it
is the device whose numbers go in the paper. So the value of running this here is **not** the
throttle figure; it is that the protocol, the plateau detection and the row
fields are all exercised before they matter, on a machine where a mistake costs
nothing.

Expect `throttle_pct` near zero here. A near-zero result on this box is a
working harness, not a null finding.

--------------------------------------------------------------------------
TWO WAYS TO ASK THE QUESTION, AND THEY ARE NOT THE SAME
--------------------------------------------------------------------------
- **Unpaced** (`paced_fps=None`): how many frames per second can this device
  sustain for ten minutes? Fills `sustained_fps_10min`.
- **Paced**: can it still hold 25 fps once it is hot? Here throughput is fixed
  by construction and the interesting quantity is whether *overruns accumulate*
  as temperature rises — a pipeline that keeps up cold and slips warm looks
  identical to one that never slips, if you only measure for ten seconds.

The second is the deployment question and the first is the headline number, so
both are supported and the row says which was run.

--------------------------------------------------------------------------
A RISING IDLE BRACKET IS EXPECTED HERE, AND THE COLD FLOOR IS STILL RIGHT
--------------------------------------------------------------------------
Every other runner treats a drifting idle bracket as a warning. This one
*deliberately heats the device*, so its closing bracket is higher by design:
the first real hold read 18.5 W before and 23.4 W after, on a machine the
contention probe confirmed was quiet throughout.

The tempting conclusion is that the post-hold reading is the "correct" baseline
— P2 does ask for idle at the same thermal state, and during the hold the
device is hot. It is the wrong conclusion. **There is no steady hot idle**: a
GPU begins cooling the instant the work stops, so 23.4 W is not a property of
the device, it is a property of how long we happened to wait. A 60 s settle
gives one number, a 300 s settle another. Lengthening the settle chases an
asymptote that does not exist.

The cold floor is reproducible, and that is what makes it the right subtrahend.
So `IdleBaseline.watts` falling back to `watts_before` when the bracket is
unstable is not a fallback here — it is the correct answer. Over-idle figures
from a hold stand, provided the machine was quiet: the drift is physics, not
contamination.

(For the record the difference is not trivial: 0.566 J/frame against the cold
floor versus 0.529 against the post-hold reading, about 6.5%.)

--------------------------------------------------------------------------
PEAK IS MEASURED, NOT ASSUMED
--------------------------------------------------------------------------
"Sustained far below peak" needs a peak to compare against, and taking it from
a separate short run would compare across thermal states rather than within
one. So peak comes from *inside* the same window: the fastest single clip, plus
a first-fifth / last-fifth split that is far less noise-sensitive than one best
clip and shows decay directly. If the last fifth is materially slower than the
first, the device is throttling under its own load.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from cslr_cost.harness.energy import (
    ContentionProbe,
    ContentionReport,
    EnergyMeter,
    EnergyReading,
    IdleBaseline,
    close_idle_baseline,
    explain_instability,
    measure_idle_baseline,
)
from cslr_cost.harness.runners.isolation import _start_energy, _stop_energy
from cslr_cost.harness.runners.paced import _rewind
from cslr_cost.harness.runners.streaming import run_streaming
from cslr_cost.harness.stages.base import Pipeline
from cslr_cost.harness.thermal import (
    ThermalTrace,
    any_temperature,
    temperature_source,
    warm_to_plateau,
)
from cslr_cost.harness.timing import (
    Durations,
    MemoryProbe,
    MemorySnapshot,
    make_timer,
)

#: P3's window. Named for what it is so a shortened run cannot be quoted as
#: though it were the real thing.
SUSTAINED_MINUTES = 10.0

#: Settle before the *closing* idle window. Much longer than the 20 s used
#: elsewhere, because ten minutes of sustained load leaves a GPU at boost
#: clocks for a while: the first hold measured a closing "idle" of 148 W
#: against a 17.2 W floor with a 5 s settle, on a demonstrably quiet machine.
POST_HOLD_SETTLE_S = 90.0

#: Below this the device is not meaningfully throttling. Chosen so an actively
#: cooled desktop reads "no", which is the honest answer for this box.
THROTTLE_NOTICEABLE_PCT = 5.0


@dataclass
class SustainedResult:
    """A thermally settled ten-minute hold."""

    warmup: ThermalTrace
    trace: ThermalTrace
    clip_durations: Durations
    n_frames: int
    window_s: float
    memory: MemorySnapshot
    paced_fps: Optional[float] = None
    overrun_frames_first_fifth: int = 0
    overrun_frames_last_fifth: int = 0
    energy: Optional[EnergyReading] = None
    energy_unavailable_reason: Optional[str] = None
    idle: Optional[IdleBaseline] = None
    contention: Optional[ContentionReport] = None
    notes: str = ""

    _fifth_fps: tuple[Optional[float], Optional[float]] = field(
        default=(None, None), repr=False
    )

    # -- throughput --------------------------------------------------------

    @property
    def n_clips(self) -> int:
        return len(self.clip_durations)

    @property
    def sustained_fps(self) -> Optional[float]:
        """Frames per second across the **whole** window. The headline."""
        if self.window_s <= 0 or not self.n_clips:
            return None
        return self.n_clips * self.n_frames / self.window_s

    @property
    def peak_fps(self) -> Optional[float]:
        """The fastest single clip in the window — the boost figure."""
        fastest = self.clip_durations.minimum
        if not fastest:
            return None
        return self.n_frames / (fastest / 1000.0)

    @property
    def first_fifth_fps(self) -> Optional[float]:
        return self._fifth_fps[0]

    @property
    def last_fifth_fps(self) -> Optional[float]:
        return self._fifth_fps[1]

    @property
    def throttle_pct(self) -> Optional[float]:
        """How far sustained falls below peak, as a percentage of peak."""
        peak, sustained = self.peak_fps, self.sustained_fps
        if not peak or not sustained:
            return None
        return 100.0 * (peak - sustained) / peak

    @property
    def decay_pct(self) -> Optional[float]:
        """Last fifth against first fifth — throttling under our own load.

        Less noise-sensitive than peak-vs-sustained, because it averages many
        clips on each side rather than trusting one fast one.
        """
        first, last = self.first_fifth_fps, self.last_fifth_fps
        if not first or not last:
            return None
        return 100.0 * (first - last) / first

    @property
    def throttling(self) -> Optional[bool]:
        decay = self.decay_pct
        return None if decay is None else decay > THROTTLE_NOTICEABLE_PCT

    @property
    def overrun_drift(self) -> Optional[int]:
        """Extra overruns in the last fifth versus the first (paced runs only).

        A pipeline that keeps up cold and slips warm is indistinguishable from
        one that never slips, if you only ever measure it for ten seconds.
        """
        if self.paced_fps is None:
            return None
        return self.overrun_frames_last_fifth - self.overrun_frames_first_fifth

    # -- reporting ---------------------------------------------------------

    @property
    def idle_drift_is_thermal(self) -> bool:
        """True when the closing bracket rose because *we* heated the device.

        Requires the machine to have been quiet (so no tenant explains it) and
        the drift to be upward (cooling toward a floor cannot raise it). Under
        those two conditions the drift is the intended outcome of the run.
        """
        if self.idle is None or self.idle.stable:
            return False
        if self.contention is not None and not self.contention.clean:
            return False
        drift = self.idle.drift_pct
        return drift is not None and drift > 0

    @property
    def energy_is_defensible(self) -> bool:
        """Unlike the other runners, a drifting bracket does not disqualify.

        A hold heats the device on purpose, and the baseline actually used is
        the reproducible cold floor — see the module docstring. What still
        disqualifies is a shared machine, which no baseline choice repairs.
        """
        if self.idle is None or self.idle.unavailable_reason is not None:
            return False
        if self.contention is not None and not self.contention.clean:
            return False
        return self.idle.stable or self.idle_drift_is_thermal

    @property
    def contention_verdict(self) -> Optional[str]:
        """The row's `contention` field. A rising bracket is expected after a
        hold, so it does not void the run -- a tenant still does."""
        from cslr_cost.harness.energy import contention_verdict

        return contention_verdict(self.contention, self.idle, after_sustained_hold=True)

    def idle_note(self) -> str:
        if self.idle_drift_is_thermal:
            return (
                f"idle rose {self.idle.watts_before:.1f} -> "
                f"{self.idle.watts_after:.1f} W ({self.idle.drift_pct:+.1f}%) across "
                "the hold, on a machine that stayed quiet — expected, since the "
                "run heats the device on purpose. Over-idle energy is computed "
                f"against the reproducible cold floor ({self.idle.watts_before:.1f} W); "
                "there is no steady hot idle to use instead."
            )
        return explain_instability(self.idle, self.contention, after_sustained_hold=True)

    def summary(self) -> dict[str, Any]:
        return {
            "sustained_fps_10min": self.sustained_fps,
            "thermal_c_start": self.trace.start_c,
            "thermal_c_end": self.trace.end_c,
            "warmup_s": round(self.warmup.warmup_s, 2),
            "peak_ram_mb": self.memory.peak_ram_mb,
            "peak_vram_mb": self.memory.peak_vram_mb,
            "peak_device_mem_mb": self.memory.peak_device_mem_mb,
            "device_mem_source": self.memory.device_mem_source,
        }

    def table(self) -> str:
        mins = self.window_s / 60.0
        mode = "unpaced" if self.paced_fps is None else f"paced {self.paced_fps:g} fps"
        lines = [
            f"sustained hold — {mins:.1f} min, {mode}, {self.n_clips} clips "
            f"x {self.n_frames} frames",
            "",
            f"  warm-up            {self.warmup.warmup_s:.0f}s"
            + ("" if self.warmup.plateaued else "  ** DID NOT PLATEAU **"),
            f"  temperature        {self.trace.start_c}°C -> {self.trace.end_c}°C"
            f"  (peak {self.trace.peak_c}°C, {self.trace.source})",
            f"  drift in window    {self.trace.slope_c_per_min():+.2f} °C/min"
            if self.trace.slope_c_per_min() is not None else "",
            "",
        ]
        if self.paced_fps is None:
            lines += [
                f"  peak               {self.peak_fps:.1f} fps  (fastest clip)"
                if self.peak_fps else "",
                f"  sustained          {self.sustained_fps:.1f} fps"
                if self.sustained_fps else "",
                f"  first fifth        {self.first_fifth_fps:.1f} fps"
                if self.first_fifth_fps else "",
                f"  last fifth         {self.last_fifth_fps:.1f} fps"
                if self.last_fifth_fps else "",
                f"  throttle           {self.throttle_pct:.1f}% below peak"
                if self.throttle_pct is not None else "",
                f"  decay              {self.decay_pct:+.1f}% first -> last fifth"
                if self.decay_pct is not None else "",
            ]
        else:
            lines += [
                f"  overruns first/last fifth   "
                f"{self.overrun_frames_first_fifth} / {self.overrun_frames_last_fifth}",
                f"  drift                       {self.overrun_drift:+d} frames"
                if self.overrun_drift is not None else "",
            ]
        lines.append("")
        if self.throttling:
            lines.append("  THROTTLING — report sustained, never peak")
        elif self.throttling is False:
            lines.append("  no meaningful throttling on this device")
        if self.contention is not None:
            lines.append(f"  {self.contention.note()}")
        if self.idle is not None:
            lines.append(f"  {self.idle_note()}")
        return "\n".join(line for line in lines if line != "")


def run_sustained(
    pipeline: Pipeline,
    source: Any,
    n_frames: int,
    *,
    minutes: float = SUSTAINED_MINUTES,
    paced_fps: Optional[float] = None,
    meter: Optional[EnergyMeter] = None,
    idle_seconds: float = 10.0,
    idle_settle_seconds: Optional[float] = None,
    check_contention: bool = True,
    plateau_max_seconds: float = 600.0,
    poll_memory: bool = True,
) -> SustainedResult:
    """P3 end to end: warm to plateau, then hold, sampling temperature.

    The idle bracket is taken **before warm-up**, not between warm-up and the
    window: the whole point of warming up is to leave the device hot, so an
    idle reading taken after it would measure a settling device rather than a
    floor, and every over-idle figure would come out too small.
    """
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")

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

    def one_clip() -> None:
        _rewind(source)
        if paced_fps is None:
            pipeline.run(source, n_frames)
        else:
            run_streaming(pipeline, source, n_frames,
                          paced_fps=paced_fps, poll_memory=False)

    # --- warm to plateau (P3), outside the measured window ---------------
    warmup = warm_to_plateau(one_clip, max_seconds=plateau_max_seconds)

    # --- the hold ---------------------------------------------------------
    uses_cuda = any(s.uses_cuda for s in pipeline)
    timer = make_timer(uses_cuda)
    mem = MemoryProbe(track_vram=uses_cuda).start() if poll_memory else None
    trace = ThermalTrace(source=temperature_source())
    clip_durations = Durations()
    overruns: list[int] = []

    if probe is not None:
        probe.mark("hold")
    meter = _start_energy(meter)
    start = time.perf_counter()
    window_s = minutes * 60.0
    next_temp = start

    while time.perf_counter() - start < window_s:
        if paced_fps is None:
            _rewind(source)
            with timer:
                pipeline.run(source, n_frames)
            clip_durations.add(timer.elapsed_ms)
            overruns.append(0)
        else:
            _rewind(source)
            t0 = time.perf_counter()
            result = run_streaming(pipeline, source, n_frames,
                                   paced_fps=paced_fps, poll_memory=False)
            clip_durations.add((time.perf_counter() - t0) * 1000.0)
            overruns.append(result.overrun_frames)

        now = time.perf_counter()
        if now >= next_temp:
            c = any_temperature()
            if c is not None:
                trace.add(c, now)
            next_temp = now + 1.0
        if mem is not None:
            mem.poll()
        if probe is not None:
            probe.poll()

    energy, energy_reason = _stop_energy(meter, uses_cuda)
    elapsed = time.perf_counter() - start
    if probe is not None:
        probe.mark(None)
    memory = mem.stop() if mem is not None else MemorySnapshot()

    if baseline is not None and meter is not None:
        # A hot GPU needs far longer to reach idle than a merely warm one, so
        # the closing bracket gets its own settle rather than reusing the
        # opening one's.
        baseline.settle_s = max(baseline.settle_s, POST_HOLD_SETTLE_S)
        if probe is not None:
            probe.mark("idle-close")
        baseline = close_idle_baseline(baseline, meter)
    report = probe.stop() if probe is not None else None

    # --- first fifth vs last fifth ---------------------------------------
    fifths: tuple[Optional[float], Optional[float]] = (None, None)
    samples = clip_durations.samples_ms
    k = len(samples) // 5
    if k >= 1:
        def fps(chunk: list[float]) -> float:
            return len(chunk) * n_frames / (sum(chunk) / 1000.0)
        fifths = (fps(samples[:k]), fps(samples[-k:]))

    first_over = sum(overruns[:k]) if k >= 1 else 0
    last_over = sum(overruns[-k:]) if k >= 1 else 0

    notes = []
    if not warmup.plateaued:
        notes.append(
            f"temperature had NOT plateaued after {warmup.warmup_s:.0f}s "
            f"(cap {plateau_max_seconds:.0f}s) — these are not steady-state numbers"
        )
    if elapsed < SUSTAINED_MINUTES * 60.0:
        notes.append(
            f"held {elapsed / 60.0:.1f} min, short of P3's 10 — not a "
            "sustained-throughput figure"
        )
    if trace.source == "unavailable":
        notes.append("no temperature sensor readable; thermal fields are null")

    return SustainedResult(
        warmup=warmup,
        trace=trace,
        clip_durations=clip_durations,
        n_frames=n_frames,
        window_s=elapsed,
        memory=memory,
        paced_fps=paced_fps,
        overrun_frames_first_fifth=first_over,
        overrun_frames_last_fifth=last_over,
        energy=energy,
        energy_unavailable_reason=energy_reason,
        idle=baseline,
        contention=report,
        notes=" | ".join(notes),
        _fifth_fps=fifths,
    )

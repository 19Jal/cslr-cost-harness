"""baseline.py — the idle baseline, and knowing when not to trust it.

P2 asks for energy reported two ways: total, and incremental over idle. The
increment is the more useful of the two -- "what does this stage cost" rather
than "what does this machine cost" -- and it is the one that can be quietly
wrong, because it is a *difference* and a difference amplifies any error in
either term.

--------------------------------------------------------------------------
TWO WAYS AN IDLE BASELINE GOES WRONG, BOTH OBSERVED HERE
--------------------------------------------------------------------------
**1. Measured too soon.** A window opened the instant a load stops measures
the decay, not the floor. On this dev box a 3 s window taken immediately after
a GPU sweep read **66.2 W**; the settled floor is **17.2 W** (14.9 GPU +
2.3 CPU package). Subtracting the first from every stage would have turned most
of the front end's over-idle energy negative. Hence `settle_seconds`.

**2. Another tenant.** This is a shared lab machine, and a second person's
training run does not announce itself. A settled 3 s idle window taken while
someone else's job was on the CPU read **87.8 W** -- higher than the
immediately-post-load figure -- and one minute later the same rail read
2.3 W. Nothing in the reading says which of those it is.

So idle is measured **twice, bracketing the sweep**, and the two are compared.
A baseline that drifts is reported as unstable rather than averaged into a
plausible-looking number: an unstable bracket means something outside this
process changed during the run, and that something also moved every stage
measurement in between. The right response is to re-run, not to correct.

`stable == False` is not a failure of the harness. It is the harness declining
to launder a contaminated measurement into a clean-looking row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cslr_cost.harness.energy.base import EnergyMeter, EnergyUnavailable

#: How far the closing idle reading may drift from the opening one before the
#: run is called contaminated. Generous, because idle draw genuinely wanders a
#: few percent with fan curves and background daemons -- but a second tenant
#: moves it by tens of percent, which is what this catches.
DEFAULT_DRIFT_TOLERANCE_PCT = 20.0

#: A closing idle this far ABOVE the opening one voids the run's contention
#: verdict, unless a sustained hold came just before it (blocker 15). Our own
#: device can still be warm 20 s after the last pass -- the v2 grid has a +21.6%
#: (+4 W) case with every window quiet -- but the two contaminated cells it
#: shipped rose +174% and +483%. Between the drift tolerance and this, the rise
#: is reported and over-idle energy withheld, as before; beyond it the row is
#: not clean. A FALL is never voided: it means the opening bracket caught our
#: device still settling from the preceding run.
VOID_RISE_PCT = 50.0


@dataclass
class IdleBaseline:
    """Idle draw measured either side of a run, and whether it held."""

    watts_before: float
    watts_after: Optional[float] = None
    settle_s: float = 0.0
    window_s: float = 0.0
    tolerance_pct: float = DEFAULT_DRIFT_TOLERANCE_PCT
    unavailable_reason: Optional[str] = None

    @property
    def watts(self) -> float:
        """The figure to subtract: the mean of the bracket when it held.

        When it did not hold, the opening reading is returned and `stable` is
        False -- the caller is expected to surface that rather than quietly use
        this number.
        """
        if self.watts_after is None or not self.stable:
            return self.watts_before
        return (self.watts_before + self.watts_after) / 2.0

    @property
    def drift_pct(self) -> Optional[float]:
        if self.watts_after is None or self.watts_before == 0:
            return None
        return 100.0 * (self.watts_after - self.watts_before) / self.watts_before

    @property
    def stable(self) -> bool:
        drift = self.drift_pct
        return True if drift is None else abs(drift) <= self.tolerance_pct

    def status(self) -> str:
        """Compact, filterable verdict for the results row.

        `note()` is prose for a human reading a terminal; this is what a row
        carries, so a table can be filtered months later without parsing
        English. Over-idle energy depends entirely on this having held, and
        until now nothing in the row recorded whether it did — the same class
        of gap as a null `git_sha`.
        """
        if self.unavailable_reason:
            return "unavailable"
        if self.watts_after is None:
            return f"unchecked ({self.watts_before:.1f} W)"
        verdict = "stable" if self.stable else "unstable"
        return (
            f"{verdict} {self.watts_before:.1f}->{self.watts_after:.1f} W "
            f"({self.drift_pct:+.1f}%)"
        )

    def note(self) -> str:
        if self.unavailable_reason:
            return f"idle baseline unavailable: {self.unavailable_reason}"
        if self.watts_after is None:
            return f"idle {self.watts_before:.1f} W (not re-checked after the run)"
        if self.stable:
            return (
                f"idle {self.watts:.1f} W "
                f"({self.watts_before:.1f} -> {self.watts_after:.1f} W, "
                f"drift {self.drift_pct:+.1f}%)"
            )
        return (
            f"IDLE BASELINE UNSTABLE: {self.watts_before:.1f} -> "
            f"{self.watts_after:.1f} W ({self.drift_pct:+.1f}%, tolerance "
            f"±{self.tolerance_pct:.0f}%). Something outside this process changed "
            "during the run and moved every stage measurement with it. "
            "Over-idle figures from this run are not defensible — re-run on a "
            "quiet machine."
        )


def measure_idle_baseline(
    meter: EnergyMeter,
    *,
    seconds: float = 10.0,
    settle_seconds: Optional[float] = None,
    tolerance_pct: float = DEFAULT_DRIFT_TOLERANCE_PCT,
) -> IdleBaseline:
    """Open the bracket: settle, then measure idle once.

    Close it with `close_idle_baseline` after the run.
    """
    settle = (
        meter.DEFAULT_SETTLE_SECONDS if settle_seconds is None else settle_seconds
    )
    try:
        watts = meter.measure_idle(seconds=seconds, settle_seconds=settle)
    except EnergyUnavailable as exc:
        return IdleBaseline(
            watts_before=float("nan"),
            settle_s=settle,
            window_s=seconds,
            tolerance_pct=tolerance_pct,
            unavailable_reason=str(exc),
        )
    return IdleBaseline(
        watts_before=watts,
        settle_s=settle,
        window_s=seconds,
        tolerance_pct=tolerance_pct,
    )


def close_idle_baseline(
    baseline: IdleBaseline, meter: EnergyMeter
) -> IdleBaseline:
    """Close the bracket, and leave the meter subtracting the agreed figure.

    The meter's `idle_watts` is updated to the bracket's value, so readings
    taken *after* this call carry the settled figure. Readings taken during the
    run already used the opening one -- which is correct, and is why the drift
    check matters rather than being cosmetic.
    """
    if baseline.unavailable_reason:
        return baseline
    try:
        after = meter.measure_idle(
            seconds=baseline.window_s, settle_seconds=baseline.settle_s
        )
    except EnergyUnavailable as exc:
        baseline.unavailable_reason = str(exc)
        return baseline

    baseline.watts_after = after
    meter.idle_watts = baseline.watts
    return baseline


def unexplained_idle_rise(baseline, after_sustained_hold: bool = False) -> bool:
    """True when the closing idle rose past `VOID_RISE_PCT` with no hold before it.

    Only a sustained hold heats the device enough to explain a rise that large
    (the first hold: 66.8 -> 148.0 W). After an isolation, paced or batch sweep
    the last pass ended 20 s before the closing window opened, and the two
    rises of that size in the v2 grid were both another process -- one of them
    an INT8 export run alongside the grid.
    """
    if baseline is None or baseline.unavailable_reason or after_sustained_hold:
        return False
    drift = baseline.drift_pct
    return drift is not None and drift > VOID_RISE_PCT


def contention_verdict(contention, baseline=None, after_sustained_hold: bool = False) -> Optional[str]:
    """The row's `contention` field: "clean", or why the run is not.

    One place, so every regime applies the same rule. Clean requires the probe
    to be quiet in every window it judged **and** the idle bracket not to have
    risen past `VOID_RISE_PCT` -- a large unexplained rise is voided, not
    explained (blocker 15). None when contention was not checked.
    """
    if contention is None:
        return None
    if not contention.clean:
        return contention.note()
    if unexplained_idle_rise(baseline, after_sustained_hold):
        return (
            f"IDLE ROSE WITH NO HOLD TO EXPLAIN IT: {baseline.watts_before:.1f} -> "
            f"{baseline.watts_after:.1f} W ({baseline.drift_pct:+.1f}%, void above "
            f"+{VOID_RISE_PCT:.0f}%). The probe saw no foreign process over tolerance, "
            "but something was drawing power it could not attribute; every stage "
            "measured before the closing bracket is suspect. Re-run on a quiet machine."
        )
    return "clean"


def explain_instability(baseline, contention, after_sustained_hold: bool = False) -> str:
    """Read the idle bracket and the contention probe *together*.

    Neither means much alone once both cover the same span. A drifting bracket
    has three causes, told apart by its direction and by what preceded it:

    - **fell, machine quiet** -- the *opening* bracket caught our own device
      still settling from the run before. Lengthen `settle_seconds`; nothing was
      contaminated. (Five such runs in the v2 grid, -25% to -80%.)
    - **rose, after a sustained hold** -- the hold heated the device on purpose
      and the closing window catches it on the way down.
    - **rose, no hold** -- small rises can be our own residual warmth; past
      `VOID_RISE_PCT` it is someone else, and `contention_verdict` voids the
      run. The first version called every quiet-machine drift "our own device
      settling after a sustained hold" -- including the two contaminated v2
      cells, neither of which followed a hold (blocker 15).
    - **machine shared** -- the probe says so, and the drift agrees.
    """
    if baseline is None:
        return "no idle baseline"
    if baseline.unavailable_reason:
        return baseline.note()
    if baseline.stable:
        return baseline.note()

    shared = contention is not None and not contention.clean
    if shared:
        return (
            f"{baseline.note()} The contention probe agrees: {contention.note()}"
        )
    if contention is None:
        return baseline.note()

    move = (
        f"{baseline.watts_before:.1f} -> {baseline.watts_after:.1f} W "
        f"({baseline.drift_pct:+.1f}%)"
    )
    unusable = "Over-idle figures from this run remain unusable."
    if baseline.drift_pct < 0:
        return (
            f"idle baseline fell {move} on a machine quiet in every window: the "
            "opening bracket caught OUR OWN device still settling from the preceding "
            f"run, not a second tenant -- lengthen settle_seconds rather than re-running. {unusable}"
        )
    if after_sustained_hold:
        return (
            f"idle baseline rose {move} after a sustained hold, on a quiet machine: "
            "OUR OWN device still settling from the hold -- lengthen settle_seconds "
            f"rather than re-running. {unusable}"
        )
    if unexplained_idle_rise(baseline):
        return (
            f"idle baseline ROSE {move} with no sustained hold before it -- VOID. "
            "The probe was quiet in every window it judged, but our own device does "
            "not warm up by this much 20 s after a sweep; something it could not "
            "attribute was drawing power. The run's contention verdict is not clean."
        )
    return (
        f"idle baseline rose {move} on a machine quiet in every window: small enough "
        "to be our own device still warm from the last pass; lengthen settle_seconds. "
        f"{unusable}"
    )

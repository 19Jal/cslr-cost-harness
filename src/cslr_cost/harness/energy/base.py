"""base.py — the energy measurement interface.

An `EnergyMeter` reports joules consumed over a measurement window.

**Counters, not samples, wherever possible.** Both rails available here expose a
monotonically-increasing energy counter — Intel RAPL in microjoules, NVML in
millijoules — so a window's energy is just `end - start`. That is *exact*: it
accumulates in hardware and cannot miss a transient the way a sampler can.

This matters more than it sounds. P1 warns that power monitors sample at
10-100 ms while pipeline stages take 1-20 ms, and concludes you cannot
attribute energy to stages from a single composed pass. That reasoning is about
*sampled power*. With a hardware counter, a >=60 s isolation window still gives
an exact figure for that window — which is why the isolation protocol works at
all. Sampling is the fallback, not the design.

**P2: idle must be subtracted and stated.** Every reading carries total joules
*and*, when an idle baseline has been measured at the same clocks and thermal
state, the incremental-over-idle figure. They answer different questions —
"what does this cost my battery" versus "what does this stage cost" — and
papers that conflate them are unusable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class EnergyReading:
    """Energy consumed over one measurement window.

    `joules` is total system energy on the rails this meter covers.
    `joules_incremental` is that minus idle draw for the same duration — None
    until an idle baseline has been measured.
    """

    joules: float
    seconds: float
    backend: str
    per_rail_joules: dict[str, float] = field(default_factory=dict)
    idle_watts: Optional[float] = None
    #: Backend-specific facts about how the window was measured -- for a
    #: sampled rail, the sample count and the refresh actually observed --
    #: so the integral's resolution is auditable after the fact.
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def watts(self) -> float:
        return self.joules / self.seconds if self.seconds > 0 else float("nan")

    @property
    def joules_incremental(self) -> Optional[float]:
        """Energy above idle. P2 — reported alongside the total, never instead."""
        if self.idle_watts is None:
            return None
        return self.joules - self.idle_watts * self.seconds

    def per_unit(self, iterations: int) -> Optional[float]:
        """Joules per iteration — P1's 'integrate over the window; divide by
        iterations'."""
        return self.joules / iterations if iterations else None

    def per_unit_incremental(self, iterations: int) -> Optional[float]:
        inc = self.joules_incremental
        if inc is None or not iterations:
            return None
        return inc / iterations


def integrate_trapezoid(samples: Sequence[tuple[float, float]]) -> float:
    """Energy from `(seconds, watts)` samples, by the trapezoidal rule.

    Trapezoid rather than rectangle because power ramps between samples;
    assuming it held constant would bias every window in the same direction,
    which is worse than being noisy. Fewer than two samples integrate to 0.0 --
    callers must refuse such a window rather than report that zero.
    """
    if len(samples) < 2:
        return 0.0
    total = 0.0
    for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
        total += (w0 + w1) / 2.0 * (t1 - t0)
    return total


class EnergyMeter(ABC):
    """Measures energy over a window.

        meter.start()
        ...work...
        reading = meter.stop()
    """

    #: Value for the results row's `energy_backend` field.
    backend: str = "unavailable"

    @staticmethod
    @abstractmethod
    def available() -> bool:
        """True if this rail can actually be read here.

        Implementations must probe rather than assume: a rail that exists but
        is unreadable (RAPL without the udev rule, NVML in a container without
        the device) must report False so the harness records an explicit
        unavailability reason instead of a zero.
        """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> EnergyReading: ...

    #: Set by `measure_idle`; folded into subsequent readings.
    idle_watts: Optional[float] = None

    #: Seconds to wait before an idle window opens, letting boost clocks and
    #: fans wind down. Measured, not guessed: on the dev box a 3 s idle window
    #: taken immediately after a GPU sweep read **66.2 W** against a settled
    #: idle of ~17 W, because the card had not yet dropped out of boost. That
    #: baseline would have been subtracted from every stage, turning most of
    #: the front end's over-idle energy negative.
    DEFAULT_SETTLE_SECONDS = 20.0

    def measure_idle(
        self, seconds: float = 10.0, settle_seconds: Optional[float] = None
    ) -> float:
        """Idle draw, measured at the same clocks and thermal state (P2).

        Deliberately does no work: it sleeps, so the machine is as close to
        idle as this process can make it. Run it adjacent to the real
        measurement, not once at program start — an idle baseline taken cold
        and applied to a warm run understates the increment.

        `settle_seconds` is the other half of "adjacent": a window opened the
        instant a load stops measures the *decay*, not the floor. Set it to 0
        only when the device is known to be already idle.
        """
        settle = self.DEFAULT_SETTLE_SECONDS if settle_seconds is None else settle_seconds
        if settle > 0:
            time.sleep(settle)
        self.start()
        time.sleep(seconds)
        reading = self.stop()
        self.idle_watts = reading.watts
        return self.idle_watts


class NullEnergyMeter(EnergyMeter):
    """Used when no rail is readable.

    Exists so the harness never has to choose between crashing and inventing a
    number. It reports why it cannot measure, and the schema turns that into a
    row with null energy plus a stated reason rather than a silent zero.
    """

    backend = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self._t0 = 0.0

    @staticmethod
    def available() -> bool:
        return True

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> EnergyReading:
        raise EnergyUnavailable(self.reason)

    def measure_idle(
        self, seconds: float = 10.0, settle_seconds: Optional[float] = None
    ) -> float:
        raise EnergyUnavailable(self.reason)


class EnergyUnavailable(RuntimeError):
    """No energy rail could be read. Carries the reason for the results row."""

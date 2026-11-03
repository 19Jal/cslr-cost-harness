"""Energy measurement.

`auto_meter()` assembles whatever rails this machine actually exposes. On a
Jetson that is the on-board INA3221, whose `VDD_IN` rail already covers CPU,
GPU and SoC, so it is used alone (`energy/tegra.py`). Elsewhere a
composite is used rather than a single rail because the pipeline spans both
processors: `pose` and `head` are GPU work, while `decode`, `crop` and
`postproc` are CPU-bound and draw nothing measurable on the GPU rail. Measuring
only one would silently attribute zero to whole stages.

When no rail is readable the result is a `NullEnergyMeter` carrying the reason,
which the schema turns into a row with null energy plus a stated explanation —
never a zero.
"""

from __future__ import annotations

import time
from typing import Optional

from cslr_cost.harness.energy.base import (  # noqa: F401
    EnergyMeter,
    EnergyReading,
    EnergyUnavailable,
    NullEnergyMeter,
)
from cslr_cost.harness.energy.nvml import NvmlMeter  # noqa: F401
from cslr_cost.harness.energy.rapl import RaplMeter, rapl_root  # noqa: F401
from cslr_cost.harness.energy.tegra import TegraIna3221Meter, find_ina3221  # noqa: F401


class CompositeMeter(EnergyMeter):
    """Several rails measured together, reported as one total plus a breakdown.

    The per-rail breakdown is kept because "the GPU did 80% of the joules" is
    itself a finding — and on a memory-bound accelerator it may not hold, which
    the protocol treats as a result rather than a failure.
    """

    def __init__(self, meters: list[EnergyMeter]) -> None:
        if not meters:
            raise ValueError("CompositeMeter needs at least one rail")
        self.meters = meters
        self.backend = "+".join(sorted(m.backend for m in meters))

    @staticmethod
    def available() -> bool:
        return RaplMeter.available() or NvmlMeter.available()

    def start(self) -> None:
        for m in self.meters:
            m.start()

    def stop(self) -> EnergyReading:
        joules = 0.0
        seconds = 0.0
        per_rail: dict[str, float] = {}
        for m in self.meters:
            r = m.stop()
            joules += r.joules
            seconds = max(seconds, r.seconds)
            for k, v in r.per_rail_joules.items():
                per_rail[f"{m.backend}:{k}"] = v
        return EnergyReading(
            joules=joules,
            seconds=seconds,
            backend=self.backend,
            per_rail_joules=per_rail,
            idle_watts=self.idle_watts,
        )

    # measure_idle is inherited: the base implementation drives start()/stop()
    # through this class and adds the settle period, which every rail needs.


from cslr_cost.harness.energy.contention import (  # noqa: E402,F401
    DEFAULT_FOREIGN_CPU_TOLERANCE,
    ContentionProbe,
    ContentionReport,
)
from cslr_cost.harness.energy.baseline import (  # noqa: E402,F401
    DEFAULT_DRIFT_TOLERANCE_PCT,
    VOID_RISE_PCT,
    IdleBaseline,
    close_idle_baseline,
    contention_verdict,
    explain_instability,
    measure_idle_baseline,
    unexplained_idle_rise,
)


def auto_meter(*, device_index: int = 0) -> EnergyMeter:
    """Every rail this machine exposes, or a NullEnergyMeter explaining why not."""
    from cslr_cost.harness.device import is_tegra

    if is_tegra():
        # Neither dev-box rail applies: RAPL is x86-only and Tegra NVML refuses
        # power queries. Reporting their absence would send the reader looking
        # for a udev rule on a board that cannot have one.
        if TegraIna3221Meter.available():
            return TegraIna3221Meter()
        return NullEnergyMeter(
            "Jetson detected but the INA3221 hwmon rails are not readable "
            "(no 'ina3221' under /sys/class/hwmon, or no VDD_IN channel); "
            "see docs/devices.md"
        )

    meters: list[EnergyMeter] = []
    reasons: list[str] = []

    if NvmlMeter.available():
        meters.append(NvmlMeter(device_index=device_index))
    else:
        reasons.append("NVML unavailable (no NVIDIA device reporting energy or power to this process)")

    if RaplMeter.available():
        meters.append(RaplMeter())
    elif rapl_root() is None:
        reasons.append(
            "RAPL powercap tree not found — in a container, mount the host's "
            "/sys/devices/virtual/powercap and set CSLR_RAPL_ROOT (scripts/dev.sh does this)"
        )
    else:
        reasons.append(
            "RAPL present but energy_uj is not readable — 0400 root-only since "
            "CVE-2020-8694; apply the udev rule in docs/devices.md"
        )

    if not meters:
        return NullEnergyMeter("; ".join(reasons))
    if len(meters) == 1:
        m = meters[0]
        if reasons:
            m.partial_reason = "; ".join(reasons)  # type: ignore[attr-defined]
        return m
    return CompositeMeter(meters)


def describe_rails() -> dict[str, object]:
    """What is readable here — for docs/devices.md and provenance."""
    info: dict[str, object] = {}
    if NvmlMeter.available():
        m = NvmlMeter()
        info["nvml"] = {
            "device": m.device_name(),
            "mode": "hardware counter" if m.uses_hardware_counter else "sampled power",
        }
    else:
        info["nvml"] = None

    root = rapl_root()
    if root and RaplMeter.available():
        info["rapl"] = {"root": str(root), "domains": [d.name for d in RaplMeter().domains]}
    else:
        info["rapl"] = None

    if TegraIna3221Meter.available():
        m = TegraIna3221Meter()
        info["tegra_ina3221"] = {
            "hwmon": str(m.hwmon),
            "rails": m.rail_labels,
            "mode": "sampled power, VDD_IN total",
            "interval_s": m.interval_s,
        }
        m.close()
    else:
        info["tegra_ina3221"] = None
    return info


__all__ = [
    "EnergyMeter",
    "EnergyReading",
    "EnergyUnavailable",
    "NullEnergyMeter",
    "NvmlMeter",
    "RaplMeter",
    "TegraIna3221Meter",
    "find_ina3221",
    "CompositeMeter",
    "auto_meter",
    "describe_rails",
    "rapl_root",
    "IdleBaseline",
    "measure_idle_baseline",
    "close_idle_baseline",
    "explain_instability",
    "contention_verdict",
    "unexplained_idle_rise",
    "DEFAULT_DRIFT_TOLERANCE_PCT",
    "VOID_RISE_PCT",
    "ContentionProbe",
    "ContentionReport",
    "DEFAULT_FOREIGN_CPU_TOLERANCE",
]

"""thermal.py — temperature and the warm-to-plateau protocol.

    "Warm up until temperature plateaus, then measure for 10 minutes. Log
     thermal_c_end. If sustained is far below peak, the device is throttling —
     that is a headline finding for real-time deployment, not a nuisance to
     smooth away."

Why this exists at all: streaming CSLR is a **sustained-load** workload. A
benchmark that runs for five seconds measures the device's boost behaviour, not
its steady state, and on a passively-cooled edge module those differ by a lot.
Reporting peak throughput for a workload that runs for minutes would be the
same class of error as quoting batch throughput as a streaming figure.

Plateau detection is deliberately conservative: it requires the temperature
*slope* to stay under a threshold across a whole window, not merely two
consecutive readings to be close. Fan behaviour is steppy, so an instantaneous
comparison declares plateau during any brief flat spot on the way up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

#: A device is considered thermally settled when the trend over the observation
#: window is under this many degrees per minute.
DEFAULT_PLATEAU_SLOPE_C_PER_MIN = 0.5
DEFAULT_PLATEAU_WINDOW_S = 60.0
DEFAULT_MAX_WARMUP_S = 600.0


@dataclass
class ThermalTrace:
    """Temperature samples over a window, and what they imply."""

    samples: list[tuple[float, float]] = field(default_factory=list)  # (t, °C)
    source: str = "unknown"
    plateaued: bool = False
    warmup_s: float = 0.0

    def add(self, celsius: float, t: Optional[float] = None) -> None:
        self.samples.append((t if t is not None else time.perf_counter(), celsius))

    @property
    def start_c(self) -> Optional[float]:
        return self.samples[0][1] if self.samples else None

    @property
    def end_c(self) -> Optional[float]:
        return self.samples[-1][1] if self.samples else None

    @property
    def peak_c(self) -> Optional[float]:
        return max((c for _, c in self.samples), default=None)

    def slope_c_per_min(self, window_s: float = DEFAULT_PLATEAU_WINDOW_S) -> Optional[float]:
        """Least-squares slope over the trailing `window_s`.

        Least squares rather than (last - first) / elapsed because a single
        noisy endpoint would otherwise decide whether the device is declared
        settled.
        """
        if len(self.samples) < 3:
            return None
        t_end = self.samples[-1][0]
        pts = [(t, c) for t, c in self.samples if t >= t_end - window_s]
        if len(pts) < 3:
            return None
        n = len(pts)
        mean_t = sum(t for t, _ in pts) / n
        mean_c = sum(c for _, c in pts) / n
        denom = sum((t - mean_t) ** 2 for t, _ in pts)
        if denom == 0:
            return 0.0
        slope_per_s = sum((t - mean_t) * (c - mean_c) for t, c in pts) / denom
        return slope_per_s * 60.0


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

def gpu_temperature() -> Optional[float]:
    """GPU temperature via NVML, or None."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None


def cpu_temperature(base: Path = Path("/sys/class/thermal")) -> Optional[float]:
    """CPU package temperature from sysfs thermal zones, or None.

    Prefers a zone whose type names a package/CPU sensor; falls back to the
    hottest zone, because on many boards the useful sensor is not zone 0 and a
    wrong-but-plausible number is worse than a hot one.
    """
    if not base.is_dir():
        return None

    best: Optional[float] = None
    for zone in sorted(base.glob("thermal_zone*")):
        try:
            ztype = (zone / "type").read_text().strip().lower()
            milli = int((zone / "temp").read_text())
        except (OSError, ValueError):
            continue
        celsius = milli / 1000.0
        if any(k in ztype for k in ("pkg", "package", "x86", "cpu", "soc")):
            return celsius
        best = celsius if best is None else max(best, celsius)
    return best


def junction_temperature(base: Path = Path("/sys/class/thermal")) -> Optional[float]:
    """Jetson junction temperature (`tj-thermal`), or None elsewhere.

    On the Orin this is the sensor the throttling policy acts on: `tj` is the
    maximum across the CPU, GPU and SoC zones. `cpu_temperature()` would pick
    `cpu-thermal`, the first zone whose name contains "cpu" -- a real sensor,
    but not the one that decides when P3's sustained run slows down, and a
    GPU-heavy stage heats `gpu-thermal` first. The `cv*` zones on the same
    board refuse reads with ENODATA; they are skipped like any unreadable zone.
    """
    if not base.is_dir():
        return None
    for zone in sorted(base.glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() != "tj-thermal":
                continue
            return int((zone / "temp").read_text()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def any_temperature() -> Optional[float]:
    """Best available temperature: Jetson junction, then GPU, then CPU."""
    for sensor in (junction_temperature, gpu_temperature, cpu_temperature):
        c = sensor()
        if c is not None:
            return c
    return None


def temperature_source() -> str:
    if junction_temperature() is not None:
        return "sysfs_tj_thermal"
    if gpu_temperature() is not None:
        return "nvml_gpu"
    if cpu_temperature() is not None:
        return "sysfs_thermal_zone"
    return "unavailable"


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------

def warm_to_plateau(
    workload: Callable[[], None],
    *,
    sensor: Callable[[], Optional[float]] = any_temperature,
    slope_threshold: float = DEFAULT_PLATEAU_SLOPE_C_PER_MIN,
    window_s: float = DEFAULT_PLATEAU_WINDOW_S,
    max_seconds: float = DEFAULT_MAX_WARMUP_S,
    poll_interval_s: float = 1.0,
) -> ThermalTrace:
    """Run `workload` repeatedly until temperature plateaus (P3).

    Returns the trace whether or not it settled. `plateaued=False` means the
    cap was hit first, and the caller must say so in the results row rather
    than presenting the numbers as steady state.
    """
    trace = ThermalTrace(source=temperature_source())
    start = time.perf_counter()
    next_poll = start

    while True:
        workload()
        now = time.perf_counter()

        if now >= next_poll:
            c = sensor()
            if c is not None:
                trace.add(c, now)
            next_poll = now + poll_interval_s

        elapsed = now - start
        if elapsed >= window_s:
            slope = trace.slope_c_per_min(window_s)
            if slope is not None and abs(slope) < slope_threshold:
                trace.plateaued = True
                break
        if elapsed >= max_seconds:
            break

    trace.warmup_s = time.perf_counter() - start
    return trace

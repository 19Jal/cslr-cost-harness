"""tegra.py — Jetson on-board energy, from the INA3221 power monitor.

The Orin is the device that has to carry the headline per-stage table, and it
is the one where none of the dev box's energy machinery works:

- **NVML** answers on Tegra — it returns a handle and the name `Orin (nvgpu)`
  — but both `nvmlDeviceGetTotalEnergyConsumption` and
  `nvmlDeviceGetPowerUsage` return NOT_SUPPORTED (verified on L4T R39.2.1).
  Before this backend existed, `auto_meter()` on the Orin returned an NVML
  meter in sampled mode whose sampler died on its first call and integrated an
  empty list: **0.0 J, 0.0 W idle**, with a `partial_reason` about RAPL.
- **RAPL** does not exist on ARM.

What the board does have is a TI INA3221 on I2C, exposed through hwmon:

    in1_label  VDD_IN          module input — the total
    in2_label  VDD_CPU_GPU_CV  CPU + GPU + CV accelerators
    in3_label  VDD_SOC         SoC

--------------------------------------------------------------------------
A SAMPLED RAIL, NOT A COUNTER
--------------------------------------------------------------------------
`energy/base.py` argues for hardware counters because a window's energy is then
exact. The INA3221 has no counter; it reports instantaneous bus voltage (mV)
and current (mA). Energy has to be integrated from samples, so this backend is
tagged `tegra_ina3221` and must never share a table column with `nvml`/`rapl`
without saying so.

The window floor still makes it defensible. P1's 60 s isolation window at the
default 10 Hz is ~600 samples, and the chip averages 512 conversions per
reading, so each sample is already a low-passed mean rather than a point that
can land between transients. What sampling *cannot* do is resolve a 10 ms
stage from a composed pass — which is P1's reason for isolating stages at all.

Two measured properties, recorded per window rather than assumed:

- **Refresh is load-dependent.** The reported current changed every ~400 ms at
  idle, ~160 ms under a TensorRT load and ~90 ms while power decayed after it.
  Values repeat while power is steady. `EnergyReading.detail` carries the
  observed interval between changes for every window.
- **Reading is not free.** Each sysfs read is an I2C transaction: ~0.56 ms per
  file, ~3.3 ms for all three rails (six files). At 10 Hz that is ~33 ms of CPU
  per second on one core — on the rail being measured. The sample rate is a
  constructor argument so the cost can be traded against resolution, and the
  per-window read cost is recorded too.

--------------------------------------------------------------------------
VDD_IN IS THE TOTAL — DO NOT SUM THE RAILS
--------------------------------------------------------------------------
The same trap as RAPL's `package-0`. At idle `VDD_IN` drew 616 mA while
`VDD_CPU_GPU_CV` + `VDD_SOC` drew 320 mA: the input rail contains the other two
plus memory, I/O and regulator losses. The total is `VDD_IN`; the rest are a
breakdown, kept because "the CPU/GPU rail moved, the SoC rail did not" is
itself a finding about which stages are compute-bound.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from cslr_cost.harness.energy.base import (
    EnergyMeter,
    EnergyReading,
    EnergyUnavailable,
    integrate_trapezoid,
)

HWMON_ROOT = Path("/sys/class/hwmon")

#: The rail whose energy is the reading's total. The others are a breakdown.
TOTAL_RAIL = "VDD_IN"

#: Default sampling interval. See the module docstring for the cost trade-off.
DEFAULT_INTERVAL_S = 0.1


def find_ina3221(root: Path = HWMON_ROOT) -> Optional[Path]:
    """The hwmon directory of the INA3221, located by name, not by number —
    `hwmonN` numbering follows probe order and is not stable across kernels."""
    try:
        candidates = sorted(root.iterdir())
    except OSError:
        return None
    for d in candidates:
        try:
            if (d / "name").read_text().strip() == "ina3221":
                return d
        except OSError:
            continue
    return None


@dataclass
class Ina3221Channel:
    label: str
    index: int
    voltage_path: Path    # in{N}_input, mV
    current_path: Path    # curr{N}_input, mA


def discover_channels(hwmon: Path) -> list[Ina3221Channel]:
    """Labelled channels that have both a voltage and a current input.

    `in7_label` is "sum of shunt voltages" and has no current counterpart, so
    the pairing requirement drops it without special-casing its name.
    """
    channels: list[Ina3221Channel] = []
    for label_file in sorted(hwmon.glob("in*_label")):
        idx = label_file.name[2:-len("_label")]
        if not idx.isdigit():
            continue
        v, c = hwmon / f"in{idx}_input", hwmon / f"curr{idx}_input"
        if not (v.exists() and c.exists()):
            continue
        try:
            label = label_file.read_text().strip()
        except OSError:
            continue
        channels.append(Ina3221Channel(label, int(idx), v, c))
    return channels


class TegraIna3221Meter(EnergyMeter):
    """Module energy on Jetson, integrated from sampled INA3221 power."""

    backend = "tegra_ina3221"

    def __init__(
        self,
        hwmon: Optional[Path] = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        rails: Optional[Sequence[str]] = None,
    ) -> None:
        hwmon = hwmon or find_ina3221()
        if hwmon is None:
            raise FileNotFoundError("no ina3221 hwmon device found")
        channels = discover_channels(hwmon)
        if rails is not None:
            channels = [c for c in channels if c.label in rails]
        if not any(c.label == TOTAL_RAIL for c in channels):
            raise FileNotFoundError(
                f"ina3221 at {hwmon} has no {TOTAL_RAIL} channel "
                f"(found {[c.label for c in channels]}); without the input rail "
                "there is no total, and summing the sub-rails would undercount"
            )
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")

        self.hwmon = hwmon
        self.channels = channels
        self.interval_s = interval_s
        self._fds: dict[str, tuple[int, int]] = {}
        self._samples: list[tuple[float, dict[str, float], int]] = []
        self._read_cost_s = 0.0
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t0 = 0.0

        self._open()
        self._read()   # fails here, at construction, if a rail is unreadable

    # -- capability --------------------------------------------------------

    @staticmethod
    def available() -> bool:
        try:
            m = TegraIna3221Meter()
        except (OSError, ValueError):
            return False
        m.close()
        return True

    @property
    def rail_labels(self) -> list[str]:
        return [c.label for c in self.channels]

    # -- raw reads ---------------------------------------------------------

    def _open(self) -> None:
        # Persistent descriptors + pread: ~10% cheaper than open/read/close per
        # sample, and the sampler runs inside the measured window.
        for c in self.channels:
            self._fds[c.label] = (
                os.open(c.voltage_path, os.O_RDONLY),
                os.open(c.current_path, os.O_RDONLY),
            )

    def close(self) -> None:
        self._halt_thread()
        for vfd, cfd in self._fds.values():
            for fd in (vfd, cfd):
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._fds = {}

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown order
        try:
            self.close()
        except Exception:
            pass

    def _read(self) -> tuple[float, dict[str, float], int]:
        """(timestamp, watts per rail, raw VDD_IN mA). Timestamp is the midpoint
        of the reads, since six I2C transactions take ~3 ms end to end."""
        t0 = time.perf_counter()
        watts: dict[str, float] = {}
        total_ma = 0
        for label, (vfd, cfd) in self._fds.items():
            mv = int(os.pread(vfd, 32, 0))
            ma = int(os.pread(cfd, 32, 0))
            watts[label] = mv * ma / 1e6
            if label == TOTAL_RAIL:
                total_ma = ma
        t1 = time.perf_counter()
        self._read_cost_s += t1 - t0
        return (t0 + t1) / 2.0, watts, total_ma

    # -- window ------------------------------------------------------------

    def _sample_once(self) -> bool:
        try:
            s = self._read()
        except (OSError, ValueError) as exc:
            self._error = f"ina3221 read failed mid-window: {exc.__class__.__name__}: {exc}"
            return False
        with self._lock:
            self._samples.append(s)
        return True

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            if not self._sample_once():
                return

    def start(self) -> None:
        self._halt_thread()
        self._samples = []
        self._read_cost_s = 0.0
        self._error = None
        self._stop.clear()
        self._t0 = time.perf_counter()
        # Sampled synchronously at both edges, so the integral spans the window
        # exactly rather than starting one interval late.
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="ina3221", daemon=True)
        self._thread.start()

    def _halt_thread(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 5 * self.interval_s))
            self._thread = None

    def stop(self) -> EnergyReading:
        self._halt_thread()
        self._sample_once()
        elapsed = time.perf_counter() - self._t0

        with self._lock:
            samples = list(self._samples)

        if self._error:
            raise EnergyUnavailable(self._error)
        if len(samples) < 2:
            # Never integrate to zero. A window this short has no energy figure.
            raise EnergyUnavailable(
                f"ina3221: {len(samples)} sample(s) in a {elapsed:.3f} s window; "
                "at least 2 are needed to integrate"
            )

        per_rail = {
            label: integrate_trapezoid([(t, w[label]) for t, w, _ in samples])
            for label in self.rail_labels
        }

        changes = [samples[i][0] for i in range(1, len(samples))
                   if samples[i][2] != samples[i - 1][2]]
        refresh = (
            (changes[-1] - changes[0]) / (len(changes) - 1) if len(changes) >= 2 else None
        )
        span = samples[-1][0] - samples[0][0]

        return EnergyReading(
            joules=per_rail[TOTAL_RAIL],
            seconds=elapsed,
            backend=self.backend,
            per_rail_joules=per_rail,
            idle_watts=self.idle_watts,
            detail={
                "total_rail": TOTAL_RAIL,
                "n_samples": len(samples),
                "mean_interval_s": span / (len(samples) - 1),
                "observed_refresh_s": refresh,
                "read_cost_s_per_s": self._read_cost_s / elapsed if elapsed > 0 else None,
            },
        )

    def __repr__(self) -> str:
        return f"<TegraIna3221Meter {self.hwmon.name} rails={self.rail_labels} every {self.interval_s}s>"

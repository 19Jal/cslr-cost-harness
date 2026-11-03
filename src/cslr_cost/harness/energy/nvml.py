"""nvml.py — NVIDIA GPU energy, via NVML.

Two paths, and the difference matters:

**Preferred — the cumulative energy counter.**
`nvmlDeviceGetTotalEnergyConsumption` returns millijoules accumulated since the
driver loaded, on Volta and later. A window's energy is `end - start`: exact,
accumulated in hardware, and immune to the sampling problem P1 describes. The
RTX 3060 supports it.

**The Orin does not, despite answering.** On Tegra, NVML initialises, returns a
handle and the name `Orin (nvgpu)`, and then reports NOT_SUPPORTED for both the
energy counter and instantaneous power (L4T R39.2.1). `available()` therefore
requires one of the two power queries to succeed; a handle alone is not a rail.
Before that check, the Orin got a sampled meter whose sampler died on its first
call and integrated nothing -- 0.0 J, silently.

**Fallback — sampling instantaneous power.**
On older parts only `nvmlDeviceGetPowerUsage` exists, giving milliwatts *now*.
That has to be sampled in a background thread and integrated, which is
approximate: NVML updates power on its own cadence (tens of ms), so short
transients are missed and the result depends on sample rate. Readings from this
path are marked `nvml-sampled` rather than `nvml`, so the two are never mixed
silently in a results table.

The Jetson's integrated GPU is not on this rail — Orin exposes its own INA3221
rails through sysfs; see `energy/tegra.py`.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from cslr_cost.harness.energy.base import (
    EnergyMeter,
    EnergyReading,
    EnergyUnavailable,
    integrate_trapezoid,
)


def _nvml():
    import pynvml

    return pynvml


class NvmlMeter(EnergyMeter):
    """GPU energy, preferring the hardware counter over sampled power."""

    backend = "nvml"

    def __init__(self, device_index: int = 0, sample_hz: float = 100.0) -> None:
        self.pynvml = _nvml()
        self.pynvml.nvmlInit()
        self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device_index = device_index

        self._has_counter = self._probe_counter()
        if not self._has_counter:
            self.backend = "nvml-sampled"

        self._sample_interval = 1.0 / sample_hz
        self._samples: list[tuple[float, float]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._start_mj = 0
        self._t0 = 0.0

    # -- capability probing --------------------------------------------------

    def _probe_counter(self) -> bool:
        try:
            self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return True
        except Exception:
            return False

    @staticmethod
    def available() -> bool:
        """True only if this device reports energy or power.

        A handle is not enough: Tegra hands one out and then refuses every
        power query, which previously produced a meter that read 0.0 J.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return False
        for query in ("nvmlDeviceGetTotalEnergyConsumption", "nvmlDeviceGetPowerUsage"):
            try:
                getattr(pynvml, query)(handle)
                return True
            except Exception:
                continue
        return False

    @property
    def uses_hardware_counter(self) -> bool:
        return self._has_counter

    def device_name(self) -> str:
        name = self.pynvml.nvmlDeviceGetName(self.handle)
        return name.decode() if isinstance(name, bytes) else name

    # -- measurement ---------------------------------------------------------

    def start(self) -> None:
        self._t0 = time.perf_counter()
        if self._has_counter:
            self._start_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return

        self._samples = []
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
            except Exception:
                break
            self._samples.append((time.perf_counter(), mw / 1000.0))
            time.sleep(self._sample_interval)

    def stop(self) -> EnergyReading:
        elapsed = time.perf_counter() - self._t0

        if self._has_counter:
            delta_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - self._start_mj
            joules = delta_mj / 1000.0
        else:
            self._stop_flag.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            if len(self._samples) < 2:
                # An empty sample list integrates to 0.0, and a zero is a claim.
                raise EnergyUnavailable(
                    f"nvml-sampled: {len(self._samples)} power sample(s) in "
                    f"{elapsed:.3f} s -- the device stopped answering power queries "
                    "or the window was too short to integrate"
                )
            joules = self._integrate(self._samples)

        return EnergyReading(
            joules=joules,
            seconds=elapsed,
            backend=self.backend,
            per_rail_joules={f"gpu{self.device_index}": joules},
            idle_watts=self.idle_watts,
        )

    @staticmethod
    def _integrate(samples: list[tuple[float, float]]) -> float:
        """Trapezoidal integration of sampled power; see `integrate_trapezoid`."""
        return integrate_trapezoid(samples)

    def __repr__(self) -> str:
        mode = "counter" if self._has_counter else "sampled"
        return f"<NvmlMeter gpu{self.device_index} {mode}>"

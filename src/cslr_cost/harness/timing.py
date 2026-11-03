"""timing.py — measuring how long a stage takes.

Two clocks, never mixed:

**CPU stages** — `time.perf_counter_ns()`. Monotonic, nanosecond resolution.

**GPU stages** — CUDA events with an explicit synchronise. This is not
fussiness. CUDA is asynchronous: a kernel launch returns to the host almost
immediately, long before the GPU has finished. Timing a GPU stage with a wall
clock therefore measures *launch* time, not *execution* time, and reports
numbers that are implausibly fast and wrong in a direction that flatters the
result. A pose stage that really takes 8 ms will happily report 0.2 ms.

Statistics are **median and p95, never mean** (P5). A mean hides the tail, and
the tail is what a user feels — and on a thermally throttling edge device the
tail is also where throttling first shows up.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

try:  # torch is present in the container; keep the module importable without it
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class Durations:
    """Per-iteration timings, in milliseconds.

    Keeps every sample: a 60 s isolation run at 8 ms/iteration is ~7500 floats,
    which costs nothing and means percentiles are exact rather than estimated
    from a running approximation.
    """

    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples_ms.append(ms)

    def __len__(self) -> int:
        return len(self.samples_ms)

    @property
    def n(self) -> int:
        return len(self.samples_ms)

    @property
    def total_ms(self) -> float:
        return sum(self.samples_ms)

    @property
    def median(self) -> Optional[float]:
        return statistics.median(self.samples_ms) if self.samples_ms else None

    @property
    def p95(self) -> Optional[float]:
        """95th percentile, nearest-rank.

        Nearest-rank rather than an interpolating estimator so the reported
        value is an observed measurement, not a synthetic point between two.
        """
        if not self.samples_ms:
            return None
        ordered = sorted(self.samples_ms)
        rank = max(1, int(round(0.95 * len(ordered))))
        return ordered[min(rank - 1, len(ordered) - 1)]

    @property
    def mean(self) -> Optional[float]:
        """Diagnostic only — never reported (P5). A large mean/median gap is a
        useful signal that something is periodically stalling."""
        return statistics.fmean(self.samples_ms) if self.samples_ms else None

    @property
    def minimum(self) -> Optional[float]:
        return min(self.samples_ms) if self.samples_ms else None

    @property
    def maximum(self) -> Optional[float]:
        return max(self.samples_ms) if self.samples_ms else None

    def summary(self) -> dict[str, Optional[float]]:
        return {
            "n": self.n,
            "median_ms": self.median,
            "p95_ms": self.p95,
            "min_ms": self.minimum,
            "max_ms": self.maximum,
            "total_ms": self.total_ms,
        }


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------

class WallTimer:
    """`perf_counter_ns` timer for CPU stages.

        with WallTimer() as t:
            stage.run(x)
        t.elapsed_ms
    """

    __slots__ = ("_t0", "elapsed_ms")

    def __init__(self) -> None:
        self._t0 = 0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "WallTimer":
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter_ns() - self._t0) / 1e6


class CudaTimer:
    """CUDA-event timer for GPU stages.

    Synchronises before starting so queued work from a previous iteration is
    not attributed to this one, and again before reading the elapsed time so
    the events have actually been reached.

    Events are reused across iterations — allocating a pair per iteration in a
    60 s loop is itself measurable.
    """

    __slots__ = ("_start", "_end", "elapsed_ms")

    def __init__(self) -> None:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError(
                "CudaTimer requires CUDA. For a CPU stage use WallTimer, or "
                "make_timer(uses_cuda=False) — do not silently fall back, "
                "because a GPU stage timed on the wall clock reports launch "
                "time rather than execution time."
            )
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms = 0.0

    def __enter__(self) -> "CudaTimer":
        torch.cuda.synchronize()
        self._start.record()
        return self

    def __exit__(self, *exc: object) -> None:
        self._end.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self._start.elapsed_time(self._end)


def make_timer(uses_cuda: bool):
    """The right timer for a stage, from its declared `uses_cuda`."""
    return CudaTimer() if uses_cuda else WallTimer()


def cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """Peak memory over a measurement window.

    `peak_vram_mb` is exact — torch tracks an allocator high-water mark that can
    be reset per window. `peak_ram_mb` is sampled, so it is a floor: a spike
    between samples is missed. Sampling rather than `ru_maxrss` because that is
    a process-lifetime high-water mark and cannot be reset, so it would report
    the largest stage for every subsequent stage.
    """

    peak_ram_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    ram_is_sampled: bool = True
    #: The process's device-memory footprint, peak over the window: what the
    #: GPU actually holds for it -- TensorRT engines and workspaces, cuDNN, the
    #: CUDA context -- which `peak_vram_mb` (PyTorch's allocator) never sees.
    #: Sampled at `DEVICE_MEM_INTERVAL_S`, so a floor, like RAM.
    peak_device_mem_mb: Optional[float] = None
    #: How it was read: "nvmap-process" (Jetson: nvmap's per-client size for
    #: this PID) or "nvml-largest-process" (dev box: NVML's per-process use; in
    #: a container PIDs do not match, so the largest allocation on a GPU the
    #: contention check found otherwise quiet is ours). None: not readable.
    device_mem_source: Optional[str] = None


#: Seconds between device-memory samples. One `sudo cat` (Jetson) or one NVML
#: call per second: negligible next to a measured window of >= 60 s.
DEVICE_MEM_INTERVAL_S = 1.0


def device_footprint_mb() -> tuple[Optional[float], Optional[str]]:
    """This process's device memory now, in MB, and how it was read."""
    import os

    try:
        from cslr_cost.harness.energy.contention import read_nvmap_clients

        clients = read_nvmap_clients(timeout_s=2.0)
        own = [c.size_kb for c in clients if c.pid == os.getpid()]
        return (sum(own) / 1024.0 if own else 0.0), "nvmap-process"
    except Exception:                                 # noqa: BLE001  not a Jetson, or no rule
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(0))
        mems = [p.usedGpuMemory for p in procs if getattr(p, "usedGpuMemory", None)]
        return (max(mems) / (1024 * 1024) if mems else 0.0), "nvml-largest-process"
    except Exception:                                 # noqa: BLE001
        return None, None


class MemoryProbe:
    """Track peak RAM and VRAM across a window.

    RAM is polled by the caller (cheap: one `/proc` read), typically once per
    iteration of an isolation loop, which for a >=60 s window is thousands of
    samples.
    """

    def __init__(self, track_vram: bool = True) -> None:
        import psutil

        self._proc = psutil.Process()
        self._track_vram = track_vram and cuda_available()
        self._peak_ram = 0.0
        self._peak_dev: Optional[float] = None
        self._dev_source: Optional[str] = None
        self._sampler = None
        self._stop_event = None

    def start(self) -> "MemoryProbe":
        import threading

        self._peak_ram = 0.0
        if self._track_vram:
            torch.cuda.reset_peak_memory_stats()
            self._peak_dev, self._dev_source = device_footprint_mb()
            if self._dev_source is not None:
                self._stop_event = threading.Event()
                self._sampler = threading.Thread(target=self._sample_device, daemon=True)
                self._sampler.start()
        self.poll()
        return self

    def _sample_device(self) -> None:
        while not self._stop_event.wait(DEVICE_MEM_INTERVAL_S):
            mb, _ = device_footprint_mb()
            if mb is not None and (self._peak_dev is None or mb > self._peak_dev):
                self._peak_dev = mb

    def poll(self) -> None:
        rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        if rss_mb > self._peak_ram:
            self._peak_ram = rss_mb

    def stop(self) -> MemorySnapshot:
        self.poll()
        vram = None
        if self._track_vram:
            vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if self._sampler is not None:
            self._stop_event.set()
            self._sampler.join(timeout=5.0)
            mb, _ = device_footprint_mb()
            if mb is not None and (self._peak_dev is None or mb > self._peak_dev):
                self._peak_dev = mb
        return MemorySnapshot(
            peak_ram_mb=round(self._peak_ram, 1),
            peak_vram_mb=round(vram, 1) if vram is not None else None,
            peak_device_mem_mb=round(self._peak_dev, 1) if self._peak_dev is not None else None,
            device_mem_source=self._dev_source,
        )

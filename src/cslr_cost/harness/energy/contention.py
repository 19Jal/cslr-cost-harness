"""contention.py — was anything else using this machine while we measured?

The idle bracket in `baseline.py` samples **at the boundaries** of a run. That
catches a tenant who is already there, or one who arrives and stays. It cannot
see one who starts and stops inside the window — and that is the case that
matters most, because it moves the stage measurements without moving either
end of the bracket.

This is not hypothetical here. The four-rung ladder run of 2026-09-08 shared
the box with another account's CUDA container for its entire duration; the
container was created three minutes into the run. Every idle bracket read
within 1.2 W of the settled floor, and isolated `pose` was flat across rungs,
so that particular run survives scrutiny — but nothing in the recorded data
*said* the machine was shared. It had to be reconstructed afterwards from
`docker inspect`, which will not be possible in a month.

--------------------------------------------------------------------------
WHAT IS ACTUALLY MEASURABLE
--------------------------------------------------------------------------
**GPU.** NVML lists the processes holding device memory. Inside a container
those are *host* PIDs while ours is namespaced, so "which of these is me" is
not answerable by matching.

What *is* answerable is whether we hold device memory at all —
`torch.cuda.memory_reserved() > 0`. If we do, one of the listed processes is
us and the rest are company; if we do not, every one of them is company.

The first version of this file assumed the former unconditionally, and its
first live test reported a quiet machine while another account's job sat on
the GPU at 20% utilisation holding 1.9 GB. A detector that fails toward
"clean" is worse than no detector, because it converts an unknown into a
false assurance. So when our own VRAM cannot be determined the probe assumes
we hold **none**, which over-reports contention rather than under-reporting
it.

**GPU on a Jetson.** NVML cannot list processes on Tegra, and other accounts'
`/proc/<pid>/fd` and `maps` are unreadable. What does name them is the nvmap
driver's per-client accounting, `/sys/kernel/debug/nvmap/iovmm/clients` —
verified on the Orin: a TensorRT job appears there by PID, holding 111,604 KB,
while it runs. debugfs is root-only, so the probe reads that one file through a
sudoers rule scoped to exactly `/usr/bin/cat` on it (`docs/devices.md`)
and never runs as root itself. Any client other than this process holding
memory is company; ours is excluded by PID, which works because the harness is
not containerised on the Orin.

Without that rule — or if the read fails partway through — the probe falls
back to total GPU load (`/sys/devices/platform/gpu.0/load`), which includes our
own work and cannot say whose load it is. That report is never `clean`: it
reads PARTIALLY CHECKED, because another account's GPU job cannot be ruled
out, and saying otherwise is the false assurance described above. The same
applies on any machine where NVML itself is missing — previously that reason
was dropped as soon as a CPU sample existed, and the report could read clean
with the GPU never looked at.

**CPU.** `/proc/stat` gives machine-wide busy time; `/proc/self/stat` gives
ours. The difference, over the same interval, is everyone else. This is the
signal that matters for RAPL, since `package-0` is a whole-socket counter and
cannot tell our work from anyone's.

--------------------------------------------------------------------------
JUDGED PER WINDOW, NOT ONLY ON THE WHOLE RUN (blocker 15)
--------------------------------------------------------------------------
A run is several windows -- the opening idle bracket, each stage alone, the
composed pass, the closing bracket -- and energy is integrated per window. The
first version judged foreign CPU on the mean of the whole run, so a burst
confined to the last two of seven 60 s stage windows averaged under the 10%
tolerance while inflating exactly those stages' joules. It happened twice in the
v2 re-grid (`ctc` energy +89% and +48%), both passed as clean, and both were
caught only because J/frame came out non-monotone across rungs.

So the runners `mark()` each window, every sample carries the window it covers
and its interval length, and a run is clean only if **every** window's
time-weighted foreign CPU is under tolerance. A window shorter than
`MIN_JUDGED_WINDOW_S` is not judged on its own -- one 2 s sample catching a cron
job would otherwise condemn it -- but still counts toward the whole-run mean.
The shortest real windows are the idle brackets, 20 s settle + 10 s idle.

--------------------------------------------------------------------------
SAMPLING MUST BE CHEAP, BECAUSE IT IS INSIDE THE WINDOW
--------------------------------------------------------------------------
A probe that costs measurable power corrupts the thing it is checking. Both
reads are a few microseconds of file I/O and one NVML call, rate-limited to
once every couple of seconds regardless of how often `poll()` is invoked — so
a stage running 70,000 iterations in 60 s pays for ~30 samples, not 70,000.

The rate limit lives here rather than in the caller for exactly that reason:
`MemoryProbe` is polled every 64th iteration, which is a wildly different
wall-clock rate for `detect` than for `head`.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: Seconds between samples. Fine enough to catch a job that runs for a few
#: seconds; coarse enough to cost nothing.
DEFAULT_INTERVAL_S = 2.0

#: Fraction of the machine's CPU capacity that may be busy with other work
#: before a run is called contended. Background daemons idle well under this.
DEFAULT_FOREIGN_CPU_TOLERANCE = 0.10

#: A window shorter than this is not judged on its own: a single 2 s sample
#: would decide it. Every measured window in the grid is at least 30 s.
MIN_JUDGED_WINDOW_S = 20.0

#: Jetson total GPU load, per mille. Readable without root; not attributable.
TEGRA_GPU_LOAD = Path("/sys/devices/platform/gpu.0/load")

TEGRA_ATTRIBUTION_REASON = (
    "Jetson: per-process GPU use is only visible through root-only debugfs "
    "(nvmap), so another account's GPU job cannot be ruled out"
)

#: nvmap's per-client GPU memory. Read via `sudo -n` against a sudoers rule that
#: permits this exact command and nothing else; `-n` makes a missing rule fail
#: immediately instead of blocking on a password prompt inside a measurement.
NVMAP_CLIENTS = "/sys/kernel/debug/nvmap/iovmm/clients"
NVMAP_READ_CMD = ("sudo", "-n", "/usr/bin/cat", NVMAP_CLIENTS)

_SIZE_UNITS_KB = {"K": 1, "M": 1024, "G": 1024 * 1024}


@dataclass
class NvmapClient:
    process: str
    pid: int
    size_kb: int


def parse_nvmap_clients(text: str) -> list[NvmapClient]:
    """Rows of `iovmm/clients`, parsed from the right.

        CLIENT                        PROCESS      PID        SIZE
        user                          trtexec  1366348     111604K
        user                   nvargus-daemon     3796          0K
        total                                              111604K

    From the right because a process name may contain spaces; the `total` row
    has no PID and is skipped.
    """
    clients: list[NvmapClient] = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4 or not parts[-2].isdigit():
            continue
        size = parts[-1]
        unit = _SIZE_UNITS_KB.get(size[-1:].upper())
        if unit is None or not size[:-1].isdigit():
            continue
        clients.append(NvmapClient(" ".join(parts[1:-2]), int(parts[-2]), int(size[:-1]) * unit))
    return clients


def read_nvmap_clients(timeout_s: float = 5.0) -> list[NvmapClient]:
    """Current nvmap clients, or an exception saying why they cannot be read."""
    r = subprocess.run(NVMAP_READ_CMD, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise PermissionError((r.stderr or "").strip() or f"exit status {r.returncode}")
    clients = parse_nvmap_clients(r.stdout)
    if not r.stdout.startswith("CLIENT"):
        raise ValueError(f"unexpected nvmap clients format: {r.stdout[:80]!r}")
    return clients


@dataclass
class ContentionSample:
    t: float
    gpu_process_count: int
    gpu_foreign_mem_mb: float
    foreign_cpu_fraction: Optional[float]
    loadavg_1: float
    gpu_utilisation: Optional[int] = None
    #: The window this sample's interval belongs to (`ContentionProbe.mark`),
    #: or None between windows.
    window: Optional[str] = None
    #: Seconds the foreign-CPU fraction was measured over.
    dt: Optional[float] = None


@dataclass
class ContentionReport:
    """What else was on the machine, sampled through the measured window."""

    samples: list[ContentionSample] = field(default_factory=list)
    foreign_cpu_tolerance: float = DEFAULT_FOREIGN_CPU_TOLERANCE
    unavailable_reason: Optional[str] = None
    #: 1 when this process held device memory and so accounts for one of the
    #: listed GPU processes; 0 when it did not, or could not be determined.
    own_gpu_processes: int = 0
    #: Set when GPU processes could not be enumerated (Jetson, or no NVML).
    #: The CPU half of the check still ran; the GPU half did not.
    gpu_attribution_unavailable: Optional[str] = None
    #: How GPU company was attributed when it could be: "nvml" or "nvmap".
    gpu_attribution_method: Optional[str] = None

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def max_gpu_process_count(self) -> int:
        return max((s.gpu_process_count for s in self.samples), default=0)

    @property
    def max_foreign_gpu_processes(self) -> int:
        return max(
            (s.gpu_process_count - self.own_gpu_processes for s in self.samples),
            default=0,
        )

    @property
    def gpu_had_company(self) -> bool:
        """True if any sample saw a GPU process that was not ours."""
        return self.max_foreign_gpu_processes > 0

    @property
    def peak_gpu_utilisation(self) -> Optional[int]:
        vals = [s.gpu_utilisation for s in self.samples if s.gpu_utilisation is not None]
        return max(vals) if vals else None

    @property
    def peak_foreign_gpu_mem_mb(self) -> float:
        return max((s.gpu_foreign_mem_mb for s in self.samples), default=0.0)

    @property
    def peak_foreign_cpu_fraction(self) -> Optional[float]:
        vals = [s.foreign_cpu_fraction for s in self.samples if s.foreign_cpu_fraction is not None]
        return max(vals) if vals else None

    @property
    def mean_foreign_cpu_fraction(self) -> Optional[float]:
        vals = [s.foreign_cpu_fraction for s in self.samples if s.foreign_cpu_fraction is not None]
        return sum(vals) / len(vals) if vals else None

    def window_foreign_cpu(self) -> dict[str, tuple[float, float]]:
        """Window label -> (time-weighted mean foreign CPU, seconds covered)."""
        acc: dict[str, tuple[float, float]] = {}
        for s in self.samples:
            if s.window is None or s.foreign_cpu_fraction is None or not s.dt:
                continue
            w, t = acc.get(s.window, (0.0, 0.0))
            acc[s.window] = (w + s.foreign_cpu_fraction * s.dt, t + s.dt)
        return {k: (w / t, t) for k, (w, t) in acc.items() if t > 0}

    @property
    def contended_windows(self) -> list[tuple[str, float, float]]:
        """(label, mean foreign CPU, seconds) for every judged window over
        tolerance. See "Judged per window" in the module docstring."""
        return [
            (k, m, t) for k, (m, t) in self.window_foreign_cpu().items()
            if t >= MIN_JUDGED_WINDOW_S and m > self.foreign_cpu_tolerance
        ]

    @property
    def n_judged_windows(self) -> int:
        return sum(1 for _m, t in self.window_foreign_cpu().values() if t >= MIN_JUDGED_WINDOW_S)

    @property
    def cpu_had_company(self) -> bool:
        mean = self.mean_foreign_cpu_fraction
        whole = mean is not None and mean > self.foreign_cpu_tolerance
        return whole or bool(self.contended_windows)

    @property
    def clean(self) -> bool:
        """No other GPU process, and foreign CPU below tolerance, throughout.

        `None` samples do not count as clean: a probe that could not read is
        not evidence of a quiet machine.
        """
        if self.unavailable_reason or not self.samples:
            return False
        if self.gpu_attribution_unavailable:
            return False
        return not self.gpu_had_company and not self.cpu_had_company

    def note(self) -> str:
        if self.unavailable_reason:
            return f"contention not checked: {self.unavailable_reason}"
        if not self.samples:
            return "contention not checked: no samples taken"

        cpu = self.mean_foreign_cpu_fraction
        cpu_txt = "n/a" if cpu is None else f"{cpu:.1%} mean"
        if self.gpu_attribution_unavailable and not self.cpu_had_company:
            util = self.peak_gpu_utilisation
            util_txt = "not readable" if util is None else f"peak {util}% (ours included)"
            return (
                f"PARTIALLY CHECKED: foreign CPU {cpu_txt} ({self.n_samples} samples), "
                f"GPU load {util_txt}; GPU not attributed -- "
                f"{self.gpu_attribution_unavailable}"
            )
        if self.clean:
            if self.gpu_attribution_method == "nvmap":
                own = "no other process holding GPU memory (nvmap)"
            else:
                own = "sole GPU process" if self.own_gpu_processes else "no GPU process at all"
            judged = self.n_judged_windows
            win_txt = f", every one of {judged} windows under tolerance" if judged else ""
            return (
                f"machine quiet through the window ({self.n_samples} samples, "
                f"foreign CPU {cpu_txt}{win_txt}, {own})"
            )

        parts = []
        if self.gpu_had_company:
            util = self.peak_gpu_utilisation
            util_txt = "" if util is None else f", peak GPU util {util}%"
            parts.append(
                f"up to {self.max_foreign_gpu_processes} other GPU process(es), "
                f"peak {self.peak_foreign_gpu_mem_mb:.0f} MB{util_txt}"
            )
        if self.gpu_attribution_unavailable:
            parts.append(f"GPU not attributed ({self.gpu_attribution_unavailable})")
        if self.cpu_had_company:
            peak = self.peak_foreign_cpu_fraction
            parts.append(
                f"foreign CPU {cpu_txt}, peak {peak:.1%} "
                f"(tolerance {self.foreign_cpu_tolerance:.0%})"
            )
            hot = self.contended_windows
            if hot:
                parts.append(
                    "over tolerance in window(s) "
                    + ", ".join(f"{k} {m:.1%} over {t:.0f} s" for k, m, t in hot)
                )
        return (
            "SHARED MACHINE DURING THE MEASURED WINDOW: "
            + "; ".join(parts)
            + ". Energy from this run is not defensible — no energy rail "
            "(RAPL's package counter, the Jetson's VDD_IN) can separate our work "
            "from anyone else's, and GPU contention moves latency too. Re-run on "
            "a quiet machine."
        )


class ContentionProbe:
    """Samples other tenants during a measured window. Rate-limited internally."""

    def __init__(
        self,
        interval_s: float = DEFAULT_INTERVAL_S,
        foreign_cpu_tolerance: float = DEFAULT_FOREIGN_CPU_TOLERANCE,
    ) -> None:
        self.interval_s = interval_s
        self.foreign_cpu_tolerance = foreign_cpu_tolerance
        self._samples: list[ContentionSample] = []
        self._last_sample = 0.0
        self._reason: Optional[str] = None
        self._handle = None
        self._n_cpu = os.cpu_count() or 1
        self._prev_total: Optional[float] = None
        self._prev_own: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._own_gpu_processes = 0
        self._gpu_attribution: Optional[str] = None
        self._tegra_load: Optional[Path] = None
        self._window: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "ContentionProbe":
        self._samples = []
        self._last_sample = 0.0
        self._window = None
        self._gpu_attribution = None
        self._tegra_load = None
        self._nvmap = False
        self._method = None
        from cslr_cost.harness.device import is_tegra

        if is_tegra():
            # Tegra NVML answers init and then refuses process queries; asking it
            # would only produce an empty list that reads as "no company".
            self._handle = None
            if os.access(TEGRA_GPU_LOAD, os.R_OK):
                self._tegra_load = TEGRA_GPU_LOAD
            try:
                read_nvmap_clients()
                self._nvmap = True
                self._method = "nvmap"
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self._gpu_attribution = (
                    f"{TEGRA_ATTRIBUTION_REASON} ({exc.__class__.__name__}: "
                    f"{str(exc)[:80]}); install the nvmap sudoers rule in "
                    "docs/devices.md"
                )
        else:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as exc:                      # noqa: BLE001
                self._handle = None
                self._reason = f"NVML unavailable ({exc.__class__.__name__})"
                self._gpu_attribution = self._reason
            else:
                self._method = "nvml"
        # With nvmap, our own PID is excluded when reading, so nothing is
        # subtracted afterwards; the torch-based guess is only for NVML inside a
        # container, where PIDs cannot be matched.
        self._own_gpu_processes = 0 if self._nvmap else self._own_holds_vram()
        self._prev_total, self._prev_own, self._prev_t = self._cpu_totals()
        self.poll(force=True)
        return self

    def poll(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_sample < self.interval_s:
            return
        self._last_sample = now
        self._samples.append(self._sample())

    def mark(self, window: Optional[str]) -> None:
        """Start a named window (None: between windows).

        Forces a sample first, so the interval up to now is credited to the
        window that is ending rather than smeared into the next one. Call it
        outside energy windows: it costs an NVML query.
        """
        self.poll(force=True)
        self._window = window

    def stop(self) -> ContentionReport:
        self.poll(force=True)
        return ContentionReport(
            samples=self._samples,
            foreign_cpu_tolerance=self.foreign_cpu_tolerance,
            unavailable_reason=self._reason if not self._samples else None,
            own_gpu_processes=self._own_gpu_processes,
            gpu_attribution_unavailable=self._gpu_attribution,
            gpu_attribution_method=None if self._gpu_attribution else self._method,
        )

    # -- the two readings --------------------------------------------------

    @staticmethod
    def _own_holds_vram() -> int:
        """1 if this process holds device memory, else 0.

        Fails to 0 deliberately: if we cannot tell, every listed GPU process is
        counted as foreign. Over-reporting contention costs a re-run;
        under-reporting it publishes a contaminated number.
        """
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.memory_reserved() > 0:
                return 1
        except Exception:                             # noqa: BLE001
            pass
        return 0

    def _gpu_utilisation(self) -> Optional[int]:
        if self._tegra_load is not None:
            try:
                return int(self._tegra_load.read_text()) // 10     # per mille -> %
            except (OSError, ValueError):
                return None
        if self._handle is None:
            return None
        try:
            import pynvml

            return int(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        except Exception:                             # noqa: BLE001
            return None

    def _sample(self) -> ContentionSample:
        count, foreign_mb = self._gpu_processes()
        foreign_cpu, dt = self._foreign_cpu_fraction()
        try:
            load1 = os.getloadavg()[0]
        except (OSError, AttributeError):
            load1 = float("nan")
        return ContentionSample(
            t=time.time(),
            gpu_process_count=count,
            gpu_foreign_mem_mb=foreign_mb,
            foreign_cpu_fraction=foreign_cpu,
            loadavg_1=load1,
            gpu_utilisation=self._gpu_utilisation(),
            window=self._window,
            dt=dt,
        )

    def _gpu_processes(self) -> tuple[int, float]:
        """(process count, memory held by processes that are not ours) in MB.

        PIDs are namespaced inside a container, so ours cannot be identified by
        matching. When we do hold device memory, the largest allocation is
        attributed to us — models dominate — and the rest counted as foreign.
        When we hold none, every allocation is foreign.

        On a Jetson with nvmap readable, PIDs *can* be matched, so the count
        returned is foreign clients only, excluding this process by PID.
        """
        if self._nvmap:
            try:
                clients = read_nvmap_clients()
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                # Losing the read mid-run must not leave the samples so far
                # reading as a quiet GPU for the whole window.
                self._nvmap = False
                self._gpu_attribution = (
                    f"nvmap read failed mid-run ({exc.__class__.__name__}: {str(exc)[:80]})"
                )
                return 0, 0.0
            own = os.getpid()
            foreign = [c for c in clients if c.pid != own and c.size_kb > 0]
            return len(foreign), sum(c.size_kb for c in foreign) / 1024.0
        if self._handle is None:
            return 0, 0.0
        try:
            import pynvml

            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        except Exception:                             # noqa: BLE001
            return 0, 0.0

        mems = sorted(
            (p.usedGpuMemory or 0) for p in procs if getattr(p, "usedGpuMemory", None)
        )
        if self._own_gpu_processes and mems:
            foreign_bytes = sum(mems[:-1])
        else:
            foreign_bytes = sum(mems)
        return len(procs), foreign_bytes / (1024 * 1024)

    def _cpu_totals(self) -> tuple[Optional[float], Optional[float], float]:
        """(machine busy seconds, our process seconds, now)."""
        now = time.perf_counter()
        try:
            with open("/proc/stat", "r") as fh:
                parts = fh.readline().split()
            vals = [float(v) for v in parts[1:]]
            ticks = os.sysconf("SC_CLK_TCK")
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)   # idle + iowait
            total_busy = (sum(vals) - idle) / ticks
        except (OSError, IndexError, ValueError):
            return None, None, now

        own = time.process_time()
        return total_busy, own, now

    def _foreign_cpu_fraction(self) -> tuple[Optional[float], Optional[float]]:
        """(foreign fraction of machine CPU, seconds it was measured over)."""
        total, own, now = self._cpu_totals()
        if total is None or self._prev_total is None or self._prev_t is None:
            self._prev_total, self._prev_own, self._prev_t = total, own, now
            return None, None

        dt = now - self._prev_t
        if dt <= 0:
            return None, None

        d_total = total - self._prev_total
        d_own = (own - self._prev_own) if (own is not None and self._prev_own is not None) else 0.0
        self._prev_total, self._prev_own, self._prev_t = total, own, now

        foreign_cpu_seconds = max(0.0, d_total - d_own)
        return min(1.0, foreign_cpu_seconds / (dt * self._n_cpu)), dt

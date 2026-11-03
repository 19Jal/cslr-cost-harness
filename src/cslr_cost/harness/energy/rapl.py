"""rapl.py — Intel RAPL, the CPU-side energy rail.

RAPL (Running Average Power Limit) exposes a monotonically-increasing
microjoule counter per power domain under sysfs:

    intel-rapl:0     package-0   whole CPU package
    intel-rapl:0:0   core        cores only
    intel-rapl:0:1   dram        memory

This rail is not optional for a whole-pipeline account. `decode`, `crop` and `postproc` are
CPU-bound and draw nothing measurable on the GPU rail, so without RAPL three of
the seven stages have no energy figure — and they are exactly the stages the
front-end cost story is about. The `dram` domain is interesting on its own,
since decode is memory-traffic heavy.

**Permissions.** Since CVE-2020-8694 (PLATYPUS) the kernel ships `energy_uj` as
0400 root-only — power traces at fine resolution leak secrets. Reading it
requires the udev rule in `docs/devices.md`, which grants group read
rather than world read. If that has not been applied, `available()` returns
False and the harness records an explicit reason rather than a zero.

**Containers.** Docker mounts its own sysfs over `/sys`, which shadows any bind
mount at a `/sys` subpath: the class symlinks are visible inside the container
but dangle. `scripts/dev.sh` therefore mounts the host tree at `/host/powercap`
and sets `CSLR_RAPL_ROOT`.

**Wraparound is handled.** The counter is finite and wraps at
`max_energy_range_uj`. On this box that is ~262 kJ — hours at typical draw, but
the range is CPU-dependent and some parts wrap in tens of minutes, which is
well inside a sustained 10-minute thermal run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cslr_cost.harness.energy.base import EnergyMeter, EnergyReading

#: Default sysfs location; `scripts/dev.sh` overrides via CSLR_RAPL_ROOT when
#: the tree has been bind-mounted into a container.
DEFAULT_ROOTS = ("/sys/class/powercap", "/sys/devices/virtual/powercap")


def rapl_root() -> Optional[Path]:
    """Locate the powercap tree, honouring CSLR_RAPL_ROOT."""
    env = os.environ.get("CSLR_RAPL_ROOT")
    candidates = (env, *DEFAULT_ROOTS) if env else DEFAULT_ROOTS
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.is_dir() and any(find_domain_dirs(p)):
            return p
    return None


def find_domain_dirs(root: Path) -> list[Path]:
    """Every RAPL domain directory under `root`, at any depth.

    The two layouts differ and both are encountered:

      /sys/class/powercap          flat — intel-rapl:0, intel-rapl:0:0, ... as
                                   symlinks side by side.
      /sys/devices/virtual/powercap
                                   nested — intel-rapl/intel-rapl:0/intel-rapl:0:0.

    The container path is the nested one (Docker's sysfs shadows /sys, so
    scripts/dev.sh bind-mounts the devices tree instead), so searching only the
    top level finds the subsystem directory and no actual domains — which
    presents as "RAPL unavailable" on a machine where it works fine.
    """
    seen: dict[str, Path] = {}
    for p in sorted(root.rglob("intel-rapl:*")):
        if (p / "energy_uj").exists() and p.name not in seen:
            seen[p.name] = p
    return list(seen.values())


class RaplDomain:
    """One power domain's counter, with wraparound handling."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = (path / "name").read_text().strip()
        self._energy = path / "energy_uj"
        self.max_uj = int((path / "max_energy_range_uj").read_text())
        self._last: Optional[int] = None

    def read_uj(self) -> int:
        return int(self._energy.read_text())

    def delta_uj(self, start_uj: int, end_uj: int) -> int:
        """End minus start, correcting for counter wraparound.

        A wrap presents as end < start. Adding the counter's range recovers the
        true delta, provided the window did not span more than one wrap — which
        would be undetectable from two samples alone, and is why windows are
        bounded (P3's 10-minute sustained run, not hours).
        """
        if end_uj >= start_uj:
            return end_uj - start_uj
        return (self.max_uj - start_uj) + end_uj

    def readable(self) -> bool:
        try:
            self.read_uj()
            return True
        except OSError:
            return False


class RaplMeter(EnergyMeter):
    """CPU package/core/dram energy from RAPL counters."""

    backend = "rapl"

    def __init__(self, root: Optional[Path] = None) -> None:
        root = root or rapl_root()
        if root is None:
            raise FileNotFoundError("no RAPL powercap tree found")

        self.domains: list[RaplDomain] = []
        for path in find_domain_dirs(root):
            try:
                d = RaplDomain(path)
            except OSError:
                continue
            if d.readable():
                self.domains.append(d)

        if not self.domains:
            raise PermissionError(
                "RAPL counters are present but not readable. energy_uj is "
                "0400 root-only since CVE-2020-8694 (PLATYPUS); apply the udev "
                "rule in docs/devices.md."
            )

        self._start_uj: dict[str, int] = {}
        self._t0 = 0.0

    @staticmethod
    def available() -> bool:
        try:
            RaplMeter()
            return True
        except (OSError, PermissionError):
            return False

    def start(self) -> None:
        import time

        self._start_uj = {d.name: d.read_uj() for d in self.domains}
        self._t0 = time.perf_counter()

    def stop(self) -> EnergyReading:
        import time

        elapsed = time.perf_counter() - self._t0
        per_rail: dict[str, float] = {}
        for d in self.domains:
            delta = d.delta_uj(self._start_uj[d.name], d.read_uj())
            per_rail[d.name] = delta / 1e6

        # `package-0` already contains core and dram on Intel client parts, so
        # summing every domain would double-count. Prefer the package total and
        # keep the sub-domains as a breakdown.
        package = next(
            (v for k, v in per_rail.items() if k.startswith("package")), None
        )
        total = package if package is not None else sum(per_rail.values())

        return EnergyReading(
            joules=total,
            seconds=elapsed,
            backend=self.backend,
            per_rail_joules=per_rail,
            idle_watts=self.idle_watts,
        )

    def __repr__(self) -> str:
        return f"<RaplMeter domains={[d.name for d in self.domains]}>"

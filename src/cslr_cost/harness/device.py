"""device.py — which device this is, what power mode it is in, and whether its
clocks are pinned.

    "On Jetson, set a power mode and pin clocks (`jetson_clocks`) so runs are
     comparable. Record the mode in every row."

Both halves of that used to be typed by hand: `--device rtx3060` and
`--power-mode default` were CLI defaults, so an Orin run that forgot the flags
would have written rows claiming to be a desktop GPU. Everything here is read
from the board instead, and none of it needs root.

--------------------------------------------------------------------------
WHAT "PINNED" MEANS, AND WHY IT IS CHECKED RATHER THAN TRUSTED
--------------------------------------------------------------------------
`jetson_clocks` pins by raising every DVFS minimum to its maximum. The
governors stay `schedutil` / `nvhost_podgov` — they simply have nowhere to go —
so the governor name says nothing, and the check is `min_freq == max_freq` on
every CPU policy and on the GPU devfreq node.

It has to be checked at run time because pinning does **not** survive a
reboot, while the `nvpmodel` mode does. A board rebooted overnight comes back
in the right power mode with dynamic clocks, and nothing in a row would say so.
Measured on the Orin Nano Super at 15W: unpinned, the CPUs wandered between
729 MHz and 1.34 GHz at idle and the GPU sat at 306 MHz; pinned, all six CPUs
read 1497600 kHz and the GPU 612 MHz, and idle `VDD_IN` rose from ~3.1 W to
~3.9 W. That 0.8 W is inside every over-idle figure, which is exactly why the
two must never be mixed.

The EMC (memory) clock is also pinned by `jetson_clocks`, but its rate is only
readable through debugfs, which is root-only. It is not checked.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEVICE_TREE = Path("/proc/device-tree")
NVPMODEL_STATUS = Path("/var/lib/nvpmodel/status")
NVPMODEL_CONF = Path("/etc/nvpmodel.conf")
SYS_ROOT = Path("/sys")
MEMINFO = Path("/proc/meminfo")

_POWER_MODEL = re.compile(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=([^\s>]+)\s*>")


def is_tegra(device_tree: Path = DEVICE_TREE) -> bool:
    """True on a Jetson. Probes the device tree rather than /etc/nv_tegra_release,
    which a container does not inherit but /proc does."""
    try:
        return b"nvidia,tegra" in (device_tree / "compatible").read_bytes()
    except OSError:
        return False


def board_model(device_tree: Path = DEVICE_TREE) -> Optional[str]:
    try:
        return (device_tree / "model").read_bytes().rstrip(b"\0").decode().strip()
    except OSError:
        return None


def _memory_gb(meminfo: Path = MEMINFO) -> Optional[int]:
    """Marketed memory size: MemTotal rounded up to a power of two.

    MemTotal excludes firmware carve-outs — the 8 GB Orin Nano reports 7.3 GiB —
    so plain rounding would name it a 7 GB part.
    """
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                gib = int(line.split()[1]) / (1024 * 1024)
                return 2 ** math.ceil(math.log2(gib))
    except (OSError, ValueError, IndexError):
        pass
    return None


def device_slug(
    device_tree: Path = DEVICE_TREE, meminfo: Path = MEMINFO
) -> Optional[str]:
    """`orin-nano-8gb-super` style identifier for the results row, or None.

    `-super` is kept because the Super firmware changes the power-mode table
    itself (15W / 25W / MAXN_SUPER), so the same mode name does not mean the
    same caps on a non-Super board.
    """
    model = board_model(device_tree)
    if not model:
        return None
    m = re.search(r"\b(Orin|Xavier|Thor)\s+(Nano|NX|AGX)?", model, re.IGNORECASE)
    if not m:
        return None
    parts = [m.group(1).lower()]
    if m.group(2):
        parts.append(m.group(2).lower())
    gb = _memory_gb(meminfo)
    if gb:
        parts.append(f"{gb}gb")
    if re.search(r"\bSuper\b", model):
        parts.append("super")
    return "-".join(parts)


@dataclass
class PowerMode:
    id: int
    name: str


def nvpmodel_mode(
    status: Path = NVPMODEL_STATUS, conf: Path = NVPMODEL_CONF
) -> Optional[PowerMode]:
    """The active `nvpmodel` mode, read without root.

    `nvpmodel -q` needs sudo; its status file and the mode table do not.
    """
    try:
        m = re.search(r"pmode:(\d+)", status.read_text())
        table = conf.read_text()
    except OSError:
        return None
    if not m:
        return None
    mode_id = int(m.group(1))
    for mid, name in _POWER_MODEL.findall(table):
        if int(mid) == mode_id:
            return PowerMode(id=mode_id, name=name)
    return PowerMode(id=mode_id, name=f"mode{mode_id}")


@dataclass
class ClockState:
    pinned: bool
    detail: str


def clock_state(sys_root: Path = SYS_ROOT) -> ClockState:
    """Whether every CPU and the GPU are held at a single frequency.

    Unreadable counts as **not pinned**: a check that cannot run must not look
    like one that passed.
    """
    unpinned: list[str] = []
    checked = 0

    cpu_dirs = sorted((sys_root / "devices/system/cpu").glob("cpu[0-9]*/cpufreq"))
    for d in cpu_dirs:
        try:
            lo = int((d / "scaling_min_freq").read_text())
            hi = int((d / "scaling_max_freq").read_text())
        except (OSError, ValueError):
            unpinned.append(f"{d.parent.name} unreadable")
            continue
        checked += 1
        if lo != hi:
            unpinned.append(f"{d.parent.name} {lo}-{hi} kHz")

    gpu_dirs = sorted((sys_root / "class/devfreq").glob("*gpu*"))
    for d in gpu_dirs:
        try:
            lo = int((d / "min_freq").read_text())
            hi = int((d / "max_freq").read_text())
        except (OSError, ValueError):
            unpinned.append(f"gpu {d.name} unreadable")
            continue
        checked += 1
        if lo != hi:
            unpinned.append(f"gpu {lo // 1_000_000}-{hi // 1_000_000} MHz")

    if not cpu_dirs or not gpu_dirs:
        missing = [n for n, ds in (("cpufreq", cpu_dirs), ("gpu devfreq", gpu_dirs)) if not ds]
        return ClockState(False, f"cannot check clocks: no {' or '.join(missing)} nodes")
    if unpinned:
        return ClockState(False, "not pinned: " + ", ".join(unpinned))
    return ClockState(True, f"pinned ({checked} clock domains at min == max)")


def describe_device() -> dict[str, object]:
    """Everything above, for `harness rails` and docs/devices.md."""
    mode = nvpmodel_mode()
    clocks = clock_state()
    return {
        "tegra": is_tegra(),
        "model": board_model(),
        "device": device_slug(),
        "power_mode": None if mode is None else mode.name,
        "clocks": clocks.detail,
    }


__all__ = [
    "ClockState",
    "PowerMode",
    "board_model",
    "clock_state",
    "describe_device",
    "device_slug",
    "is_tegra",
    "nvpmodel_mode",
]

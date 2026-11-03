"""Jetson device identity, clock pinning, and INA3221 energy.

The failure these exist for: on the Orin, before this backend, `auto_meter()`
returned an NVML meter that answered every question and measured nothing —
0.0 J, 0.0 W idle — while the CLI labelled the rows `rtx3060` at power mode
`default`. Every check below is about a board saying something true about
itself or refusing, never a plausible default.

The sysfs trees are built in `tmp_path` so these run anywhere; the tests
marked for a real Jetson skip elsewhere.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cslr_cost.harness.device import (
    clock_state,
    device_slug,
    is_tegra,
    nvpmodel_mode,
)
from cslr_cost.harness.energy import EnergyUnavailable, auto_meter
from cslr_cost.harness.energy.tegra import (
    TOTAL_RAIL,
    TegraIna3221Meter,
    discover_channels,
    find_ina3221,
)

ON_JETSON = is_tegra()
HAVE_INA = TegraIna3221Meter.available()


# ---------------------------------------------------------------------------
# fake sysfs
# ---------------------------------------------------------------------------

def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _device_tree(root: Path, compatible: bytes, model: str) -> Path:
    dt = root / "device-tree"
    dt.mkdir()
    (dt / "compatible").write_bytes(compatible)
    (dt / "model").write_bytes(model.encode() + b"\0")
    return dt


def _meminfo(root: Path, kb: int) -> Path:
    p = root / "meminfo"
    p.write_text(f"MemTotal:        {kb} kB\nMemFree:          1 kB\n")
    return p


def _ina3221(root: Path, rails: dict[str, tuple[int, int]]) -> Path:
    """rails: label -> (mV, mA), on channels 1..N, plus the shunt-sum channel."""
    hw = root / "hwmon" / "hwmon7"
    _write(hw / "name", "ina3221\n")
    for i, (label, (mv, ma)) in enumerate(rails.items(), start=1):
        _write(hw / f"in{i}_label", label + "\n")
        _write(hw / f"in{i}_input", f"{mv}\n")
        _write(hw / f"curr{i}_input", f"{ma}\n")
    _write(hw / "in7_label", "sum of shunt voltages\n")
    _write(hw / "in7_input", "5600\n")
    _write(root / "hwmon" / "hwmon0" / "name", "pwmfan\n")
    return hw


ORIN_RAILS = {"VDD_IN": (5032, 616), "VDD_CPU_GPU_CV": (5032, 112), "VDD_SOC": (5032, 208)}


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_tegra_is_detected_from_the_device_tree(tmp_path):
    dt = _device_tree(tmp_path, b"nvidia,p3767-0005\0nvidia,tegra234\0", "Jetson")
    assert is_tegra(dt)


def test_a_non_jetson_is_not_tegra(tmp_path):
    dt = _device_tree(tmp_path, b"raspberrypi,5-model-b\0brcm,bcm2712\0", "Pi 5")
    assert not is_tegra(dt)
    assert not is_tegra(tmp_path / "absent")


def test_slug_rounds_memory_up_to_the_marketed_size(tmp_path):
    """MemTotal excludes firmware carve-outs: the 8 GB Orin Nano reports 7.3 GiB.
    Plain rounding would publish it as a 7 GB part."""
    dt = _device_tree(
        tmp_path, b"nvidia,tegra234\0",
        "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super",
    )
    assert device_slug(dt, _meminfo(tmp_path, 7_664_640)) == "orin-nano-8gb-super"


def test_slug_omits_super_on_a_non_super_board(tmp_path):
    dt = _device_tree(tmp_path, b"nvidia,tegra234\0", "NVIDIA Jetson Orin Nano Developer Kit")
    assert device_slug(dt, _meminfo(tmp_path, 7_664_640)) == "orin-nano-8gb"


def test_unrecognised_model_gives_no_slug_rather_than_a_guess(tmp_path):
    dt = _device_tree(tmp_path, b"nvidia,tegra234\0", "Some Carrier Board")
    assert device_slug(dt, _meminfo(tmp_path, 7_664_640)) is None


# ---------------------------------------------------------------------------
# power mode
# ---------------------------------------------------------------------------

CONF = """
< POWER_MODEL ID=0 NAME=15W >
CPU_ONLINE CORE_0 1
< POWER_MODEL ID=1 NAME=25W >
CPU_ONLINE CORE_0 1
< POWER_MODEL ID=2 NAME=MAXN_SUPER >
CPU_ONLINE CORE_0 1
< PM_CONFIG DEFAULT=1 >
"""


def test_power_mode_is_named_from_the_mode_table(tmp_path):
    _write(tmp_path / "status", "pmode:0000\n")
    _write(tmp_path / "nvpmodel.conf", CONF)
    mode = nvpmodel_mode(tmp_path / "status", tmp_path / "nvpmodel.conf")
    assert (mode.id, mode.name) == (0, "15W")


def test_power_mode_not_in_table_keeps_its_id(tmp_path):
    _write(tmp_path / "status", "pmode:0007\n")
    _write(tmp_path / "nvpmodel.conf", CONF)
    assert nvpmodel_mode(tmp_path / "status", tmp_path / "nvpmodel.conf").name == "mode7"


def test_unreadable_power_mode_is_none_not_default(tmp_path):
    assert nvpmodel_mode(tmp_path / "nope", tmp_path / "nope.conf") is None


# ---------------------------------------------------------------------------
# clocks
# ---------------------------------------------------------------------------

def _clocks(root: Path, cpu: list[tuple[int, int]], gpu: tuple[int, int]) -> Path:
    for i, (lo, hi) in enumerate(cpu):
        d = root / "devices/system/cpu" / f"cpu{i}" / "cpufreq"
        _write(d / "scaling_min_freq", f"{lo}\n")
        _write(d / "scaling_max_freq", f"{hi}\n")
    g = root / "class/devfreq/17000000.gpu"
    _write(g / "min_freq", f"{gpu[0]}\n")
    _write(g / "max_freq", f"{gpu[1]}\n")
    return root


def test_pinned_means_min_equals_max_everywhere(tmp_path):
    """jetson_clocks leaves the governors as schedutil/nvhost_podgov and raises
    the minimums instead, so the governor name proves nothing."""
    root = _clocks(tmp_path, [(1497600, 1497600)] * 6, (612000000, 612000000))
    state = clock_state(root)
    assert state.pinned, state.detail


def test_one_dynamic_cpu_is_not_pinned(tmp_path):
    root = _clocks(tmp_path, [(1497600, 1497600)] * 5 + [(729600, 1497600)],
                   (612000000, 612000000))
    state = clock_state(root)
    assert not state.pinned
    assert "cpu5" in state.detail


def test_dynamic_gpu_is_not_pinned(tmp_path):
    """The state after a reboot: nvpmodel persists, jetson_clocks does not."""
    root = _clocks(tmp_path, [(1497600, 1497600)] * 6, (306000000, 612000000))
    state = clock_state(root)
    assert not state.pinned
    assert "gpu 306-612 MHz" in state.detail


def test_no_clock_nodes_is_not_pinned(tmp_path):
    """A check that cannot run must not look like one that passed."""
    state = clock_state(tmp_path)
    assert not state.pinned
    assert "cannot check" in state.detail


# ---------------------------------------------------------------------------
# INA3221 discovery and integration
# ---------------------------------------------------------------------------

def test_hwmon_is_found_by_name_not_number(tmp_path):
    _ina3221(tmp_path, ORIN_RAILS)
    assert find_ina3221(tmp_path / "hwmon").name == "hwmon7"


def test_shunt_sum_channel_is_not_mistaken_for_a_rail(tmp_path):
    hw = _ina3221(tmp_path, ORIN_RAILS)
    assert [c.label for c in discover_channels(hw)] == list(ORIN_RAILS)


def test_total_is_vdd_in_not_the_sum_of_rails(tmp_path):
    """VDD_IN contains the other two plus memory, I/O and regulator losses —
    the RAPL package-0 trap again. Summing would overstate by ~50% here."""
    hw = _ina3221(tmp_path, ORIN_RAILS)
    m = TegraIna3221Meter(hwmon=hw, interval_s=0.01)
    m.start()
    time.sleep(0.2)
    r = m.stop()
    m.close()

    assert r.backend == "tegra_ina3221"
    assert r.joules == pytest.approx(r.per_rail_joules[TOTAL_RAIL])
    assert r.watts == pytest.approx(5.032 * 0.616, rel=0.05)
    assert sum(r.per_rail_joules.values()) > 1.4 * r.joules
    assert r.detail["n_samples"] >= 2


def test_a_board_without_vdd_in_is_refused(tmp_path):
    """Without the input rail there is no total; the sub-rails would undercount."""
    hw = _ina3221(tmp_path, {"VDD_CPU_GPU_CV": (5032, 112), "VDD_SOC": (5032, 208)})
    with pytest.raises(FileNotFoundError, match="VDD_IN"):
        TegraIna3221Meter(hwmon=hw)


def test_a_window_too_short_to_integrate_refuses_rather_than_reading_zero(tmp_path, monkeypatch):
    """Two samples are the minimum for a trapezoid. One sample integrates to
    0.0 — the exact silent zero the Orin produced before this backend."""
    hw = _ina3221(tmp_path, ORIN_RAILS)
    m = TegraIna3221Meter(hwmon=hw, interval_s=60.0)
    real_sample = m._sample_once
    calls = {"n": 0}

    def only_first() -> bool:
        calls["n"] += 1
        return real_sample() if calls["n"] == 1 else True

    monkeypatch.setattr(m, "_sample_once", only_first)
    m.start()
    with pytest.raises(EnergyUnavailable, match="at least 2"):
        m.stop()
    m.close()


def test_a_rail_that_stops_reading_mid_window_is_refused(tmp_path):
    hw = _ina3221(tmp_path, ORIN_RAILS)
    m = TegraIna3221Meter(hwmon=hw, interval_s=0.01)
    m.start()
    time.sleep(0.05)
    for vfd, cfd in m._fds.values():      # simulate the device vanishing
        import os
        os.close(cfd)
    time.sleep(0.05)
    with pytest.raises(EnergyUnavailable, match="read failed"):
        m.stop()
    m._fds = {}


def test_sampler_thread_does_not_outlive_the_window(tmp_path):
    import threading

    hw = _ina3221(tmp_path, ORIN_RAILS)
    m = TegraIna3221Meter(hwmon=hw, interval_s=0.01)
    m.start()
    m.stop()
    m.close()
    assert not any(t.name == "ina3221" for t in threading.enumerate())


# ---------------------------------------------------------------------------
# on a real Jetson
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ON_JETSON, reason="not a Jetson")
def test_auto_meter_on_a_jetson_is_the_ina3221_not_nvml():
    """The regression: Tegra NVML hands out a handle, so it looked available."""
    m = auto_meter()
    assert isinstance(m, TegraIna3221Meter), m
    m.close()


@pytest.mark.skipif(not HAVE_INA, reason="no readable INA3221")
def test_ina3221_measures_real_nonzero_power():
    m = TegraIna3221Meter()
    m.start()
    end = time.perf_counter() + 1.5
    while time.perf_counter() < end:
        pass
    r = m.stop()
    m.close()
    assert r.joules > 0
    assert 1.0 < r.watts < 30.0, "an Orin module draws a few watts, capped by its power mode"
    assert set(r.per_rail_joules) >= {"VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"}


@pytest.mark.skipif(not ON_JETSON, reason="not a Jetson")
def test_this_jetson_identifies_itself():
    assert device_slug() is not None
    assert nvpmodel_mode() is not None

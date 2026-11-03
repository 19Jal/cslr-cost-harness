"""Tests for the energy and thermal layers.

The recurring theme: an unmeasurable rail must produce an explicit refusal, not
a zero. A zero in a results table is a claim — "this stage is free" — and a
false one.
"""

from __future__ import annotations

import itertools
import time

import pytest

from cslr_cost.harness.energy import (
    CompositeMeter,
    EnergyUnavailable,
    NullEnergyMeter,
    NvmlMeter,
    RaplMeter,
    TegraIna3221Meter,
    auto_meter,
    describe_rails,
    rapl_root,
)
from cslr_cost.harness.energy.base import EnergyReading
from cslr_cost.harness.energy.rapl import RaplDomain, find_domain_dirs
from cslr_cost.harness.thermal import (
    ThermalTrace,
    any_temperature,
    temperature_source,
    warm_to_plateau,
)

HAVE_RAPL = RaplMeter.available()
HAVE_NVML = NvmlMeter.available()


def burn(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


# ---------------------------------------------------------------------------
# Reading arithmetic
# ---------------------------------------------------------------------------

def test_incremental_is_none_until_idle_is_measured():
    """P2 wants total AND over-idle. Without a baseline the increment is
    unknown — which must read as None, not as equal to the total."""
    r = EnergyReading(joules=100.0, seconds=10.0, backend="test")
    assert r.joules_incremental is None
    assert r.watts == 10.0


def test_incremental_subtracts_idle_over_the_window():
    r = EnergyReading(joules=100.0, seconds=10.0, backend="test", idle_watts=4.0)
    assert r.joules_incremental == pytest.approx(60.0)


def test_per_unit_divides_by_iterations():
    """P1: integrate power over the window; divide by iterations."""
    r = EnergyReading(joules=50.0, seconds=10.0, backend="test", idle_watts=1.0)
    assert r.per_unit(100) == pytest.approx(0.5)
    assert r.per_unit_incremental(100) == pytest.approx(0.4)


def test_per_unit_with_zero_iterations_is_none_not_a_crash():
    r = EnergyReading(joules=50.0, seconds=10.0, backend="test")
    assert r.per_unit(0) is None


# ---------------------------------------------------------------------------
# Unavailability is explicit
# ---------------------------------------------------------------------------

def test_null_meter_refuses_rather_than_returning_zero():
    m = NullEnergyMeter("no rail on this machine")
    m.start()
    with pytest.raises(EnergyUnavailable, match="no rail"):
        m.stop()


def test_null_meter_carries_the_reason_for_the_results_row():
    m = NullEnergyMeter("RAPL present but not readable")
    assert "not readable" in m.reason
    assert m.backend == "unavailable"


def test_auto_meter_always_returns_something_usable():
    m = auto_meter()
    assert m.backend
    if not (HAVE_RAPL or HAVE_NVML or TegraIna3221Meter.available()):
        assert isinstance(m, NullEnergyMeter)


def test_nvml_with_a_handle_but_no_power_queries_is_unavailable(monkeypatch):
    """Tegra NVML: init and a handle succeed, every power query is
    NOT_SUPPORTED. That used to count as available and read 0.0 J."""
    import sys, types

    class NotSupported(Exception):
        pass

    def refuse(*_a, **_k):
        raise NotSupported("NVML return code 3")

    fake = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: object(),
        nvmlDeviceGetTotalEnergyConsumption=refuse,
        nvmlDeviceGetPowerUsage=refuse,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert not NvmlMeter.available()


# ---------------------------------------------------------------------------
# RAPL
# ---------------------------------------------------------------------------

def test_domain_discovery_handles_the_nested_layout(tmp_path):
    """/sys/class/powercap is flat; /sys/devices/virtual/powercap is nested, and
    the container mounts the nested one. Searching only the top level finds the
    subsystem directory and no domains — which presents as 'RAPL unavailable'
    on a machine where it works."""
    nested = tmp_path / "intel-rapl" / "intel-rapl:0" / "intel-rapl:0:0"
    nested.mkdir(parents=True)
    for d, name in ((nested.parent, "package-0"), (nested, "core")):
        (d / "energy_uj").write_text("1000")
        (d / "name").write_text(name)
        (d / "max_energy_range_uj").write_text("262143328850")

    found = {p.name for p in find_domain_dirs(tmp_path)}
    assert found == {"intel-rapl:0", "intel-rapl:0:0"}


def test_wraparound_is_corrected(tmp_path):
    """The counter is finite. A wrap presents as end < start, and treating that
    as a negative delta would silently corrupt a long sustained run."""
    d = tmp_path / "intel-rapl:0"
    d.mkdir()
    (d / "energy_uj").write_text("5")
    (d / "name").write_text("package-0")
    (d / "max_energy_range_uj").write_text("100")

    dom = RaplDomain(d)
    assert dom.delta_uj(10, 40) == 30           # normal
    assert dom.delta_uj(90, 10) == 20           # wrapped: (100-90) + 10


@pytest.mark.skipif(not HAVE_RAPL, reason="RAPL not readable (see docs/devices.md)")
def test_rapl_measures_real_cpu_energy():
    m = RaplMeter()
    m.start()
    burn(0.5)
    r = m.stop()
    assert r.joules > 0
    assert r.watts > 1.0, "a busy CPU core draws more than a watt"
    assert "package-0" in r.per_rail_joules


@pytest.mark.skipif(not HAVE_RAPL, reason="RAPL not readable")
def test_rapl_does_not_double_count_subdomains():
    """package-0 already contains core and dram on Intel client parts, so
    summing every domain would overstate CPU energy."""
    m = RaplMeter()
    m.start()
    burn(0.3)
    r = m.stop()
    if "core" in r.per_rail_joules:
        assert r.joules == pytest.approx(r.per_rail_joules["package-0"])
        assert r.joules >= r.per_rail_joules["core"] * 0.9


# ---------------------------------------------------------------------------
# NVML
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_NVML, reason="no NVIDIA device")
def test_nvml_prefers_the_hardware_counter():
    """Sampling is the fallback, not the design — and the two must be
    distinguishable in the results, hence a different backend name."""
    m = NvmlMeter()
    if m.uses_hardware_counter:
        assert m.backend == "nvml"
    else:
        assert m.backend == "nvml-sampled"


@pytest.mark.skipif(not HAVE_NVML, reason="no NVIDIA device")
def test_nvml_measures_positive_energy():
    m = NvmlMeter()
    m.start()
    burn(0.5)
    r = m.stop()
    assert r.joules > 0 and r.watts > 0


def test_sampled_nvml_with_no_samples_refuses_rather_than_reading_zero(monkeypatch):
    """The Orin failure mode: the sampler dies on its first query, the list
    stays empty, and the empty list used to integrate to 0.0 J."""
    m = NvmlMeter.__new__(NvmlMeter)
    m._has_counter = False
    m._samples = []
    m._thread = None
    m.backend = "nvml-sampled"
    m.device_index = 0
    m.idle_watts = None
    import threading
    m._stop_flag = threading.Event()
    m._t0 = time.perf_counter()
    with pytest.raises(EnergyUnavailable, match="power sample"):
        m.stop()


def test_sampled_integration_uses_trapezoid():
    """Rectangle integration would bias every window the same direction."""
    samples = [(0.0, 10.0), (1.0, 20.0)]          # ramp 10W -> 20W over 1s
    assert NvmlMeter._integrate(samples) == pytest.approx(15.0)
    assert NvmlMeter._integrate([(0.0, 5.0)]) == 0.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (HAVE_RAPL and HAVE_NVML), reason="needs both rails")
def test_composite_sums_rails_and_keeps_the_breakdown():
    """Measuring one rail would attribute zero to whole stages: pose and head
    are GPU work, decode and crop are CPU-bound."""
    m = auto_meter()
    assert isinstance(m, CompositeMeter)
    m.start()
    burn(0.5)
    r = m.stop()
    assert r.joules > 0
    assert any(k.startswith("nvml") for k in r.per_rail_joules)
    assert any(k.startswith("rapl") for k in r.per_rail_joules)
    assert "rapl" in r.backend and "nvml" in r.backend


def test_describe_rails_reports_what_is_actually_readable():
    info = describe_rails()
    assert set(info) == {"nvml", "rapl", "tegra_ina3221"}
    assert (info["rapl"] is not None) == HAVE_RAPL
    assert (info["nvml"] is not None) == HAVE_NVML


# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------

def test_slope_is_least_squares_not_endpoint_difference():
    """A single noisy endpoint must not decide whether the device is settled."""
    t = ThermalTrace()
    for i in range(60):
        t.add(50.0, t=float(i))          # perfectly flat
    t.add(80.0, t=60.0)                  # one spurious spike
    slope = t.slope_c_per_min(window_s=61)
    assert abs(slope) < 40, "an endpoint-difference slope would read ~30 C/min"


def test_rising_temperature_is_not_a_plateau():
    t = ThermalTrace()
    for i in range(60):
        t.add(40.0 + i * 0.2, t=float(i))     # +12 C/min
    assert t.slope_c_per_min(window_s=60) > 5


def test_flat_temperature_is_a_plateau():
    t = ThermalTrace()
    for i in range(60):
        t.add(60.0 + (0.05 if i % 2 else -0.05), t=float(i))
    assert abs(t.slope_c_per_min(window_s=60)) < 0.5


def test_trace_reports_none_when_no_samples():
    t = ThermalTrace()
    assert t.start_c is None and t.end_c is None and t.peak_c is None


def test_warm_to_plateau_reports_when_it_gave_up():
    """Hitting the cap without settling must be visible, or a still-warming
    device gets reported as steady state."""
    temps = itertools.count(40.0, 0.5)   # never settles, never runs out
    trace = warm_to_plateau(
        workload=lambda: None,
        sensor=lambda: next(temps),
        window_s=0.05,
        max_seconds=0.2,
        poll_interval_s=0.0,
        slope_threshold=0.0001,
    )
    assert not trace.plateaued
    assert trace.warmup_s >= 0.2


def test_junction_zone_is_preferred_over_the_first_cpu_zone(tmp_path):
    """On the Orin, cpu-thermal is zone 0 and would win a name match; the
    throttling policy acts on tj-thermal. Unreadable zones (the Orin's cv*
    zones return ENODATA) must be skipped, not crash the protocol."""
    from cslr_cost.harness.thermal import cpu_temperature, junction_temperature

    for i, (ztype, temp) in enumerate(
        [("cpu-thermal", "44000"), ("cv0-thermal", ""), ("gpu-thermal", "51000"),
         ("tj-thermal", "51500")]
    ):
        z = tmp_path / f"thermal_zone{i}"
        z.mkdir()
        (z / "type").write_text(ztype + "\n")
        (z / "temp").write_text(temp)
    assert junction_temperature(tmp_path) == pytest.approx(51.5)
    assert cpu_temperature(tmp_path) == pytest.approx(44.0)


def test_no_junction_zone_off_jetson(tmp_path):
    from cslr_cost.harness.thermal import junction_temperature

    z = tmp_path / "thermal_zone0"
    z.mkdir()
    (z / "type").write_text("x86_pkg_temp\n")
    (z / "temp").write_text("40000")
    assert junction_temperature(tmp_path) is None


def test_temperature_sensor_is_present_on_this_box():
    assert temperature_source() != "unavailable"
    assert any_temperature() > 0

"""Was the machine ours while we measured?

The idle bracket samples the *boundaries* of a run. This samples *through* it,
which is the only way to see a tenant who arrives and leaves inside the window
— the case that moves every stage measurement without moving either end of the
bracket.
"""

from __future__ import annotations

import pytest

from cslr_cost.harness.energy.contention import (
    DEFAULT_FOREIGN_CPU_TOLERANCE,
    ContentionProbe,
    ContentionReport,
    ContentionSample,
)


def _sample(procs=1, foreign_mb=0.0, cpu=0.01, util=5):
    return ContentionSample(
        t=0.0,
        gpu_process_count=procs,
        gpu_foreign_mem_mb=foreign_mb,
        foreign_cpu_fraction=cpu,
        loadavg_1=0.5,
        gpu_utilisation=util,
    )


# -- the failure that mattered ---------------------------------------------


def test_a_process_we_do_not_own_is_company_even_when_we_hold_no_vram():
    """The first version assumed one listed GPU process was always us, and
    reported a quiet machine while another account's job held 1.9 GB at 20%
    utilisation. A detector that fails toward 'clean' is worse than none: it
    converts an unknown into a false assurance."""
    r = ContentionReport(samples=[_sample(procs=1, foreign_mb=1894.0)], own_gpu_processes=0)
    assert r.max_foreign_gpu_processes == 1
    assert r.gpu_had_company
    assert not r.clean


def test_our_own_process_is_not_counted_as_company():
    r = ContentionReport(samples=[_sample(procs=1)], own_gpu_processes=1)
    assert r.max_foreign_gpu_processes == 0
    assert not r.gpu_had_company
    assert r.clean


def test_company_is_detected_alongside_our_own_process():
    r = ContentionReport(samples=[_sample(procs=2, foreign_mb=1146.0)], own_gpu_processes=1)
    assert r.max_foreign_gpu_processes == 1
    assert not r.clean
    assert "1 other GPU process" in r.note()


def test_unknown_own_vram_is_assumed_to_be_none():
    """Over-reporting contention costs a re-run; under-reporting it publishes a
    contaminated number. The asymmetry decides the default."""
    probe = ContentionProbe()
    assert probe._own_gpu_processes == 0


# -- CPU side ---------------------------------------------------------------


def test_foreign_cpu_above_tolerance_is_company():
    """RAPL's package-0 is a whole-socket counter; it cannot tell our work from
    anyone else's, so a busy machine corrupts CPU energy even with the GPU
    idle."""
    busy = [_sample(cpu=0.4), _sample(cpu=0.5)]
    r = ContentionReport(samples=busy, own_gpu_processes=1)
    assert r.cpu_had_company
    assert not r.clean
    assert "foreign CPU" in r.note()


def test_background_daemons_do_not_trip_it():
    quiet = [_sample(cpu=0.02), _sample(cpu=0.03)]
    r = ContentionReport(samples=quiet, own_gpu_processes=1)
    assert not r.cpu_had_company
    assert r.clean


def test_tolerance_is_on_the_mean_not_a_single_spike():
    """One 2-second sample catching a cron job should not condemn a 10-minute
    run; sustained load should."""
    spike = [_sample(cpu=0.02)] * 9 + [_sample(cpu=0.6)]
    r = ContentionReport(samples=spike, own_gpu_processes=1)
    assert r.peak_foreign_cpu_fraction == pytest.approx(0.6)
    assert not r.cpu_had_company


# -- not knowing is not the same as being clean -----------------------------


def test_no_samples_is_not_clean():
    assert not ContentionReport(samples=[]).clean


def test_an_unavailable_probe_is_not_clean():
    r = ContentionReport(samples=[_sample()], unavailable_reason="NVML unavailable")
    assert not r.clean
    assert "not checked" in r.note()


# -- cost -------------------------------------------------------------------


def test_sampling_is_rate_limited_regardless_of_poll_frequency():
    """A probe that costs measurable power corrupts what it is checking.
    `MemoryProbe` is polled every 64th iteration, which is a wildly different
    wall-clock rate for `detect` than for `head`, so the limit lives here."""
    probe = ContentionProbe(interval_s=3600.0)
    probe.start()
    before = len(probe._samples)
    for _ in range(10_000):
        probe.poll()
    assert len(probe._samples) == before


def test_the_report_reaches_the_row_as_a_field_not_a_note():
    """Contaminated rows have to be filterable months later, not findable by
    reading free text."""
    from cslr_cost.harness.schema import ResultRow

    row = ResultRow(
        stage="pose", regime="batch", measurement_mode="isolated", batch_size=1,
        input_resolution="290x360", resampled=True,
        contention="SHARED MACHINE DURING THE MEASURED WINDOW: 1 other GPU process",
        energy_backend="unavailable",
        energy_unavailable_reason="contention test; energy not measured",
    )
    row.validate()
    assert "SHARED MACHINE" in row.to_dict()["contention"]


# -- when the GPU cannot be attributed -------------------------------------


def test_unattributed_gpu_is_never_clean_even_with_a_quiet_cpu():
    """Jetson: per-process GPU use is root-only. A quiet CPU does not rule out
    another account's GPU job, so the report must not say clean."""
    r = ContentionReport(
        samples=[_sample(procs=0, util=None), _sample(procs=0, util=None)],
        gpu_attribution_unavailable="Jetson: per-process GPU use is root-only",
    )
    assert not r.clean
    note = r.note()
    assert note.startswith("PARTIALLY CHECKED")
    assert "SHARED MACHINE" not in note, "nothing was detected; do not claim company"


def test_unattributed_gpu_with_busy_cpu_still_reports_company():
    r = ContentionReport(
        samples=[_sample(procs=0, cpu=0.5), _sample(procs=0, cpu=0.5)],
        gpu_attribution_unavailable="no NVML",
    )
    assert not r.clean
    assert r.note().startswith("SHARED MACHINE")
    assert "GPU not attributed" in r.note()


def test_on_a_jetson_the_probe_reads_gpu_load_and_flags_attribution(monkeypatch, tmp_path):
    from cslr_cost.harness import device
    from cslr_cost.harness.energy import contention

    load = tmp_path / "load"
    load.write_text("982\n")
    monkeypatch.setattr(device, "is_tegra", lambda *a, **k: True)
    monkeypatch.setattr(contention, "TEGRA_GPU_LOAD", load)

    def no_rule(*a, **k):
        raise PermissionError("sudo: a password is required")

    monkeypatch.setattr(contention, "read_nvmap_clients", no_rule)

    report = ContentionProbe(interval_s=0.0).start().stop()
    assert report.gpu_attribution_unavailable
    assert report.peak_gpu_utilisation == 98
    assert not report.clean


def test_missing_nvml_is_not_forgotten_once_cpu_samples_exist(monkeypatch):
    """The old bug: the NVML reason was dropped whenever samples existed, so a
    machine with no NVML at all could report clean without looking at the GPU."""
    import sys
    from cslr_cost.harness import device

    monkeypatch.setattr(device, "is_tegra", lambda *a, **k: False)
    monkeypatch.setitem(sys.modules, "pynvml", None)   # import raises ImportError

    report = ContentionProbe(interval_s=0.0).start().stop()
    assert report.n_samples > 0
    assert report.gpu_attribution_unavailable
    assert not report.clean


# -- nvmap: per-process GPU attribution on a Jetson ------------------------------

# Verbatim from the Orin, captured while a TensorRT job ran as the measuring user.
NVMAP_WHILE_TRTEXEC_RAN = """CLIENT                        PROCESS      PID        SIZE
user                          trtexec  1366348     111604K
user                   nvargus-daemon     3796          0K
total                                              111604K
"""


def test_nvmap_clients_parse_from_the_real_output():
    from cslr_cost.harness.energy.contention import parse_nvmap_clients

    clients = parse_nvmap_clients(NVMAP_WHILE_TRTEXEC_RAN)
    assert [(c.process, c.pid, c.size_kb) for c in clients] == [
        ("trtexec", 1366348, 111604),
        ("nvargus-daemon", 3796, 0),
    ], "the total row must be skipped, and sizes kept in KB"


def _jetson_with_nvmap(monkeypatch, tmp_path, clients_text):
    from cslr_cost.harness import device
    from cslr_cost.harness.energy import contention

    load = tmp_path / "load"
    load.write_text("0\n")
    monkeypatch.setattr(device, "is_tegra", lambda *a, **k: True)
    monkeypatch.setattr(contention, "TEGRA_GPU_LOAD", load)
    monkeypatch.setattr(contention, "read_nvmap_clients",
                        lambda *a, **k: contention.parse_nvmap_clients(clients_text()))
    monkeypatch.setattr(ContentionProbe, "_foreign_cpu_fraction", lambda self: (0.01, 2.0))


def test_our_own_gpu_memory_is_not_company(monkeypatch, tmp_path):
    """The harness itself shows up in nvmap once TensorRT allocates."""
    import os

    own = os.getpid()
    _jetson_with_nvmap(monkeypatch, tmp_path, lambda: (
        "CLIENT PROCESS PID SIZE\n"
        f"user python3 {own} 250000K\n"
        "user nvargus-daemon 3796 0K\n"
    ))
    report = ContentionProbe(interval_s=0.0).start().stop()
    assert report.gpu_attribution_unavailable is None
    assert report.gpu_attribution_method == "nvmap"
    assert report.clean, report.note()
    assert "nvmap" in report.note()


def test_another_process_holding_gpu_memory_is_company(monkeypatch, tmp_path):
    _jetson_with_nvmap(monkeypatch, tmp_path, lambda: NVMAP_WHILE_TRTEXEC_RAN)
    report = ContentionProbe(interval_s=0.0).start().stop()
    assert not report.clean
    assert report.max_foreign_gpu_processes == 1
    assert report.peak_foreign_gpu_mem_mb == pytest.approx(111604 / 1024)
    assert report.note().startswith("SHARED MACHINE")


def test_losing_the_nvmap_read_mid_run_is_not_clean(monkeypatch, tmp_path):
    """A rule removed, or sudo timing out, partway through: the samples taken
    before it must not stand in for the whole window."""
    from cslr_cost.harness.energy import contention

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 2:
            raise PermissionError("sudo: a password is required")
        return "CLIENT PROCESS PID SIZE\nuser nvargus-daemon 3796 0K\n"

    _jetson_with_nvmap(monkeypatch, tmp_path, flaky)
    report = ContentionProbe(interval_s=0.0).start()
    for _ in range(3):
        report.poll(force=True)
    report = report.stop()
    assert not report.clean
    assert "mid-run" in (report.gpu_attribution_unavailable or "")


def test_without_the_sudoers_rule_the_reason_says_how_to_fix_it(monkeypatch, tmp_path):
    from cslr_cost.harness import device
    from cslr_cost.harness.energy import contention

    monkeypatch.setattr(device, "is_tegra", lambda *a, **k: True)
    monkeypatch.setattr(contention, "TEGRA_GPU_LOAD", tmp_path / "absent")

    def no_rule(*a, **k):
        raise PermissionError("sudo: a password is required")

    monkeypatch.setattr(contention, "read_nvmap_clients", no_rule)
    report = ContentionProbe(interval_s=0.0).start().stop()
    assert not report.clean
    assert "docs/devices.md" in report.note()
    assert report.note().startswith("PARTIALLY CHECKED")

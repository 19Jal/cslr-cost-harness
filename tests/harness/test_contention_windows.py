"""Blocker 15: contention judged per window, and a large idle rise voided.

Two v2 cells passed the contention check while contaminated. The verdict was a
mean over the whole isolation phase, so a burst confined to the last stage
windows (`head`, `ctc`) averaged under the 10% tolerance while inflating those
stages' energy; and the idle bracket's rise (18.8 -> 109.5 W, 18.7 -> 51.2 W)
was explained as "our own device settling after a sustained hold", though no
hold preceded either. Both were caught only by non-monotone J/frame. These
tests pin the fix to those numbers.
"""

from __future__ import annotations

import pytest

from cslr_cost.harness.energy import (
    VOID_RISE_PCT,
    IdleBaseline,
    contention_verdict,
    explain_instability,
)
from cslr_cost.harness.energy.contention import (
    DEFAULT_FOREIGN_CPU_TOLERANCE,
    MIN_JUDGED_WINDOW_S,
    ContentionProbe,
    ContentionReport,
    ContentionSample,
)


def _s(cpu, window, dt=2.0, procs=1):
    return ContentionSample(
        t=0.0, gpu_process_count=procs, gpu_foreign_mem_mb=0.0,
        foreign_cpu_fraction=cpu, loadavg_1=0.5, gpu_utilisation=5,
        window=window, dt=dt,
    )


def _window(label, cpu, seconds=60.0, dt=2.0):
    return [_s(cpu, label, dt) for _ in range(int(seconds / dt))]


STAGES = ("decode", "detect", "crop", "pose", "postproc", "head", "ctc")


def _isolation_phase(burst=None, level=0.25, quiet=0.02):
    """Idle brackets plus seven 60 s stage windows, as the grid runs them."""
    samples = _window("idle-open", quiet, 30.0)
    for st in STAGES:
        samples += _window(f"stage:{st}", level if burst and st in burst else quiet)
    samples += _window("composed", quiet)
    samples += _window("idle-close", level if burst else quiet, 30.0)
    return ContentionReport(samples=samples, own_gpu_processes=1)


# -- per-window judging -------------------------------------------------------


def test_a_burst_in_the_last_stage_windows_is_no_longer_averaged_away():
    """The v2 failure: a whole-run mean under tolerance, two windows far over."""
    r = _isolation_phase(burst={"head", "ctc"})
    assert r.mean_foreign_cpu_fraction < DEFAULT_FOREIGN_CPU_TOLERANCE, "the old verdict passed"
    assert not r.clean
    hot = {k for k, _m, _t in r.contended_windows}
    assert hot == {"stage:head", "stage:ctc", "idle-close"}
    note = r.note()
    assert "stage:head" in note and "stage:ctc" in note and "SHARED MACHINE" in note


def test_a_quiet_run_is_clean_and_says_how_many_windows_were_judged():
    r = _isolation_phase()
    assert r.clean
    assert r.n_judged_windows == 10          # 2 idle + 7 stages + composed
    assert "every one of 10 windows under tolerance" in r.note()


def test_one_cron_spike_does_not_condemn_even_the_shortest_window():
    """The mean-not-spike rule survives per-window judging: a 30 s idle
    bracket absorbs one 2 s sample at 90%."""
    spiky = _window("idle-close", 0.02, 28.0) + [_s(0.9, "idle-close")]
    r = ContentionReport(samples=_window("stage:pose", 0.02) + spiky, own_gpu_processes=1)
    assert r.clean


def test_a_window_too_short_to_judge_still_counts_toward_the_whole_run():
    short = [_s(0.9, "blip", dt=1.0)] * 4                  # 4 s, far over tolerance
    assert 4.0 < MIN_JUDGED_WINDOW_S
    r = ContentionReport(samples=_window("stage:pose", 0.02) + short, own_gpu_processes=1)
    assert r.contended_windows == []
    assert r.window_foreign_cpu()["blip"][1] == pytest.approx(4.0)


def test_windows_are_time_weighted_not_sample_weighted():
    """`mark()` forces a sample, so a window can end in a millisecond-long
    interval; it must not weigh as much as a 2 s one."""
    samples = _window("stage:pose", 0.0, 30.0) + [_s(1.0, "stage:pose", dt=0.001)]
    mean, secs = ContentionReport(samples=samples).window_foreign_cpu()["stage:pose"]
    assert mean < 0.001 and secs == pytest.approx(30.001)


def test_samples_without_a_window_are_not_judged_as_one():
    r = ContentionReport(samples=[_s(0.02, None)] * 30, own_gpu_processes=1)
    assert r.window_foreign_cpu() == {} and r.clean


# -- the probe tags its samples ------------------------------------------------


def test_mark_credits_the_interval_so_far_to_the_window_that_is_ending(monkeypatch):
    monkeypatch.setattr(ContentionProbe, "_foreign_cpu_fraction", lambda self: (0.05, 2.0))
    monkeypatch.setattr(ContentionProbe, "_gpu_processes", lambda self: (1, 0.0))
    monkeypatch.setattr(ContentionProbe, "_gpu_utilisation", lambda self: 5)
    p = ContentionProbe().start()            # sample 0: before any window
    p.mark("stage:pose")                     # sample 1 closes the unnamed gap
    p.poll(force=True)                       # sample 2: inside stage:pose
    p.mark(None)                             # sample 3 closes stage:pose
    r = p.stop()                             # sample 4: after it
    assert [s.window for s in r.samples] == [None, None, "stage:pose", "stage:pose", None]
    assert all(s.dt == 2.0 for s in r.samples)


class _Recorder:
    """Stands in for ContentionProbe inside a runner and records the marks."""

    def __init__(self, *a, **k):
        self.marks: list = []

    def start(self):
        return self

    def poll(self, force=False):
        pass

    def mark(self, label):
        self.marks.append(label)

    def stop(self):
        return ContentionReport(samples=[_s(0.01, "x")], own_gpu_processes=1)


def test_the_isolation_sweep_marks_every_stage_and_the_composed_pass(monkeypatch):
    from cslr_cost.harness.fixtures import FixtureStore
    from cslr_cost.harness.runners import isolation
    from cslr_cost.harness.stages import Pipeline, synthetic_pipeline

    rec = _Recorder()
    monkeypatch.setattr(isolation, "ContentionProbe", lambda *a, **k: rec)
    pipe = Pipeline(synthetic_pipeline(costs=dict.fromkeys(STAGES, 0.1)))
    store = FixtureStore.capture(pipe, "clip.mp4", 2)
    isolation.measure_isolation_sweep(
        pipe, store, "clip.mp4", 2, meter=None, idle_seconds=0.0,
        min_seconds=0.02, warmup_seconds=0.0,
    )
    opened = [m for m in rec.marks if m is not None]
    assert opened == [f"stage:{s}" for s in STAGES] + ["composed"]
    assert rec.marks.count(None) == len(opened), "every window is closed"


# -- the idle bracket: void a large rise, explain a fall ----------------------


QUIET = ContentionReport(samples=_window("stage:pose", 0.02), own_gpu_processes=1)


@pytest.mark.parametrize("before,after", [(18.8, 109.5), (18.7, 51.2)])
def test_the_two_v2_contaminations_are_now_void(before, after):
    """The bracket rose +483% and +174% with the probe quiet: not clean."""
    idle = IdleBaseline(watts_before=before, watts_after=after)
    verdict = contention_verdict(QUIET, idle)
    assert verdict != "clean" and "IDLE ROSE WITH NO HOLD" in verdict
    assert "VOID" in explain_instability(idle, QUIET)
    assert "OUR OWN device" not in explain_instability(idle, QUIET)


def test_a_small_rise_is_not_voided_but_is_not_called_settling_after_a_hold():
    """v2's one quiet +21.6% (18.6 -> 22.7 W): plausibly our own warmth.
    Over-idle energy stays withheld by `stable`; the row stays clean."""
    idle = IdleBaseline(watts_before=18.6, watts_after=22.7)
    assert not idle.stable and idle.drift_pct < VOID_RISE_PCT
    assert contention_verdict(QUIET, idle) == "clean"
    note = explain_instability(idle, QUIET)
    assert "still warm from the last pass" in note and "sustained hold" not in note


def test_a_fall_is_our_own_device_settling_never_a_tenant():
    """Five v2 runs, -25% to -80%: the OPENING bracket caught the device hot."""
    idle = IdleBaseline(watts_before=88.6, watts_after=17.9)
    assert contention_verdict(QUIET, idle) == "clean"
    note = explain_instability(idle, QUIET)
    assert "OUR OWN device still settling from the preceding run" in note
    assert "lengthen settle_seconds".lower() in note.lower()


def test_after_a_sustained_hold_a_large_rise_is_expected():
    idle = IdleBaseline(watts_before=66.8, watts_after=148.0)
    assert contention_verdict(QUIET, idle, after_sustained_hold=True) == "clean"
    assert "after a sustained hold" in explain_instability(idle, QUIET, after_sustained_hold=True)


def test_a_shared_machine_is_reported_as_such_whatever_the_bracket_did():
    busy = _isolation_phase(burst={"ctc"}, level=0.9)
    for idle in (None, IdleBaseline(18.0, 18.2), IdleBaseline(18.8, 109.5)):
        v = contention_verdict(busy, idle)
        assert v.startswith("SHARED MACHINE")


def test_no_probe_means_no_verdict_and_no_bracket_means_the_probe_decides():
    assert contention_verdict(None, IdleBaseline(18.8, 109.5)) is None
    assert contention_verdict(QUIET, None) == "clean"
    assert contention_verdict(QUIET, IdleBaseline(18.8, None)) == "clean"

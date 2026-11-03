"""P3's warm-to-plateau then ten-minute hold.

The dev box will barely throttle — it is an actively cooled desktop GPU. The
device this protocol exists for is the power-capped, quietly fan-cooled Orin,
so the value of
exercising it here is that the plateau detection, the decay arithmetic and the
row fields are all proven before they matter.
"""

from __future__ import annotations

import pytest

from cslr_cost.harness.rows import RunContext, row_from_sustained
from cslr_cost.harness.runners.sustained import (
    SUSTAINED_MINUTES,
    THROTTLE_NOTICEABLE_PCT,
    SustainedResult,
    run_sustained,
)
from cslr_cost.harness.stages.base import Pipeline
from cslr_cost.harness.stages.synthetic import SyntheticStage
from cslr_cost.harness.thermal import ThermalTrace
from cslr_cost.harness.timing import Durations, MemorySnapshot

N_FRAMES = 4


def _pipeline(cost_ms=0.5):
    return Pipeline([
        SyntheticStage("decode", "video", "frames", cost_ms),
        SyntheticStage("head", "sequence", "gloss_logits", cost_ms,
                       granularity="clip"),
    ])


def _result(clip_ms, *, n_frames=N_FRAMES, paced=None, plateaued=True):
    d = Durations()
    for ms in clip_ms:
        d.add(ms)
    trace = ThermalTrace(source="nvml_gpu")
    for i, c in enumerate((40.0, 45.0, 50.0)):
        trace.add(c, float(i))
    warm = ThermalTrace(source="nvml_gpu")
    warm.plateaued = plateaued
    warm.warmup_s = 60.0
    total_s = sum(clip_ms) / 1000.0
    k = len(clip_ms) // 5
    fifths = (None, None)
    if k >= 1:
        fifths = (
            k * n_frames / (sum(clip_ms[:k]) / 1000.0),
            k * n_frames / (sum(clip_ms[-k:]) / 1000.0),
        )
    return SustainedResult(
        warmup=warm, trace=trace, clip_durations=d, n_frames=n_frames,
        window_s=total_s, memory=MemorySnapshot(), paced_fps=paced,
        _fifth_fps=fifths,
    )


# -- the arithmetic P3 turns on --------------------------------------------


def test_sustained_is_the_whole_window_not_the_best_clip():
    """'Report sustained throughput, not peak.' They differ exactly when the
    device is throttling, which is the finding."""
    r = _result([100.0] * 9 + [50.0])          # one fast clip among slow ones
    assert r.peak_fps == pytest.approx(N_FRAMES / 0.050)
    assert r.sustained_fps < r.peak_fps
    assert r.throttle_pct > 0


def test_a_steady_device_shows_no_decay():
    r = _result([100.0] * 20)
    assert r.decay_pct == pytest.approx(0.0, abs=1e-6)
    assert r.throttling is False


def test_a_throttling_device_is_flagged():
    """Slowing under its own sustained load: the last fifth is materially
    slower than the first."""
    r = _result([100.0] * 10 + [200.0] * 10)
    assert r.decay_pct > THROTTLE_NOTICEABLE_PCT
    assert r.throttling is True


def test_an_outlier_between_the_fifths_fools_peak_but_not_decay():
    """peak-vs-sustained rests on a single best clip anywhere in the window.
    The fifths split only looks at the ends, so a mid-run spike cannot reach
    it."""
    clips = [100.0] * 20
    clips[10] = 10.0                           # implausible, and mid-window
    r = _result(clips)
    assert r.throttle_pct > 50                 # peak is fooled
    assert abs(r.decay_pct) < 1e-6             # decay never sees it


def test_an_outlier_inside_a_fifth_is_diluted_not_ignored():
    """Averaging over k clips is not immunity — worth being precise about,
    because a 10-minute hold has few enough clips that k is small."""
    clips = [100.0] * 20
    clips[3] = 10.0                            # inside the first fifth (k=4)
    r = _result(clips)
    assert r.throttle_pct > 50
    # One clip 10x fast among four moves the fifth, but by far less than it
    # moves a max-based peak.
    assert 0 < r.decay_pct < r.throttle_pct


# -- honesty about what was actually run ------------------------------------


def test_a_short_hold_says_it_is_not_a_sustained_figure():
    pipe = _pipeline()
    r = run_sustained(pipe, None, N_FRAMES, minutes=0.02,
                      plateau_max_seconds=0.5, check_contention=False,
                      poll_memory=False)
    assert "short of P3's 10" in r.notes


def test_failing_to_plateau_is_reported_not_hidden():
    pipe = _pipeline()
    r = run_sustained(pipe, None, N_FRAMES, minutes=0.02,
                      plateau_max_seconds=0.2, check_contention=False,
                      poll_memory=False)
    if not r.warmup.plateaued:
        assert "NOT plateaued" in r.notes
        assert "not steady-state" in r.notes


def test_minutes_must_be_positive():
    with pytest.raises(ValueError, match="minutes"):
        run_sustained(_pipeline(), None, N_FRAMES, minutes=0)


def test_the_window_is_the_protocols():
    assert SUSTAINED_MINUTES == 10.0


# -- paced holds ask a different question -----------------------------------


def test_a_paced_hold_tracks_overrun_drift_not_throughput():
    """Throughput is fixed by the pacing, so the question becomes whether the
    pipeline still keeps up once hot. A run that keeps up cold and slips warm
    looks identical to one that never slips, over ten seconds."""
    r = _result([100.0] * 20, paced=25.0)
    r.overrun_frames_first_fifth = 0
    r.overrun_frames_last_fifth = 37
    assert r.overrun_drift == 37
    assert "overruns first/last fifth" in r.table()


def test_an_unpaced_hold_has_no_overrun_drift():
    assert _result([100.0] * 20).overrun_drift is None


# -- the row ----------------------------------------------------------------


def test_this_is_the_row_that_fills_the_thermal_fields():
    """Every other measurement is far too short to have a meaningful start and
    end temperature, which is why they are null in every other row."""
    r = _result([100.0] * 20)
    row = row_from_sustained(
        r, ctx=RunContext(device="rtx3060", input_resolution="290x360",
                          resampled=True),
        pipeline=_pipeline(),
    )
    row.validate()
    assert row.sustained_fps_10min == pytest.approx(r.sustained_fps)
    assert row.thermal_c_start == 40.0 and row.thermal_c_end == 50.0
    assert row.warmup_s == pytest.approx(60.0)
    assert "P3 sustained hold" in row.notes


def test_a_paced_hold_row_obeys_the_streaming_regime_rules():
    r = _result([100.0] * 20, paced=25.0)
    row = row_from_sustained(
        r, ctx=RunContext(device="rtx3060", input_resolution="290x360",
                          resampled=True),
        pipeline=_pipeline(),
    )
    row.validate()
    assert row.regime == "streaming_simulated"
    assert row.batch_size == 1 and row.paced_fps == 25.0


# -- reading the two checks together ----------------------------------------


def test_a_drifting_baseline_on_a_quiet_machine_is_not_blamed_on_a_tenant():
    """Both checks now span the idle brackets, so a drift with a quiet probe
    means our own device had not settled — and saying "something outside this
    process changed" over a run where demonstrably nothing did sends you
    hunting for a tenant that is not there."""
    from cslr_cost.harness.energy import (
        ContentionReport,
        IdleBaseline,
        explain_instability,
    )
    from cslr_cost.harness.energy.contention import ContentionSample

    quiet = ContentionReport(
        samples=[ContentionSample(0.0, 1, 0.0, 0.01, 0.5, 5)], own_gpu_processes=1
    )
    drifted = IdleBaseline(watts_before=66.8, watts_after=148.0)
    # These are the first sustained hold's numbers; a rise this large is only
    # "our own device" when a hold preceded it (blocker 15).
    note = explain_instability(drifted, quiet, after_sustained_hold=True)
    assert "OUR OWN device still settling" in note
    assert "lengthen settle_seconds" in note
    assert "Over-idle figures from this run remain unusable" in note


def test_a_drifting_baseline_on_a_shared_machine_still_says_re_run():
    from cslr_cost.harness.energy import (
        ContentionReport,
        IdleBaseline,
        explain_instability,
    )
    from cslr_cost.harness.energy.contention import ContentionSample

    busy = ContentionReport(
        samples=[ContentionSample(0.0, 2, 900.0, 0.4, 4.0, 90)], own_gpu_processes=1
    )
    note = explain_instability(IdleBaseline(66.8, 148.0), busy)
    assert "UNSTABLE" in note and "probe agrees" in note


def test_the_closing_bracket_settles_far_longer_after_a_hold():
    """Ten minutes of load leaves a GPU at boost; 20 s catches it on the way
    down, which is how a 148 W 'idle' got recorded against a 17.2 W floor."""
    from cslr_cost.harness.runners.sustained import POST_HOLD_SETTLE_S

    assert POST_HOLD_SETTLE_S >= 60.0


# -- a hold heats the device on purpose -------------------------------------


def _with_idle(before, after, quiet=True):
    from cslr_cost.harness.energy import ContentionReport, IdleBaseline
    from cslr_cost.harness.energy.contention import ContentionSample

    r = _result([100.0] * 20)
    r.idle = IdleBaseline(watts_before=before, watts_after=after)
    cpu = 0.01 if quiet else 0.5
    r.contention = ContentionReport(
        samples=[ContentionSample(0.0, 1, 0.0, cpu, 0.5, 5)], own_gpu_processes=1
    )
    return r


def test_a_rising_bracket_on_a_quiet_machine_does_not_disqualify_a_hold():
    """Every other runner treats drift as a warning. This one heats the device
    deliberately, so its closing bracket is higher by design."""
    r = _with_idle(18.5, 23.4)
    assert r.idle_drift_is_thermal
    assert r.energy_is_defensible
    assert "expected, since the run heats the device" in r.idle_note()


def test_the_cold_floor_is_the_baseline_because_it_is_reproducible():
    """There is no steady hot idle -- a GPU cools the moment work stops, so a
    post-hold reading measures how long we waited, not the device."""
    r = _with_idle(18.5, 23.4)
    assert r.idle.watts == pytest.approx(18.5)
    assert "reproducible cold floor" in r.idle_note()


def test_a_shared_machine_still_disqualifies_a_hold():
    """No choice of baseline repairs a second tenant."""
    r = _with_idle(18.5, 23.4, quiet=False)
    assert not r.idle_drift_is_thermal
    assert not r.energy_is_defensible


def test_a_falling_bracket_is_not_explained_by_our_own_heating():
    """Cooling toward a floor cannot raise it, so a downward drift needs
    another explanation and keeps the ordinary warning."""
    r = _with_idle(40.0, 18.0)
    assert not r.idle_drift_is_thermal

"""Tests for timing, isolation and reconciliation.

The headline test is `test_reconciliation_closes_on_known_costs`: with synthetic
stages of known cost, the isolated sum must match the composed run. That is
The reconciliation gate, validated where ground truth exists — before real
models make a discrepancy ambiguous between "harness bug" and "real overhead".
"""

from __future__ import annotations

import time

import pytest

from cslr_cost.harness.fixtures import FixtureStore
from cslr_cost.harness.reconcile import reconcile
from cslr_cost.harness.runners import (
    measure_all_stages_isolated,
    measure_pipeline_composed,
    measure_stage_isolated,
)
from cslr_cost.harness.stages import Pipeline, spin_ms, synthetic_pipeline
from cslr_cost.harness.timing import (
    Durations,
    MemoryProbe,
    WallTimer,
    cuda_available,
    make_timer,
)

N_FRAMES = 4

# Short windows keep the suite fast. Real measurements use P1's 60s floor;
# `measure_stage_isolated` annotates the row when the window is below it.
FAST = dict(min_seconds=0.12, warmup_seconds=0.02)


@pytest.fixture
def pipeline() -> Pipeline:
    # Small, distinct costs so a mis-scaled stage is obvious.
    return Pipeline(synthetic_pipeline(costs={
        "decode": 0.5, "detect": 0.4, "crop": 0.2, "pose": 1.0,
        "postproc": 0.1, "head": 2.0, "ctc": 0.5,
    }))


@pytest.fixture
def fixtures(pipeline) -> FixtureStore:
    return FixtureStore.capture(pipeline, "clip.mp4", N_FRAMES)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_percentiles_are_observed_values_not_interpolated():
    d = Durations()
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]:
        d.add(float(v))
    assert d.median == 5.5
    assert d.p95 in d.samples_ms, "p95 must be an observed sample (nearest-rank)"
    assert d.p95 == 100.0


def test_p95_is_never_below_the_median():
    d = Durations()
    for v in range(100):
        d.add(float(v))
    assert d.p95 >= d.median


def test_empty_durations_report_none_not_zero():
    """Zero would read as 'this stage is free'; None reads as 'not measured'."""
    d = Durations()
    assert d.median is None and d.p95 is None


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------

def test_wall_timer_measures_real_elapsed_time():
    with WallTimer() as t:
        spin_ms(15)
    assert 13 <= t.elapsed_ms <= 60


def test_make_timer_picks_wall_for_cpu_stages():
    assert isinstance(make_timer(uses_cuda=False), WallTimer)


def test_cuda_timer_refuses_to_silently_fall_back():
    """A GPU stage timed on the wall clock reports launch time, not execution
    time — fast, wrong, and flattering. Better to fail loudly."""
    from cslr_cost.harness.timing import CudaTimer

    if cuda_available():
        with CudaTimer() as t:
            pass
        assert t.elapsed_ms >= 0
    else:
        with pytest.raises(RuntimeError, match="requires CUDA"):
            CudaTimer()


@pytest.mark.skipif(not cuda_available(), reason="needs CUDA")
def test_cuda_timer_sees_async_work_a_wall_clock_would_miss():
    """The reason CUDA events exist: a launch returns before the GPU finishes."""
    import torch

    a = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()

    t0 = time.perf_counter_ns()
    for _ in range(50):
        a @ a
    naive_ms = (time.perf_counter_ns() - t0) / 1e6   # launches only
    torch.cuda.synchronize()

    from cslr_cost.harness.timing import CudaTimer

    with CudaTimer() as t:
        for _ in range(50):
            a @ a
    assert t.elapsed_ms > naive_ms


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def test_memory_probe_reports_a_positive_peak():
    probe = MemoryProbe(track_vram=False).start()
    blob = bytearray(32 * 1024 * 1024)
    probe.poll()
    snap = probe.stop()
    assert snap.peak_ram_mb > 0
    del blob


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_isolated_measurement_recovers_the_nominal_cost(pipeline, fixtures):
    pose = next(s for s in pipeline if s.name == "pose")
    m = measure_stage_isolated(pose, fixtures, **FAST)
    assert m.iterations > 1
    assert 0.9 <= m.median_ms <= 1.6, f"nominal 1.0 ms, measured {m.median_ms}"


def test_warmup_is_excluded_from_the_window(pipeline, fixtures):
    """Warm-up iterations must not appear as samples, or the one-off cost of a
    cold start lands in the median and the p95 tail."""
    pose = next(s for s in pipeline if s.name == "pose")
    m = measure_stage_isolated(pose, fixtures, min_seconds=0.1, warmup_seconds=0.1)
    assert m.warmup_s >= 0.1
    assert m.iterations == len(m.durations)


def test_short_window_is_flagged_as_invalid_for_energy(pipeline, fixtures):
    """Below P1's 60s floor the timing is fine but the energy integral is not,
    and the row must say so rather than look like a full measurement."""
    pose = next(s for s in pipeline if s.name == "pose")
    m = measure_stage_isolated(pose, fixtures, **FAST)
    assert "60s floor" in m.notes and "NOT for energy" in m.notes


def test_isolation_does_not_run_upstream_stages(pipeline, fixtures):
    before = {s.name: s.n_calls for s in pipeline}
    pose = next(s for s in pipeline if s.name == "pose")
    measure_stage_isolated(pose, fixtures, **FAST)
    for s in pipeline:
        if s.name != "pose":
            assert s.n_calls == before[s.name], f"{s.name} ran during isolated pose"


def test_per_clip_scaling_follows_granularity(pipeline, fixtures):
    m = measure_all_stages_isolated(pipeline, fixtures, **FAST)
    pose, head = m["pose"], m["head"]
    # A clip stage is already per-clip and must not be scaled again.
    assert head.per_clip_ms(N_FRAMES) == pytest.approx(head.median_ms)
    # A low-variance frame stage: the block median and median x n agree.
    assert pose.per_clip_ms(N_FRAMES) == pytest.approx(
        pose.median_ms * N_FRAMES, rel=0.15
    )


def _measurement(samples, granularity="frame"):
    from cslr_cost.harness.runners.isolation import StageMeasurement
    from cslr_cost.harness.timing import MemorySnapshot

    d = Durations()
    for s in samples:
        d.add(s)
    return StageMeasurement(
        name="x", granularity=granularity, measurement_mode="isolated",
        durations=d, memory=MemorySnapshot(), window_s=1.0, warmup_s=0.0,
        iterations=len(samples),
    )


def test_per_clip_uses_clip_sized_block_medians_not_scaled_medians():
    """The composed side's every sample is a clip **total**, and a total
    concentrates around the mean. Pairing it with `median x n_frames` compares
    different quantities — which produced a clean-looking +26.85% residual on
    the paced regime with no physical cause at all."""
    # Four frames per clip. Three of every four frames cost 1 ms; the fourth
    # costs 21 ms — the post-idle frame in a paced regime.
    samples = [1.0, 1.0, 1.0, 21.0] * 8
    m = _measurement(samples)

    assert m.median_ms == pytest.approx(1.0)          # median hides the spike
    assert m.per_clip_ms(4) == pytest.approx(24.0)    # 1+1+1+21, the real total
    assert m.per_clip_ms(4) != pytest.approx(m.median_ms * 4)


def test_block_median_is_robust_to_a_single_outlier():
    """Robust to a hiccup, unlike a mean: one 10-second stall must not decide
    whether the gate passes."""
    samples = [2.0] * 40
    samples[7] = 10_000.0
    m = _measurement(samples)
    assert m.per_clip_ms(4) == pytest.approx(8.0)


def test_it_falls_back_when_there_are_not_two_full_blocks():
    """A clip stage measured at a 5 s cadence rarely has more."""
    m = _measurement([3.0, 3.0, 3.0])
    assert m.per_clip_ms(4) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Reconciliation — the 5% gate
# ---------------------------------------------------------------------------

def test_reconciliation_closes_on_known_costs(pipeline, fixtures):
    """The gate, validated where ground truth is known."""
    isolated = measure_all_stages_isolated(pipeline, fixtures, **FAST)
    composed = measure_pipeline_composed(pipeline, "clip.mp4", N_FRAMES, **FAST)
    r = reconcile(isolated, composed, N_FRAMES)

    assert r.passes, r.table()
    assert abs(r.residual_pct) <= 5.0


def test_reconciliation_flags_a_mis_scaled_stage(pipeline, fixtures):
    """Scaling a per-clip stage as if per-frame is the error that produced a
    -1537% residual during bring-up. It must fail the gate, loudly."""
    isolated = measure_all_stages_isolated(pipeline, fixtures, **FAST)
    composed = measure_pipeline_composed(pipeline, "clip.mp4", N_FRAMES, **FAST)

    isolated["head"].granularity = "frame"      # the bug
    r = reconcile(isolated, composed, n_frames=64)
    assert not r.passes
    assert "FAIL" in r.verdict()


def test_a_stage_without_samples_is_never_treated_as_free(pipeline, fixtures):
    isolated = measure_all_stages_isolated(pipeline, fixtures, **FAST)
    composed = measure_pipeline_composed(pipeline, "clip.mp4", N_FRAMES, **FAST)
    isolated["pose"].durations = Durations()   # no samples
    with pytest.raises(ValueError, match="silently treated as costing zero"):
        reconcile(isolated, composed, N_FRAMES)


def test_shares_are_of_the_isolated_sum_not_the_composed_total(pipeline, fixtures):
    """The residual belongs to the pipeline; distributing it across stages would
    invent precision the measurement does not have."""
    isolated = measure_all_stages_isolated(pipeline, fixtures, **FAST)
    composed = measure_pipeline_composed(pipeline, "clip.mp4", N_FRAMES, **FAST)
    r = reconcile(isolated, composed, N_FRAMES)
    assert sum(r.share(s) for s in r.per_stage_ms) == pytest.approx(100.0)


def test_table_renders_verdict_and_every_stage(pipeline, fixtures):
    isolated = measure_all_stages_isolated(pipeline, fixtures, **FAST)
    composed = measure_pipeline_composed(pipeline, "clip.mp4", N_FRAMES, **FAST)
    table = reconcile(isolated, composed, N_FRAMES).table()
    for stage in ["decode", "detect", "crop", "pose", "postproc", "head", "ctc"]:
        assert stage in table
    assert "residual" in table and ("PASS" in table or "FAIL" in table)

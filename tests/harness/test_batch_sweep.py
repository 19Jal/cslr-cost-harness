"""The batch regime inside an energy window.

Synthetic stages and a constant-power meter, so the arithmetic is checkable:
at P watts, a window of W seconds that completes C clips of N frames must
report P*W/C joules per video and P*W/(C*N) per frame, and nothing else.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from cslr_cost.harness.energy.base import EnergyMeter, EnergyReading
from cslr_cost.harness.rows import RunContext, row_from_batch
from cslr_cost.harness.runners.batch import measure_batch_sweep
from cslr_cost.harness.stages.base import Pipeline
from cslr_cost.harness.stages.synthetic import SyntheticStage

N = 8
WATTS = 50.0


class _ConstantPowerMeter(EnergyMeter):
    backend = "nvml"   # a real backend name, so the row validates

    def __init__(self, watts: float) -> None:
        self.watts = watts
        self.idle_watts: Optional[float] = None
        self._t0 = 0.0

    @staticmethod
    def available() -> bool:
        return True

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> EnergyReading:
        dt = time.perf_counter() - self._t0
        return EnergyReading(joules=self.watts * dt, seconds=dt, backend=self.backend,
                             idle_watts=self.idle_watts)


class _Source:
    def rewind(self) -> None:
        pass


def _pipeline() -> Pipeline:
    return Pipeline([
        SyntheticStage("decode", "video", "frames", 1.0),
        SyntheticStage("pose", "frames", "keypoints", 1.0),
        SyntheticStage("head", "sequence", "gloss_logits", 2.0, granularity="clip"),
    ])


def _ctx() -> RunContext:
    return RunContext(
        device="rtx3060", input_resolution="290x360", resampled=True,
        corpus="phoenix-2014", split=None, clip_id="synthetic", n_clips=1,
        arm="P", device_power_mode="default", source_resolution="210x260",
    )


def _sweep(meter=None, **kw):
    return measure_batch_sweep(
        _pipeline(), _Source(), N, batch_size=4, min_seconds=0.3, warmup_clips=1,
        meter=meter, idle_seconds=0.05, idle_settle_seconds=0.0,
        check_contention=False, **kw,
    )


def test_window_joules_divide_by_clips_completed_and_by_frames():
    sweep = _sweep(_ConstantPowerMeter(WATTS))
    c = sweep.composed
    assert c.iterations == len(sweep.results) >= 2
    assert c.energy is not None

    row = row_from_batch(sweep.last, ctx=_ctx(), pipeline=_pipeline(), sweep=sweep).validate()
    per_video = c.energy.joules / c.iterations
    assert row.joules_per_video == pytest.approx(per_video)
    assert row.joules_per_frame == pytest.approx(per_video / N)
    assert row.iterations == c.iterations
    assert row.batch_size == 4 and row.regime == "batch" and row.paced_fps is None


def test_over_idle_energy_is_present_once_the_bracket_closes():
    sweep = _sweep(_ConstantPowerMeter(WATTS))
    assert sweep.idle is not None
    assert sweep.energy_is_defensible          # constant power: the bracket cannot drift
    row = row_from_batch(sweep.last, ctx=_ctx(), pipeline=_pipeline(), sweep=sweep)
    assert row.idle_watts == pytest.approx(WATTS, rel=0.05)
    # the device draws exactly its idle figure, so over-idle is ~0, not the total
    assert abs(row.joules_per_frame_incremental) < 0.05 * row.joules_per_frame


def test_batch_latency_is_pooled_over_every_clip_in_the_window():
    sweep = _sweep()
    pooled = sweep.batch_durations
    assert pooled.n == sum(r.batch_durations.n for r in sweep.results)
    assert pooled.n == len(sweep.results) * 2          # 8 frames at batch 4
    row = row_from_batch(sweep.last, ctx=_ctx(), pipeline=_pipeline(), sweep=sweep)
    assert row.latency_ms_median == pytest.approx(pooled.median)
    assert row.seconds_per_video == pytest.approx(sweep.composed.median_ms / 1000.0, abs=1e-4)


def test_no_meter_is_a_stated_reason_not_a_zero():
    sweep = _sweep(meter=None)
    assert sweep.composed.energy is None
    assert not sweep.energy_is_defensible
    row = row_from_batch(sweep.last, ctx=_ctx(), pipeline=_pipeline(), sweep=sweep).validate()
    assert row.joules_per_frame is None
    assert row.energy_backend == "unavailable"
    assert row.energy_unavailable_reason


def test_without_a_sweep_the_row_is_unchanged():
    sweep = _sweep()
    row = row_from_batch(sweep.last, ctx=_ctx(), pipeline=_pipeline()).validate()
    assert row.energy_backend == "unavailable"
    assert row.iterations is None
    assert row.latency_ms_median == pytest.approx(sweep.last.batch_durations.median)

"""The two regimes of protocol rule P7, against known-cost synthetic stages.

Stubs rather than real models on purpose: the pacing loop's correctness is a
property of the loop, and with a stage that costs a known 2 ms a mispaced run
is obvious. With a real model a 20% discrepancy is ambiguous between "the
runner is wrong" and "that is genuine overhead".
"""

from __future__ import annotations

import time

import pytest

from cslr_cost.harness.runners.batch import (
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    run_batched,
)
from cslr_cost.harness.runners.streaming import (
    DEFAULT_PACED_FPS,
    LOOKAHEAD_CAUSAL,
    LOOKAHEAD_FULL_CLIP,
    run_streaming,
)
from cslr_cost.harness.stages.base import Pipeline, Stage
from cslr_cost.harness.stages.synthetic import SyntheticStage

FPS = 50.0            # 20 ms budget, so tests stay quick
BUDGET_MS = 1000.0 / FPS
N = 10


def _pipeline(frame_cost_ms: float = 2.0, clip_cost_ms: float = 3.0) -> Pipeline:
    return Pipeline(
        [
            SyntheticStage("decode", "video", "frames", frame_cost_ms),
            SyntheticStage("pose", "frames", "keypoints", frame_cost_ms),
            SyntheticStage("head", "sequence", "gloss_logits", clip_cost_ms,
                           granularity="clip"),
        ]
    )


class _BatchingStage(SyntheticStage):
    """A stage whose batch call costs one unit regardless of batch size."""

    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_calls = 0

    def run_batch(self, units):
        self.batch_calls += 1
        self._n_calls += len(units)
        from cslr_cost.harness.stages.synthetic import spin_ms

        spin_ms(self.cost_ms)
        return list(units)


# -- streaming: P7 rules 1 and 2 -------------------------------------------


def test_batch_size_is_one_by_construction_not_by_argument():
    """P7 rule 1 has no exceptions, so batch size is not a parameter of this
    runner at all -- there is nothing to pass wrongly."""
    pipe = _pipeline()
    result = run_streaming(pipe, None, N, paced_fps=FPS)
    assert result.batch_size == 1
    assert result.regime == "streaming_simulated"
    with pytest.raises(TypeError):
        run_streaming(pipe, None, N, paced_fps=FPS, batch_size=4)


def test_frames_are_actually_paced():
    pipe = _pipeline(frame_cost_ms=1.0)
    start = time.perf_counter()
    result = run_streaming(pipe, None, N, paced_fps=FPS)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    expected = (N - 1) * BUDGET_MS
    assert elapsed_ms >= expected, "the loop ran faster than the pacing allows"
    assert result.pacing_jitter.median < BUDGET_MS / 2


def test_a_fast_pipeline_keeps_up_and_logs_no_overrun():
    result = run_streaming(_pipeline(frame_cost_ms=1.0), None, N, paced_fps=FPS)
    assert result.kept_up
    assert result.overrun_frames == 0
    assert result.overrun_ms_total == 0.0
    assert result.lag_ms_final == pytest.approx(0.0, abs=5.0)


def test_a_slow_pipeline_logs_every_overrun():
    """P7 rule 2: accumulated overrun is a more honest 'can it run live'
    answer than mean fps."""
    per_frame = BUDGET_MS * 0.8          # two stages => 1.6x the budget
    result = run_streaming(_pipeline(frame_cost_ms=per_frame), None, 5, paced_fps=FPS)
    assert not result.kept_up
    assert result.overrun_frames == 5
    assert result.overrun_rate == 1.0
    assert result.overrun_ms_total > 0
    assert result.lag_ms_final > 0, "a pipeline that never keeps up must end behind"


class _SpikyStage(SyntheticStage):
    """Slow on its first call, fast afterwards — an isolated spike."""

    def run(self, x):
        cost = self.cost_ms if self._n_calls == 0 else 0.5
        self._n_calls += 1
        from cslr_cost.harness.stages.synthetic import spin_ms

        spin_ms(cost)
        return x


def test_an_isolated_spike_is_recovered_and_the_two_ledgers_diverge():
    """The reason both ledgers exist. One slow frame among fast ones overruns
    its slot -- so `overrun_ms_total` is nonzero -- but the pacing loop absorbs
    it and the run still ends on schedule. Reading either number alone would
    call this the same as a pipeline that cannot keep up."""
    pipe = Pipeline([_SpikyStage("decode", "video", "frames", BUDGET_MS * 2)])
    result = run_streaming(pipe, None, 8, paced_fps=FPS)
    assert result.overrun_frames == 1
    assert result.overrun_ms_total > BUDGET_MS / 2
    assert result.lag_ms_final < result.overrun_ms_total


def test_a_pipeline_that_never_keeps_up_ends_at_least_as_far_behind():
    """With every frame over budget there is no slack to recover, so the final
    lag is the overrun total plus the loop's own untimed overhead. That excess
    is a check on the runner: it should be small."""
    result = run_streaming(_pipeline(frame_cost_ms=BUDGET_MS * 0.8), None, 5, paced_fps=FPS)
    assert result.overrun_frames == 5
    assert result.lag_ms_final >= result.overrun_ms_total - 1e-6
    overhead_ms = result.lag_ms_final - result.overrun_ms_total
    assert overhead_ms < 5.0, f"harness overhead of {overhead_ms:.2f} ms/run is too high"


def test_the_pacing_gap_is_slept_not_spun():
    """The single most important property of this runner.

    The streaming-minus-batch energy gap is what the regime exists to produce.
    Busy-spinning the wait would draw full CPU power during exactly the
    interval that is meant to be idle, erasing the gap and reporting streaming
    as costing the same as batch.
    """
    pipe = _pipeline(frame_cost_ms=1.0, clip_cost_ms=0.0)
    cpu_start = time.process_time()
    result = run_streaming(pipe, None, N, paced_fps=FPS)
    cpu_used = time.process_time() - cpu_start

    assert result.wall_s > 0.15, "test is too short to distinguish sleep from spin"
    assert cpu_used < result.wall_s * 0.6, (
        f"used {cpu_used:.3f}s CPU over {result.wall_s:.3f}s wall — the pacing "
        "gap is being spun, which would erase the streaming energy gap"
    )
    assert result.idle_s > 0


def test_duty_cycle_reflects_the_idle_fraction():
    result = run_streaming(_pipeline(frame_cost_ms=1.0), None, N, paced_fps=FPS)
    assert 0.0 < result.duty_cycle < 0.5


# -- streaming: P7 rule 3, first-gloss latency -----------------------------


def test_bidirectional_first_gloss_latency_is_the_whole_clip():
    """'If the head is bidirectional, it consumes the whole clip and your
    first-gloss latency is the whole clip. Report that.'"""
    result = run_streaming(_pipeline(), None, N, paced_fps=FPS,
                           head_lookahead=LOOKAHEAD_FULL_CLIP)
    clip_ms = (N - 1) * BUDGET_MS
    assert result.first_gloss_latency_ms >= clip_ms
    assert "bidirectional" in result.first_gloss_latency_note


def test_a_causal_head_does_not_inherit_the_clip_bound():
    result = run_streaming(_pipeline(), None, N, paced_fps=FPS,
                           head_lookahead=LOOKAHEAD_CAUSAL)
    assert result.first_gloss_latency_ms < (N - 1) * BUDGET_MS
    assert result.first_gloss_latency_note is None


def test_pacing_rate_must_be_positive():
    with pytest.raises(ValueError, match="paced_fps"):
        run_streaming(_pipeline(), None, N, paced_fps=0)


def test_default_pacing_matches_corpus_capture_rate():
    assert DEFAULT_PACED_FPS == 25.0


# -- batch ------------------------------------------------------------------


def test_batch_rows_never_carry_pacing():
    """The schema rejects a batch row with paced_fps: a paced batch run is
    neither regime."""
    result = run_batched(_pipeline(), None, N, batch_size=4)
    assert result.regime == "batch"
    assert result.paced_fps is None


def test_batch_size_must_be_at_least_one():
    with pytest.raises(ValueError, match="batch_size"):
        run_batched(_pipeline(), None, N, batch_size=0)


def test_every_frame_is_processed_across_uneven_batches():
    pipe = _pipeline()
    run_batched(pipe, None, 10, batch_size=4)      # 4 + 4 + 2
    for stage in pipe.frame_stages:
        assert stage.n_calls == 10


def test_only_genuinely_batching_stages_are_reported_as_such():
    """A pipeline's batch speed-up usually comes from one stage. Crediting the
    whole pipeline for it is a common way to overstate batch throughput."""
    pipe = Pipeline(
        [
            SyntheticStage("decode", "video", "frames", 1.0),
            _BatchingStage("pose", "frames", "keypoints", 1.0),
        ]
    )
    result = run_batched(pipe, None, 8, batch_size=4)
    assert result.batched_stages == ("pose",)
    assert set(result.stage_durations) == {"decode", "pose"}


def test_a_batching_stage_gets_one_call_per_batch():
    pose = _BatchingStage("pose", "video", "frames", 1.0)
    pipe = Pipeline([pose])
    run_batched(pipe, None, 12, batch_size=4)
    assert pose.batch_calls == 3
    assert pose.n_calls == 12


def test_default_run_batch_loops_so_serial_stages_cost_the_same():
    """`decode` is inherently serial and `crop`'s warp is per-image. Their
    batch cost equalling their streaming cost is a finding, not a gap."""
    stage = SyntheticStage("decode", "video", "frames", 1.0)
    assert stage.supports_batching is False
    out = stage.run_batch([None, None, None])
    assert len(out) == 3 and stage.n_calls == 3


def test_ms_per_frame_is_throughput_not_latency():
    result = run_batched(_pipeline(frame_cost_ms=1.0), None, 8, batch_size=4)
    assert result.ms_per_frame == pytest.approx(result.batch_durations.median / 4)
    assert result.throughput_fps == pytest.approx(1000.0 / result.ms_per_frame)


def test_clip_stages_still_run_once_under_batching():
    """Batching does not change the pipeline's shape: stages 1-5 are per frame,
    head and ctc per clip."""
    pipe = _pipeline()
    head = pipe.clip_stages[0]
    run_batched(pipe, None, 12, batch_size=4)
    assert head.n_calls == 1


def test_protocol_batch_sizes():
    assert BATCH_SIZES == (16, 32, 64) and DEFAULT_BATCH_SIZE in BATCH_SIZES

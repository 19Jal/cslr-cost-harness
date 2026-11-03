"""Isolation under streaming pacing — P1 measured inside P7's regime.

The reason this exists rather than a `paced=True` flag: on this box the same
stage costs ~1.8x more per frame when fed one every 40 ms than when run back to
back. Copying the unpaced numbers into the streaming rows would have understated
a real-time deployment by nearly a factor of two.
"""

from __future__ import annotations

import pytest

from cslr_cost.harness.fixtures import FixtureStore
from cslr_cost.harness.reconcile import reconcile
from cslr_cost.harness.runners.paced import (
    THIN_SAMPLE_COUNT,
    _rewind,
    measure_all_stages_paced,
    measure_stage_paced,
    measure_streaming_composed,
    stage_period_s,
)
from cslr_cost.harness.stages.base import Pipeline
from cslr_cost.harness.stages.synthetic import SyntheticStage

FPS = 100.0          # 10 ms slots, so tests stay quick
N_FRAMES = 8


def _frame_stage(cost_ms=1.0, name="decode"):
    return SyntheticStage(name, "video", "frames", cost_ms)


def _clip_stage(cost_ms=1.0):
    return SyntheticStage("head", "sequence", "gloss_logits", cost_ms,
                          granularity="clip")


def _fixtures(stage, n=N_FRAMES):
    return FixtureStore({stage.input_key: [None] * n
                         if stage.granularity == "frame" else None})


# -- cadence ----------------------------------------------------------------


def test_a_frame_stage_gets_one_slot_per_frame():
    assert stage_period_s(_frame_stage(), 25.0, 128) == pytest.approx(0.04)


def test_a_clip_stage_gets_one_slot_per_clip():
    """head and ctc run once per clip, as a burst after the last frame. Pacing
    them at 25 Hz would measure something no deployment does."""
    assert stage_period_s(_clip_stage(), 25.0, 128) == pytest.approx(5.12)


def test_cadence_rejects_nonsense():
    with pytest.raises(ValueError, match="paced_fps"):
        stage_period_s(_frame_stage(), 0, 128)
    with pytest.raises(ValueError, match="n_frames"):
        stage_period_s(_frame_stage(), 25.0, 0)


# -- what is measured -------------------------------------------------------


def test_the_recorded_value_is_service_time_not_the_slot():
    """At 20-38% duty cycle most of a paced run is sleep. Recording wall time
    would make every stage report its slot width and reconcile to the duty
    cycle instead of to the work."""
    stage = _frame_stage(cost_ms=1.0)
    m = measure_stage_paced(
        stage, _fixtures(stage), paced_fps=FPS, n_frames=N_FRAMES,
        min_seconds=1.0, warmup_seconds=0.2, poll_memory=False,
    )
    assert m.median_ms == pytest.approx(1.0, abs=0.8)
    assert m.median_ms < 1000.0 / FPS, "this looks like slot width, not service time"


def test_calls_actually_happen_at_the_paced_rate():
    stage = _frame_stage(cost_ms=0.5)
    m = measure_stage_paced(
        stage, _fixtures(stage), paced_fps=FPS, n_frames=N_FRAMES,
        min_seconds=1.0, warmup_seconds=0.1, poll_memory=False,
    )
    assert m.iterations == pytest.approx(FPS * 1.0, rel=0.25)


def test_a_stage_over_its_slot_is_recorded_as_such():
    stage = _frame_stage(cost_ms=1000.0 / FPS * 2)      # twice the budget
    m = measure_stage_paced(
        stage, _fixtures(stage), paced_fps=FPS, n_frames=N_FRAMES,
        min_seconds=0.5, warmup_seconds=0.05, poll_memory=False,
    )
    assert "over budget" in m.notes


def test_a_thin_sample_count_is_declared_not_hidden():
    """A clip stage at a 5 s cadence cannot produce many samples in a 60 s
    window. Measuring it unpaced instead would silently drop the ramp it pays
    after seconds of near-idle — the largest such gap in the pipeline."""
    stage = _clip_stage(cost_ms=1.0)
    m = measure_stage_paced(
        stage, FixtureStore({"sequence": object()}),
        paced_fps=FPS, n_frames=100, min_seconds=1.0, warmup_seconds=0.0,
        poll_memory=False,
    )
    assert m.iterations < THIN_SAMPLE_COUNT
    assert "thin" in m.notes and "p95 should not be quoted" in m.notes


def test_the_paced_cadence_is_recorded_on_the_measurement():
    stage = _frame_stage()
    m = measure_stage_paced(
        stage, _fixtures(stage), paced_fps=FPS, n_frames=N_FRAMES,
        min_seconds=0.4, warmup_seconds=0.05, poll_memory=False,
    )
    assert f"paced {FPS:g} fps" in m.notes


# -- the whole protocol -----------------------------------------------------


def _pipeline():
    return Pipeline([
        SyntheticStage("decode", "video", "frames", 0.5),
        SyntheticStage("pose", "frames", "keypoints", 0.5),
        SyntheticStage("head", "sequence", "gloss_logits", 0.5, granularity="clip"),
    ])


def test_every_stage_is_measured_and_reconciles():
    pipe = _pipeline()
    fx = FixtureStore({
        "video": [None] * N_FRAMES,
        "frames": [None] * N_FRAMES,
        "sequence": object(),
    })
    iso = measure_all_stages_paced(
        pipe, fx, paced_fps=FPS, n_frames=N_FRAMES,
        min_seconds=0.6, warmup_seconds=0.05, poll_memory=False,
    )
    assert set(iso) == {"decode", "pose", "head"}

    composed, results = measure_streaming_composed(
        pipe, None, N_FRAMES, paced_fps=FPS, min_seconds=0.4,
        warmup_clips=0, poll_memory=False,
    )
    assert composed.granularity == "clip"
    assert composed.measurement_mode == "composed"
    assert results and results[0].regime == "streaming_simulated"

    rec = reconcile(iso, composed, n_frames=N_FRAMES)
    assert rec.sum_isolated_ms > 0 and rec.composed_ms > 0


def test_the_composed_value_is_a_clips_service_time_not_its_wall_time():
    """A paced clip takes n_frames/fps seconds by construction. Reconciling
    that would compare the isolated sum against the pacing rate."""
    pipe = _pipeline()
    composed, _ = measure_streaming_composed(
        pipe, None, N_FRAMES, paced_fps=FPS, min_seconds=0.4,
        warmup_clips=0, poll_memory=False,
    )
    wall_ms_per_clip = N_FRAMES / FPS * 1000.0
    assert composed.median_ms < wall_ms_per_clip
    assert "service time per clip" in composed.notes


def test_rewind_is_optional_on_the_source():
    """Fixtures and stub sources have no decoder to seek."""
    _rewind(None)
    _rewind(object())

    class Seekable:
        def __init__(self):
            self.n = 0

        def rewind(self):
            self.n += 1

    s = Seekable()
    _rewind(s)
    assert s.n == 1

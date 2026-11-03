"""Tests for the stage interface, composition, and fixture-based isolation.

The exit criterion for this chunk is at the bottom: the seven-stage skeleton
runs composed, is replayed stage-by-stage in isolation, and emits seven
schema-valid rows — all without a single real model.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cslr_cost.harness.fixtures import (
    FixtureStore,
    MissingFixtureError,
    ordered_boundaries,
)
from cslr_cost.harness.schema import STAGES, ResultRow
from cslr_cost.harness.stages import (
    Pipeline,
    Stage,
    SyntheticStage,
    spin_ms,
    synthetic_pipeline,
)
from cslr_cost.harness.writer import JsonlWriter, load_rows


N_FRAMES = 8


@pytest.fixture
def pipeline() -> Pipeline:
    return Pipeline(synthetic_pipeline(height=64, width=64))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_pipeline_is_the_seven_stages_in_order(pipeline):
    assert [s.name for s in pipeline] == [
        "decode", "detect", "crop", "pose", "postproc", "head", "ctc"
    ]
    assert all(s.name in STAGES for s in pipeline)


def test_disconnected_pipeline_is_refused():
    """A mismatched pipeline still runs and still produces plausible numbers,
    so it must fail at construction rather than at analysis."""
    a = SyntheticStage("decode", "video", "frames", 0.0)
    b = SyntheticStage("pose", "crops", "keypoints_raw", 0.0)  # expects crops
    with pytest.raises(ValueError, match="not connected"):
        Pipeline([a, b])


def test_granularity_split_is_five_frame_two_clip(pipeline):
    """Stages 1-5 are per-frame, head and ctc per-clip. Per-frame energy
    normalisation rests on this asymmetry."""
    per_frame = [s.name for s in pipeline if s.granularity == "frame"]
    per_clip = [s.name for s in pipeline if s.granularity == "clip"]
    assert per_frame == ["decode", "detect", "crop", "pose", "postproc"]
    assert per_clip == ["head", "ctc"]


def test_composed_run_reaches_glosses(pipeline):
    out = pipeline.run("clip.mp4", N_FRAMES)
    assert isinstance(out, list) and len(out) > 0


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def test_spin_burns_real_time():
    t0 = time.perf_counter()
    spin_ms(20)
    assert (time.perf_counter() - t0) * 1000 >= 18


def test_spin_burns_cpu_rather_than_sleeping():
    """A sleeping stub would pass a timing check while making every energy
    measurement a lie, so the cost must show up as CPU time, not wall time."""
    t0_cpu, t0_wall = time.process_time(), time.perf_counter()
    spin_ms(30)
    cpu, wall = time.process_time() - t0_cpu, time.perf_counter() - t0_wall
    assert cpu >= 0.8 * wall, f"only {cpu:.4f}s CPU over {wall:.4f}s wall — is it sleeping?"


# ---------------------------------------------------------------------------
# Fixtures / isolation
# ---------------------------------------------------------------------------

def test_capture_records_every_boundary(pipeline):
    store = FixtureStore.capture(pipeline, "clip.mp4", N_FRAMES)
    assert ordered_boundaries(store.keys()) == [
        "video", "frames", "boxes", "crops", "keypoints_raw",
        "keypoints", "sequence", "gloss_logits", "glosses",
    ]
    assert store.covers(pipeline.stages)


def test_isolation_does_not_pay_upstream_cost(pipeline):
    """The whole point of P1: replaying 'pose' alone must not invoke decode,
    detect or crop — otherwise isolated pose is really measuring four stages."""
    store = FixtureStore.capture(pipeline, "clip.mp4", N_FRAMES)
    calls_before = {s.name: s.n_calls for s in pipeline}

    pose = next(s for s in pipeline if s.name == "pose")
    for i in range(5):
        pose.run(store.input_for(pose, i))

    for s in pipeline:
        expected = calls_before[s.name] + (5 if s.name == "pose" else 0)
        assert s.n_calls == expected, f"{s.name} ran during isolated pose"


def test_isolated_input_is_the_real_artifact(pipeline):
    """Faithful isolation means the real shape and dtype, not a placeholder."""
    store = FixtureStore.capture(pipeline, "clip.mp4", N_FRAMES)
    pose = next(s for s in pipeline if s.name == "pose")
    crop = store.input_for(pose)
    assert isinstance(crop, np.ndarray) and crop.ndim == 3, "expected one frame's crop"
    assert len(store.units_for(pose)) == N_FRAMES, "one captured unit per frame"


def test_missing_fixture_explains_itself(pipeline):
    store = FixtureStore({"frames": np.zeros((2, 4, 4, 3), np.uint8)})
    pose = next(s for s in pipeline if s.name == "pose")
    with pytest.raises(MissingFixtureError, match="crops"):
        store.input_for(pose)


def test_fixture_round_trip(tmp_path, pipeline):
    store = FixtureStore.capture(
        pipeline, "clip.mp4", N_FRAMES, provenance={"input_resolution": "360p", "clip_id": "c0"}
    )
    path = store.save(tmp_path / "fx.pkl")
    back = FixtureStore.load(path, expect={"input_resolution": "360p"})
    assert np.array_equal(back["keypoints"], store["keypoints"])


def test_fixture_from_a_different_resolution_is_refused(tmp_path, pipeline):
    """A 360p fixture replayed for a 1080p measurement yields a cost curve that
    is flat because the input never actually varied."""
    store = FixtureStore.capture(pipeline, "clip.mp4", N_FRAMES, provenance={"input_resolution": "360p"})
    path = store.save(tmp_path / "fx.pkl")
    with pytest.raises(ValueError, match="does not match"):
        FixtureStore.load(path, expect={"input_resolution": "1080p"})


# ---------------------------------------------------------------------------
# The exit criterion for this chunk
# ---------------------------------------------------------------------------

def test_skeleton_emits_seven_schema_valid_rows(tmp_path, pipeline):
    """End-to-end skeleton: compose, capture, isolate each stage, emit rows.

    Energy is null with a stated reason — the energy layer does not exist yet.
    That is the schema's honesty rule doing its job: the gap is visible in the
    results file rather than showing up as a stage that appears to cost nothing.
    """
    store = FixtureStore.capture(pipeline, "clip.mp4", N_FRAMES)
    out = tmp_path / "results.jsonl"

    with JsonlWriter(out) as w:
        for stage in pipeline:
            reps = 5
            t0 = time.perf_counter_ns()
            for i in range(reps):
                stage.run(store.input_for(stage, i))
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6 / reps

            units = N_FRAMES if stage.granularity == "frame" else 1
            w.write(ResultRow(
                stage=stage.name,
                regime="streaming_simulated",
                measurement_mode="isolated",
                batch_size=1,
                paced_fps=25,
                input_resolution="360p",
                resampled=True,
                source_resolution="210x260",
                device="rtx3060",
                latency_ms_median=elapsed_ms,
                iterations=reps,
                joules_per_frame=None,
                energy_backend="unavailable",
                energy_unavailable_reason="energy layer not implemented yet (Day 10-11)",
                notes=f"{units} unit(s) per call; {stage.config()['notes']}",
            ))

    rows = load_rows(out, validate=True)
    assert len(rows) == 7
    assert [r["stage"] for r in rows] == [
        "decode", "detect", "crop", "pose", "postproc", "head", "ctc"
    ]
    assert all(r["latency_ms_median"] > 0 for r in rows), "a stage reported zero cost"
    assert all(r["joules_per_frame"] is None for r in rows)
    assert all(r["energy_unavailable_reason"] for r in rows)


def test_isolated_costs_track_the_nominal_ordering(pipeline):
    """Pose should dominate the per-frame stages — the shape the profile is
    predicted to have, and a cheap check that costs are not being mixed up."""
    store = FixtureStore.capture(pipeline, "clip.mp4", N_FRAMES)
    measured = {}
    for stage in pipeline:
        t0 = time.perf_counter_ns()
        stage.run(store.input_for(stage))
        measured[stage.name] = (time.perf_counter_ns() - t0) / 1e6

    per_frame = {k: v for k, v in measured.items()
                 if k in {"decode", "detect", "crop", "pose", "postproc"}}
    assert max(per_frame, key=per_frame.get) == "pose"

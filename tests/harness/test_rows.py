"""Measurements into results rows — the join between the harness and the schema.

Every row here goes through `ResultRow.validate()`, so these tests assert the
harness cannot emit a row the schema would refuse. A measurement that cannot be
recorded is a measurement that did not happen.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from cslr_cost.harness.energy.base import EnergyReading
from cslr_cost.harness.reconcile import reconcile
from cslr_cost.harness.rows import (
    RunContext,
    UnknownConfigKey,
    pipeline_config,
    row_from_batch,
    row_from_stage,
    row_from_streaming,
    rows_from_isolation,
)
from cslr_cost.harness.runners.batch import run_batched
from cslr_cost.harness.runners.isolation import StageMeasurement
from cslr_cost.harness.runners.streaming import run_streaming
from cslr_cost.harness.schema import ResultRow
from cslr_cost.harness.stages.base import Pipeline, Stage
from cslr_cost.harness.stages.decode import VideoDecodeStage
from cslr_cost.harness.stages.synthetic import SyntheticStage
from cslr_cost.harness.timing import Durations, MemorySnapshot

N_FRAMES = 20


def _ctx(**kw):
    base = dict(device="rtx3060", input_resolution="290x360", resampled=True)
    base.update(kw)
    return RunContext(**base)


def _pipeline():
    return Pipeline(
        [
            SyntheticStage("decode", "video", "frames", 0.2),
            SyntheticStage("pose", "frames", "keypoints", 0.2),
            SyntheticStage("head", "sequence", "gloss_logits", 0.2, granularity="clip"),
        ]
    )


def _measurement(name="pose", granularity="frame", energy=None, iterations=100):
    return StageMeasurement(
        name=name,
        granularity=granularity,
        measurement_mode="isolated",
        durations=Durations(),
        memory=MemorySnapshot(),
        window_s=60.0,
        warmup_s=5.0,
        iterations=iterations,
        energy=energy,
        energy_unavailable_reason=None if energy else "no rail readable here",
    )


def _reading(joules=10.0, seconds=60.0, idle_watts=None):
    return EnergyReading(
        joules=joules, seconds=seconds, backend="nvml+rapl", idle_watts=idle_watts
    )


# -- config plumbing --------------------------------------------------------


def test_stage_config_keys_are_schema_fields():
    """The invariant behind having no translation table: a config key that is
    not a row field would be silently dropped, producing rows that differ in
    what was measured and agree in what they claim."""
    row_fields = {f.name for f in fields(ResultRow)}
    cfg = pipeline_config(_pipeline())
    for key in cfg:
        assert key in row_fields, f"{key} is not a ResultRow field"


def test_an_unknown_config_key_is_an_error_not_a_dropped_field():
    class Rogue(Stage):
        name, input_key, output_key = "rogue", "video", "frames"

        def config(self):
            return {"gpu_temperature_at_dawn": 42}

        def run(self, x):
            return x

    with pytest.raises(UnknownConfigKey, match="not a ResultRow field"):
        pipeline_config(Pipeline([Rogue()]))


def test_notes_from_several_stages_are_joined_not_overwritten():
    cfg = pipeline_config(_pipeline())
    assert cfg["notes"].count("|") >= 2
    for name in ("decode", "pose", "head"):
        assert name in cfg["notes"]


# -- energy: null with a reason, or a number with a rail --------------------


def test_a_missing_rail_becomes_a_null_with_a_stated_reason():
    """A silent zero would read as a stage that costs nothing — which for the
    CPU-bound front-end stages is exactly the wrong conclusion."""
    row = row_from_stage(
        _measurement(energy=None),
        ctx=_ctx(), n_frames=N_FRAMES, regime="batch", config={},
    )
    row.validate()
    assert row.joules_per_frame is None
    assert row.energy_backend == "unavailable"
    assert "no rail readable" in row.energy_unavailable_reason


def test_a_number_names_its_rail():
    row = row_from_stage(
        _measurement(energy=_reading()),
        ctx=_ctx(), n_frames=N_FRAMES, regime="batch", config={},
    )
    row.validate()
    assert row.energy_backend == "nvml+rapl"
    assert row.joules_per_frame == pytest.approx(10.0 / 100)


def test_incremental_energy_is_null_until_an_idle_baseline_exists():
    row = row_from_stage(
        _measurement(energy=_reading(idle_watts=None)),
        ctx=_ctx(), n_frames=N_FRAMES, regime="batch", config={},
    )
    assert row.joules_per_frame_incremental is None

    row = row_from_stage(
        _measurement(energy=_reading(joules=100.0, seconds=60.0, idle_watts=1.0)),
        ctx=_ctx(), n_frames=N_FRAMES, regime="batch", config={},
    )
    assert row.joules_per_frame_incremental == pytest.approx((100.0 - 60.0) / 100)


def test_a_clip_stage_normalises_energy_per_frame_not_per_call():
    """One iteration of `head` is a whole clip. Dividing its joules by
    iterations alone inflates the head's per-frame cost by n_frames — and the
    head's share of system joules has to be right."""
    frame_stage = _measurement("pose", "frame", _reading(joules=10.0), iterations=100)
    clip_stage = _measurement("head", "clip", _reading(joules=10.0), iterations=100)

    assert frame_stage.joules_per_frame(N_FRAMES) == pytest.approx(0.1)
    assert clip_stage.joules_per_frame(N_FRAMES) == pytest.approx(0.1 / N_FRAMES)


# -- regime consistency, enforced by the schema -----------------------------


def test_a_streaming_row_validates_and_carries_its_pacing():
    pipe = _pipeline()
    result = run_streaming(pipe, None, 6, paced_fps=100.0)
    row = row_from_streaming(result, ctx=_ctx(), pipeline=pipe)
    row.validate()
    assert row.regime == "streaming_simulated"
    assert row.batch_size == 1
    assert row.paced_fps == 100.0
    assert row.overrun_frames is not None


def test_a_batch_row_validates_and_never_carries_pacing():
    """A paced batch run is neither regime, and the schema says so."""
    pipe = _pipeline()
    result = run_batched(pipe, None, 8, batch_size=4)
    row = row_from_batch(result, ctx=_ctx(), pipeline=pipe)
    row.validate()
    assert row.regime == "batch"
    assert row.paced_fps is None


def test_a_batch_rows_latency_is_per_batch_and_says_so():
    pipe = _pipeline()
    result = run_batched(pipe, None, 8, batch_size=4)
    row = row_from_batch(result, ctx=_ctx(), pipeline=pipe)
    assert "per batch of 4" in row.notes
    assert "throughput-equivalent" in row.notes


def test_the_bidirectional_first_gloss_note_reaches_the_row():
    pipe = _pipeline()
    result = run_streaming(pipe, None, 6, paced_fps=100.0)
    row = row_from_streaming(result, ctx=_ctx(), pipeline=pipe)
    assert row.first_gloss_latency_ms is not None
    assert "bidirectional" in row.first_gloss_latency_note


# -- the resampled rule -----------------------------------------------------


def test_a_resampled_row_cannot_carry_wer():
    """Rule 3, enforced by assertion rather than discipline: the ladder
    upscales 210x260 source, so accuracy measured there is indefensible."""
    row = row_from_stage(
        _measurement(), ctx=_ctx(resampled=True), n_frames=N_FRAMES,
        regime="batch", config={},
    )
    row.validate()
    assert row.wer is None
    assert row._wer_note

    row.wer = 21.24
    with pytest.raises(Exception):
        row.validate()


# -- the whole protocol -----------------------------------------------------


def test_isolation_emits_one_row_per_stage_plus_the_composed_total():
    pipe = _pipeline()
    isolated = {s.name: _measurement(s.name, s.granularity) for s in pipe}
    composed = StageMeasurement(
        name="end_to_end", granularity="clip", measurement_mode="composed",
        durations=Durations(), memory=MemorySnapshot(), window_s=60.0,
        warmup_s=5.0, iterations=10,
    )
    rows = rows_from_isolation(
        isolated, composed, ctx=_ctx(), pipeline=pipe, n_frames=N_FRAMES
    )
    assert [r.stage for r in rows] == ["decode", "pose", "head", "end_to_end"]
    for row in rows:
        row.validate()


def test_the_residual_is_attached_to_every_row_not_just_the_composed_one():
    """It is a property of the measurement as a whole; a per-stage row read
    without it looks more precise than it is."""
    pipe = _pipeline()
    isolated = {}
    for s in pipe:
        m = _measurement(s.name, s.granularity)
        m.durations.add(1.0)
        isolated[s.name] = m
    composed = StageMeasurement(
        name="end_to_end", granularity="clip", measurement_mode="composed",
        durations=Durations(), memory=MemorySnapshot(), window_s=60.0,
        warmup_s=5.0, iterations=10,
    )
    composed.durations.add(50.0)

    rec = reconcile(isolated, composed, n_frames=N_FRAMES)
    rows = rows_from_isolation(
        isolated, composed, ctx=_ctx(), pipeline=pipe, n_frames=N_FRAMES,
        reconciliation=rec,
    )
    assert all(r.residual_pct == pytest.approx(rec.residual_pct) for r in rows)


def test_every_row_carries_the_whole_pipelines_configuration():
    """A row that names only its own stage cannot be matched to the composed
    row it should reconcile against."""
    pipe = _pipeline()
    isolated = {s.name: _measurement(s.name, s.granularity) for s in pipe}
    composed = StageMeasurement(
        name="end_to_end", granularity="clip", measurement_mode="composed",
        durations=Durations(), memory=MemorySnapshot(), window_s=60.0,
        warmup_s=5.0, iterations=10,
    )
    rows = rows_from_isolation(
        isolated, composed, ctx=_ctx(), pipeline=pipe, n_frames=N_FRAMES
    )
    shared = ("frontend_model", "head_model", "detector_policy", "decode_path",
              "beam_width", "device", "input_resolution", "resampled")
    for key in shared:
        values = {getattr(r, key) for r in rows}
        assert len(values) == 1, f"{key} differs across rows of one run: {values}"

    # config_hash deliberately includes `stage`, so it identifies a stage in a
    # configuration rather than the configuration alone — which is what the
    # extraction cache and run dedup key on.
    assert len({r.config_hash for r in rows}) == len(rows)
    assert len({r.run_id for r in rows}) == len(rows)


# -- provenance -------------------------------------------------------------


def test_git_provenance_can_come_from_the_environment(monkeypatch):
    """In a git worktree `.git` is a file pointing outside the container's bind
    mount, so `git rev-parse` fails inside and every row silently loses the one
    field tying it to the code that produced it. `scripts/dev.sh` resolves both
    on the host and passes them in."""
    from cslr_cost.harness.schema import _git_dirty, _git_sha

    monkeypatch.setenv("CSLR_GIT_SHA", "deadbeef" * 5)
    monkeypatch.setenv("CSLR_GIT_DIRTY", "true")
    assert _git_sha() == "deadbeef" * 5
    assert _git_dirty() is True

    monkeypatch.setenv("CSLR_GIT_DIRTY", "false")
    assert _git_dirty() is False


def test_a_row_records_whether_the_tree_was_dirty(monkeypatch):
    """A measurement from a dirty tree is not reproducible from its sha, so the
    row says so rather than implying clean provenance."""
    monkeypatch.setenv("CSLR_GIT_SHA", "abc123")
    monkeypatch.setenv("CSLR_GIT_DIRTY", "true")
    row = row_from_stage(
        _measurement(), ctx=_ctx(), n_frames=N_FRAMES, regime="batch", config={}
    )
    row.validate()
    assert row.git_sha == "abc123" and row.git_dirty is True


def test_decode_path_is_the_schema_vocabulary_not_a_free_string():
    """`decode_path` is a comparison axis. A free-form string would make
    `nvdec` and `opencv-ffmpeg/hardware` two incomparable rows describing the
    same thing, so the schema constrains it to a closed set."""
    from cslr_cost.harness.schema import DECODE_PATHS
    from cslr_cost.harness.stages.decode import (
        SCHEMA_DECODE_PATH,
        VideoDecodeStage,
    )

    for value in SCHEMA_DECODE_PATH.values():
        assert value in DECODE_PATHS

    assert VideoDecodeStage(accel="software").config()["decode_path"] == "cpu"


def test_decode_reports_the_granted_path_not_the_requested_one():
    class FakeSource:
        accel = "vaapi"
        codec = "h264"

    stage = VideoDecodeStage(accel="hardware").bind(FakeSource())
    cfg = stage.config()
    assert cfg["decode_path"] == "vaapi"
    assert "granted 'vaapi'" in cfg["notes"]


def test_an_unbound_decode_stage_says_it_is_reporting_the_request():
    cfg = VideoDecodeStage(accel="hardware").config()
    assert "rather than what was granted" in cfg["notes"]


def test_a_row_records_whether_the_idle_baseline_held():
    """`joules_per_frame_incremental` is a *difference*, so it is only as good
    as the baseline subtracted from it. A row could previously read
    `contention: clean` while its bracket had drifted 16.6 -> 46.4 W, with
    nothing recording the fact — the same class of gap as a null git_sha."""
    from cslr_cost.harness.energy import IdleBaseline

    stable = IdleBaseline(watts_before=17.2, watts_after=17.6)
    drifted = IdleBaseline(watts_before=16.6, watts_after=46.4)
    assert stable.status().startswith("stable")
    assert drifted.status().startswith("unstable")
    assert "+179" in drifted.status() or "+180" in drifted.status()

    row = row_from_stage(
        _measurement(), ctx=_ctx(), n_frames=N_FRAMES, regime="batch",
        config={}, contention="clean", idle_baseline=drifted.status(),
    )
    row.validate()
    assert row.to_dict()["idle_baseline"].startswith("unstable")
    assert row.contention == "clean"     # the two are independent signals


def test_an_unchecked_baseline_is_distinguishable_from_a_held_one():
    from cslr_cost.harness.energy import IdleBaseline

    assert IdleBaseline(watts_before=17.2).status().startswith("unchecked")
    assert IdleBaseline(watts_before=float("nan"),
                        unavailable_reason="no rail").status() == "unavailable"

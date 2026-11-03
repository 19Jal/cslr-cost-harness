"""The batch regime's in-situ decomposition.

Why it exists: on the Orin at FP16 the stages timed in isolation added up to
MORE than the composed clip, by up to 8.9% -- the negative residual Harness.MD
calls a bug. The cause was not in the harness arithmetic but in the pipeline:
FFmpeg decodes on worker threads that run ahead of the reader, so inside the
pipeline decode overlaps the GPU stages, and isolation, which calls decode back
to back, serialises that away. `ReadAheadDecode` below reproduces the mechanism
with no video and no GPU, so the property is tested on any machine.
"""

from __future__ import annotations

import time

import pytest

from cslr_cost.harness.fixtures import FixtureStore
from cslr_cost.harness.reconcile import reconcile
from cslr_cost.harness.rows import RunContext, rows_from_isolation
from cslr_cost.harness.runners.isolation import (
    INSITU_ENERGY_REASON,
    measure_all_stages_isolated,
    measure_batch_insitu,
    measure_pipeline_composed,
)
from cslr_cost.harness.stages.base import Pipeline, Stage
from cslr_cost.harness.stages.synthetic import SyntheticStage, synthetic_pipeline

N_FRAMES = 8
FAST = dict(min_seconds=0.4, warmup_seconds=0.05)


class ReadAheadDecode(Stage):
    """A decoder whose next frame is prepared in the background, `work_ms`
    after the previous one was taken -- FFmpeg's frame threads, in miniature.
    Called back to back it waits the full `work_ms`; called after the caller
    spent that long elsewhere, it returns at once."""

    name = "decode"
    input_key = "video"
    output_key = "frames"
    granularity = "frame"
    uses_cuda = False

    def __init__(self, work_ms: float) -> None:
        self.work_s = work_ms / 1000.0
        self._ready = 0.0

    def run(self, x):
        now = time.perf_counter()
        while now < self._ready:
            now = time.perf_counter()
        self._ready = now + self.work_s
        return x


def _overlapping_pipeline(work_ms: float = 2.0) -> Pipeline:
    return Pipeline([
        ReadAheadDecode(work_ms),
        SyntheticStage("pose", "frames", "keypoints", work_ms),     # the GPU stage
        SyntheticStage("head", "sequence", "gloss_logits", 1.0, granularity="clip"),
    ])


def test_overlap_breaks_isolation_but_not_in_situ():
    """The Orin's FP16 failure, reproduced. Isolated decode pays its whole
    read-ahead every call; in the pipeline, pose hides it. Isolation's sum
    overshoots the composed clip -- a negative residual -- while the in-situ
    decomposition adds up."""
    pipe = _overlapping_pipeline()
    fx = FixtureStore.capture(pipe, "clip.mp4", N_FRAMES)
    isolated = measure_all_stages_isolated(pipe, fx, **FAST)
    composed = measure_pipeline_composed(pipe, "clip.mp4", N_FRAMES, **FAST)
    iso_rec = reconcile(isolated, composed, n_frames=N_FRAMES)
    assert iso_rec.residual_pct < -20.0, iso_rec.table()
    assert not iso_rec.passes

    per, total = measure_batch_insitu(pipe, "clip.mp4", N_FRAMES, **FAST)
    rec = reconcile(per, total, n_frames=N_FRAMES)
    assert rec.passes, rec.table()
    assert per["decode"].median_ms < 0.5 * isolated["decode"].per_clip_ms(N_FRAMES), (
        "in situ, decode should cost far less than alone -- pose hid its read-ahead")


def test_every_stage_is_timed_in_place_and_reconciles():
    pipe = Pipeline(synthetic_pipeline(costs={
        "decode": 0.3, "detect": 0.1, "crop": 0.1, "pose": 0.5,
        "postproc": 0.05, "head": 1.0, "ctc": 0.3,
    }))
    per, total = measure_batch_insitu(pipe, "clip.mp4", N_FRAMES, **FAST)
    assert set(per) == {s.name for s in pipe}
    for m in list(per.values()) + [total]:
        assert (m.granularity, m.measurement_mode) == ("clip", "composed")
        assert m.iterations >= 3
    # Each sample is already a clip total: 8 frames x 0.5 ms for pose.
    assert per["pose"].median_ms == pytest.approx(0.5 * N_FRAMES, rel=0.5)
    assert per["head"].median_ms == pytest.approx(1.0, rel=0.5)
    assert reconcile(per, total, n_frames=N_FRAMES).passes


def test_it_never_rewinds_between_clips():
    """It must read the source exactly as `measure_pipeline_composed` does --
    straight through, wrapping mid-clip -- or decode would be timed on
    different frames and different seeks than the clip it is reconciled with."""
    class Source:
        rewinds = 0

        def rewind(self):
            self.rewinds += 1

    src = Source()
    pipe = Pipeline([SyntheticStage("decode", "video", "frames", 0.1)])
    measure_batch_insitu(pipe, src, N_FRAMES, **FAST)
    assert src.rewinds == 0


def test_in_situ_rows_say_why_they_carry_no_energy():
    """They used to read 'no reason recorded'. A null energy must say why."""
    pipe = _overlapping_pipeline(work_ms=0.5)
    per, total = measure_batch_insitu(pipe, "clip.mp4", N_FRAMES, **FAST)
    rows = rows_from_isolation(
        per, total,
        ctx=RunContext(device="test", input_resolution="290x360", resampled=True),
        pipeline=pipe, n_frames=N_FRAMES,
        reconciliation=reconcile(per, total, n_frames=N_FRAMES),
        regime="batch", batch_size=1, contention="clean",
    )
    assert len(rows) == len(per) + 1
    for row in rows:
        row.validate()
        d = row.to_dict()
        assert d["joules_per_frame"] is None
        assert d["energy_unavailable_reason"] == INSITU_ENERGY_REASON
        assert "in situ" in d["notes"]

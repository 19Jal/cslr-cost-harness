"""rows.py — measurements into results rows.

The harness measures; `schema.py` says what a defensible row looks like. This
module is the join between them, and it is deliberately the only place that
knows both.

--------------------------------------------------------------------------
STAGE CONFIG KEYS ARE SCHEMA FIELD NAMES
--------------------------------------------------------------------------
`Stage.config()` returns keys that are already `ResultRow` fields --
`frontend_model`, `detector_policy`, `beam_width` -- rather than
stage-flavoured names mapped here. A translation table would be one more place
for a field to be quietly dropped, and a dropped config field does not fail: it
produces rows that differ in what was measured and agree in what they say they
measured, which is unrecoverable after the fact.

Anything a stage cannot express as a schema field goes to `notes`, and notes
from several stages are joined rather than overwritten. An unknown key is an
error, not a warning.

--------------------------------------------------------------------------
EVERY ROW CARRIES THE WHOLE PIPELINE'S CONFIGURATION
--------------------------------------------------------------------------
A `decode` row records which pose model the run used, even though decode's cost
does not depend on it. Without that, a per-stage row could not be matched to the
composed row it reconciles against, and a table could not be filtered to one
pipeline configuration.

Note `config_hash` covers `stage` and `measurement_mode` as well, so it
identifies *a stage in a configuration* rather than the configuration alone --
which is what the extraction cache and run dedup key on. Rows of
one run therefore share their pipeline fields and differ in their hashes, by
design.

--------------------------------------------------------------------------
ENERGY IS NULL WITH A REASON, OR A NUMBER WITH A RAIL
--------------------------------------------------------------------------
The schema refuses anything else, and this module never fabricates the missing
half. If the meter could not read, `energy_backend` is "unavailable" and
`energy_unavailable_reason` says why. A silent zero would read as a stage that
costs nothing -- which for the CPU-bound front-end stages is exactly the wrong
conclusion, and exactly the one a whole-pipeline account exists to correct.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

from cslr_cost.harness.reconcile import Reconciliation
from cslr_cost.harness.runners.batch import BatchResult, BatchSweep
from cslr_cost.harness.runners.isolation import StageMeasurement
from cslr_cost.harness.runners.streaming import StreamingResult
from cslr_cost.harness.runners.sustained import SustainedResult
from cslr_cost.harness.schema import ResultRow
from cslr_cost.harness.stages.base import Pipeline

_ROW_FIELDS = {f.name for f in fields(ResultRow)}

#: What a config key may be: a schema field, or `notes` (which is joined).
_CONFIG_JOINABLE = "notes"


class UnknownConfigKey(KeyError):
    """A stage's `config()` named something the schema has no field for."""


@dataclass
class RunContext:
    """Provenance shared by every row of one run.

    Separate from the measurement because it is the part a device, a corpus or
    an operator decides -- not something the timing loop can observe.
    """

    device: str
    input_resolution: str
    resampled: bool

    corpus: Optional[str] = None
    split: Optional[str] = None
    clip_id: Optional[str] = None
    n_clips: Optional[int] = None
    arm: Optional[str] = None
    device_power_mode: Optional[str] = None
    source_resolution: Optional[str] = None
    temporal_stride: Optional[int] = None
    thermal_c_start: Optional[float] = None
    thermal_c_end: Optional[float] = None
    notes: str = ""

    def as_fields(self) -> dict[str, Any]:
        out = {
            "device": self.device,
            "input_resolution": self.input_resolution,
            "resampled": self.resampled,
            "corpus": self.corpus,
            "split": self.split,
            "clip_id": self.clip_id,
            "n_clips": self.n_clips,
            "arm": self.arm,
            "device_power_mode": self.device_power_mode,
            "source_resolution": self.source_resolution,
            "temporal_stride": self.temporal_stride,
            "thermal_c_start": self.thermal_c_start,
            "thermal_c_end": self.thermal_c_end,
        }
        return {k: v for k, v in out.items() if v is not None}


def pipeline_config(pipeline: Pipeline) -> dict[str, Any]:
    """Every stage's `config()`, merged, with `notes` joined rather than lost.

    Raises on a key the schema cannot store: a config field that silently
    vanishes produces rows that disagree about what was measured while
    appearing to agree.
    """
    merged: dict[str, Any] = {}
    notes: list[str] = []

    for stage in pipeline:
        for key, value in stage.config().items():
            if key == _CONFIG_JOINABLE:
                if value:
                    notes.append(f"{stage.name}: {value}")
                continue
            if key not in _ROW_FIELDS:
                raise UnknownConfigKey(
                    f"stage '{stage.name}' config key {key!r} is not a ResultRow "
                    f"field. Either add it to the schema (and say why in the "
                    f"module docstring's extensions table) or fold it into "
                    f"'notes' — do not drop it."
                )
            merged[key] = value

    # The stride is the pipeline's, not any stage's. Absent at
    # stride 1, so rows from before it existed keep their config hash.
    merged.update(pipeline.config() if hasattr(pipeline, "config") else {})

    if notes:
        merged[_CONFIG_JOINABLE] = " | ".join(notes)
    return merged


def _energy_fields(
    measurement: StageMeasurement, n_frames: int
) -> dict[str, Any]:
    reading = measurement.energy
    if reading is None:
        return {
            "energy_backend": "unavailable",
            "energy_unavailable_reason": (
                measurement.energy_unavailable_reason or "no reason recorded"
            ),
        }
    return {
        "energy_backend": reading.backend,
        "joules_per_frame": measurement.joules_per_frame(n_frames),
        "joules_per_frame_incremental": measurement.joules_per_frame_incremental(n_frames),
        "joules_per_video": reading.per_unit(measurement.iterations)
        if measurement.granularity == "clip"
        else measurement.joules_per_frame(n_frames) * n_frames,
        "idle_watts": reading.idle_watts,
    }


def _merge_notes(*parts: Optional[str]) -> str:
    return " | ".join(p for p in parts if p)


def row_from_stage(
    measurement: StageMeasurement,
    *,
    ctx: RunContext,
    n_frames: int,
    regime: str,
    config: Mapping[str, Any],
    batch_size: int = 1,
    paced_fps: Optional[float] = None,
    residual_pct: Optional[float] = None,
    contention: Optional[str] = None,
    idle_baseline: Optional[str] = None,
) -> ResultRow:
    """One `StageMeasurement` -> one validated row."""
    cfg = dict(config)
    cfg_notes = cfg.pop(_CONFIG_JOINABLE, "")

    return ResultRow(
        stage=measurement.name,
        regime=regime,
        measurement_mode=measurement.measurement_mode,
        batch_size=batch_size,
        paced_fps=paced_fps,
        latency_ms_median=measurement.median_ms,
        latency_ms_p95=measurement.p95_ms,
        iterations=measurement.iterations,
        window_s=round(measurement.window_s, 3),
        warmup_s=round(measurement.warmup_s, 3),
        residual_pct=residual_pct,
        contention=contention,
        idle_baseline=idle_baseline,
        peak_ram_mb=measurement.memory.peak_ram_mb,
        peak_vram_mb=measurement.memory.peak_vram_mb,
        peak_device_mem_mb=measurement.memory.peak_device_mem_mb,
        device_mem_source=measurement.memory.device_mem_source,
        notes=_merge_notes(cfg_notes, measurement.notes, ctx.notes),
        **_energy_fields(measurement, n_frames),
        **ctx.as_fields(),
        **cfg,
    )


def rows_from_isolation(
    isolated: Mapping[str, StageMeasurement],
    composed: StageMeasurement,
    *,
    ctx: RunContext,
    pipeline: Pipeline,
    n_frames: int,
    reconciliation: Optional[Reconciliation] = None,
    regime: str = "batch",
    batch_size: int = 1,
    paced_fps: Optional[float] = None,
    contention: Optional[str] = None,
    idle_baseline: Optional[str] = None,
) -> list[ResultRow]:
    """P1's whole protocol as rows: each stage isolated, plus the composed total.

    The residual is attached to **every** row of the run, not only the composed
    one. It is a property of the measurement as a whole, and a per-stage row
    read on its own without it would look more precise than it is.
    """
    config = pipeline_config(pipeline)
    residual = reconciliation.residual_pct if reconciliation is not None else None

    common = dict(
        ctx=ctx,
        n_frames=n_frames,
        regime=regime,
        config=config,
        batch_size=batch_size,
        paced_fps=paced_fps,
        residual_pct=residual,
        contention=contention,
        idle_baseline=idle_baseline,
    )
    rows = [row_from_stage(m, **common) for m in isolated.values()]
    rows.append(row_from_stage(composed, **common))
    return rows


def row_from_streaming(
    result: StreamingResult,
    *,
    ctx: RunContext,
    pipeline: Pipeline,
    contention: Optional[str] = None,
    idle_baseline: Optional[str] = None,
) -> ResultRow:
    """P7's streaming regime as one end-to-end row.

    `batch_size` and `regime` come from the result object, where they are
    non-writable fields -- so a streaming row cannot claim a batch size it did
    not run at.
    """
    cfg = dict(pipeline_config(pipeline))
    cfg_notes = cfg.pop(_CONFIG_JOINABLE, "")

    return ResultRow(
        stage="end_to_end",
        regime=result.regime,
        measurement_mode="composed",
        batch_size=result.batch_size,
        paced_fps=result.paced_fps,
        latency_ms_median=result.frame_durations.median,
        latency_ms_p95=result.frame_durations.p95,
        first_gloss_latency_ms=result.first_gloss_latency_ms,
        first_gloss_latency_note=result.first_gloss_latency_note,
        overrun_frames=result.overrun_frames,
        overrun_ms_total=round(result.overrun_ms_total, 3),
        seconds_per_video=round(result.wall_s, 4),
        peak_ram_mb=result.memory.peak_ram_mb,
        peak_vram_mb=result.memory.peak_vram_mb,
        peak_device_mem_mb=result.memory.peak_device_mem_mb,
        device_mem_source=result.memory.device_mem_source,
        contention=contention,
        idle_baseline=idle_baseline,
        energy_backend="unavailable",
        energy_unavailable_reason=(
            "regime runners do not open an energy window: the paced loop is "
            "mostly idle by design, so a window over it measures the device's "
            "idle draw as much as the pipeline's work. Per-stage energy comes "
            "from the isolation runner (P1)."
        ),
        notes=_merge_notes(
            cfg_notes,
            f"duty cycle {result.duty_cycle:.1%}" if result.duty_cycle is not None else "",
            f"pacing jitter median {result.pacing_jitter.median:.3f} ms"
            if result.pacing_jitter.median is not None
            else "",
            "" if result.kept_up else "DID NOT KEEP UP at this pacing",
            ctx.notes,
        ),
        **ctx.as_fields(),
        **cfg,
    )


def _batch_note(result: BatchResult) -> str:
    if result.ms_per_frame is None:
        return ""
    unit = f"batch of {result.batch_size}"
    if result.stride > 1:
        unit += f" kept frames ({result.batch_size * result.stride} decoded, stride {result.stride})"
    return f"latency is per {unit}; throughput-equivalent {result.ms_per_frame:.3f} ms/frame"


def row_from_batch(
    result: BatchResult,
    *,
    ctx: RunContext,
    pipeline: Pipeline,
    sweep: Optional[BatchSweep] = None,
    contention: Optional[str] = None,
    idle_baseline: Optional[str] = None,
) -> ResultRow:
    """P7's batch regime as one end-to-end row.

    `latency_ms_median` is **per batch**, not per frame -- nothing is available
    until the batch finishes. The throughput-equivalent goes to
    `seconds_per_video` and `notes`, never into a latency field.

    Given the `BatchSweep` that produced `result`, the row also carries that
    window's energy (joules per video over every clip completed), per-batch
    latency pooled over all of those clips, and the median clip's wall time.
    Without one, energy is unavailable and says why.
    """
    cfg = dict(pipeline_config(pipeline))
    cfg_notes = cfg.pop(_CONFIG_JOINABLE, "")

    if sweep is not None:
        pooled = sweep.batch_durations
        composed = sweep.composed
        measured: dict[str, Any] = dict(
            latency_ms_median=pooled.median,
            latency_ms_p95=pooled.p95,
            seconds_per_video=round(composed.median_ms / 1000.0, 4),
            iterations=composed.iterations,
            window_s=round(composed.window_s, 3),
            peak_ram_mb=composed.memory.peak_ram_mb,
            peak_vram_mb=composed.memory.peak_vram_mb,
            peak_device_mem_mb=composed.memory.peak_device_mem_mb,
            device_mem_source=composed.memory.device_mem_source,
            contention=contention,
            idle_baseline=idle_baseline,
            **_energy_fields(composed, sweep.n_frames),
        )
    else:
        measured = dict(
            latency_ms_median=result.batch_durations.median,
            latency_ms_p95=result.batch_durations.p95,
            seconds_per_video=round(result.wall_s, 4),
            peak_ram_mb=result.memory.peak_ram_mb,
            peak_vram_mb=result.memory.peak_vram_mb,
            peak_device_mem_mb=result.memory.peak_device_mem_mb,
            device_mem_source=result.memory.device_mem_source,
            energy_backend="unavailable",
            energy_unavailable_reason=(
                "regime runners do not open an energy window; per-stage energy "
                "comes from the isolation runner (P1)."
            ),
        )

    return ResultRow(
        stage="end_to_end",
        regime=result.regime,
        measurement_mode="composed",
        batch_size=result.batch_size,
        paced_fps=result.paced_fps,          # None: the schema rejects otherwise
        **measured,
        notes=_merge_notes(
            cfg_notes,
            _batch_note(result),
            f"batched stages: {', '.join(result.batched_stages) or 'none'}",
            ctx.notes,
        ),
        **ctx.as_fields(),
        **cfg,
    )


def row_from_sustained(
    result: SustainedResult,
    *,
    ctx: RunContext,
    pipeline: Pipeline,
    contention: Optional[str] = None,
    idle_baseline: Optional[str] = None,
) -> ResultRow:
    """P3's ten-minute hold as one row.

    This is the only row that fills `sustained_fps_10min`, and the only one
    whose thermal fields are not null — every other measurement is far too
    short to have a meaningful start and end temperature.

    Energy here is the most representative figure the harness produces: it is
    integrated over ten thermally-settled minutes rather than a 60 s window on
    a device that may still be warming.
    """
    cfg = dict(pipeline_config(pipeline))
    cfg_notes = cfg.pop(_CONFIG_JOINABLE, "")

    energy_fields: dict[str, Any]
    if result.energy is None:
        energy_fields = {
            "energy_backend": "unavailable",
            "energy_unavailable_reason": (
                result.energy_unavailable_reason or "no reason recorded"
            ),
        }
    else:
        joules_per_frame = (
            result.energy.joules / (result.n_clips * result.n_frames)
            if result.n_clips else None
        )
        incremental = result.energy.joules_incremental
        energy_fields = {
            "energy_backend": result.energy.backend,
            "joules_per_frame": joules_per_frame,
            "joules_per_frame_incremental": (
                None if incremental is None or not result.n_clips
                else incremental / (result.n_clips * result.n_frames)
            ),
            "joules_per_video": (
                result.energy.joules / result.n_clips if result.n_clips else None
            ),
            "idle_watts": result.energy.idle_watts,
        }

    paced = result.paced_fps
    return ResultRow(
        stage="end_to_end",
        regime="streaming_simulated" if paced else "batch",
        measurement_mode="composed",
        batch_size=1,
        paced_fps=paced,
        latency_ms_median=result.clip_durations.median,
        latency_ms_p95=result.clip_durations.p95,
        sustained_fps_10min=result.sustained_fps,
        seconds_per_video=round(result.window_s / result.n_clips, 4)
        if result.n_clips else None,
        thermal_c_start=result.trace.start_c,
        thermal_c_end=result.trace.end_c,
        warmup_s=round(result.warmup.warmup_s, 2),
        iterations=result.n_clips,
        window_s=round(result.window_s, 2),
        peak_ram_mb=result.memory.peak_ram_mb,
        peak_vram_mb=result.memory.peak_vram_mb,
        peak_device_mem_mb=result.memory.peak_device_mem_mb,
        device_mem_source=result.memory.device_mem_source,
        contention=contention,
        idle_baseline=idle_baseline,
        overrun_frames=(
            result.overrun_frames_first_fifth + result.overrun_frames_last_fifth
            if paced else None
        ),
        notes=_merge_notes(
            cfg_notes,
            f"P3 sustained hold, {result.window_s / 60.0:.1f} min",
            f"peak {result.peak_fps:.1f} fps" if result.peak_fps else "",
            f"throttle {result.throttle_pct:.1f}% below peak"
            if result.throttle_pct is not None else "",
            f"decay {result.decay_pct:+.1f}% first->last fifth"
            if result.decay_pct is not None else "",
            "" if result.warmup.plateaued else "DID NOT PLATEAU before the window",
            result.notes,
            ctx.notes,
        ),
        **energy_fields,
        **ctx.as_fields(),
        **cfg,
    )

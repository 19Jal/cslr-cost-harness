"""schema.py — the results contract.

One JSONL, one row per measurement. Flat, verbose, append-only.

This module is the harness/pipeline interface. Everything that emits a
measurement goes through `ResultRow` + `validate`, so the rules that make the
numbers defensible are enforced structurally rather than by discipline:

  1. `resampled == true`  =>  WER fields MUST be null. Upscaling adds no
     information, so accuracy on a resampled clip is indefensible.
  2. `regime == streaming_simulated`  =>  `batch_size == 1`. No exceptions, no
     "effective batch". Quoting a batch-32 figure as streaming is the exact
     reporting error this harness exists to prevent.
  3. Energy is never silently zero. A null `joules_per_frame` requires an
     explicit `energy_unavailable_reason`. (Idle must be subtracted *and*
     stated; an unmeasured stage must say so.)
  4. The non-negotiable fields must be present on every row, even when null.

`regime` and `batch_size` are the two most likely to be forgotten and the two
that most easily invalidate a comparison, so both are required and cross-checked.

--------------------------------------------------------------------------
FIELDS AND WHY THEY EXIST
--------------------------------------------------------------------------
Each field below is traceable to a requirement of the measurement protocol
(docs/protocol.md). Changing one changes the results contract:

  joules_per_frame_incremental, idle_watts   P2 "report BOTH total system
                                             power and incremental-over-idle"
  energy_backend,                            P2 idle must be *stated*; a null
  energy_unavailable_reason                  energy must never read as zero
  iterations, window_s                       P1 joules/iter = integral(P dt) /
                                             iterations — needed to audit
  residual_pct                               P1 "report the residual
                                             explicitly"
  overrun_frames, overrun_ms_total           P7 rule 2 "log the overrun;
                                             accumulated overrun is a more
                                             honest answer than mean fps"
  thermal_c_start, warmup_s                  P3 warm up until plateau, then
                                             measure; A has only thermal_c_end
  clip_id, n_clips, harness_version          provenance / reproducibility
  idle_baseline                              P2's over-idle figures are only
                                             as good as the baseline; the row
                                             must say whether it held. Null =
                                             not checked.
  contention                                 P2's idle subtraction assumes the
                                             machine is ours. On a shared box it
                                             is not, and RAPL's package counter
                                             cannot separate our work from a
                                             second tenant's. Null = not checked.
  detect_runtime, frontend_runtime,          the same model is a
                                             different measurement under PyTorch
                                             and under a TensorRT engine. Null =
  head_runtime                               recorded before the field existed
                                             (every such row is PyTorch). The
                                             detector has its own, because the
                                             `--detector rtmdet` ablation can
                                             run PyTorch detection against
                                             TensorRT pose, and a row that
                                             cannot say so is unreadable.
  trt_version, detect_engine,                an engine is
  frontend_engine, head_engine               (model, precision, device, TensorRT
                                             version, profile), and tactic
                                             selection can differ between two
                                             builds of the same config, so the
                                             row carries the engine file's hash.
  frontend_checkpoint,                       a model NAME is not an
  frontend_input                             identity. `rtmpose-m` ships as a
                                             COCO-WholeBody checkpoint and as a
                                             COCO+UBody one, and they score
                                             differently -- 58.2 published for
                                             the first against 0.631 measured
                                             for the second. A row naming only
                                             the family cannot say which was
                                             measured. `frontend_input` is
                                             separate because 256x192 and
                                             384x288 are two cost regimes, not
                                             two rows of one grid: the crop is
                                             fixed at every ladder rung
                                             (Appendix F #10), so the ladder has
                                             never measured the larger one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

HARNESS_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

#: The seven pipeline stages, in order, plus the composed whole.
#: Note stage 1 is *video decode*, not camera capture.
STAGES = (
    "decode",
    "detect",
    "crop",
    "pose",
    "postproc",
    "head",
    "ctc",
    "end_to_end",
)

#: P7: streaming-simulated is batch 1 paced to native fps; batch is unpaced.
REGIMES = ("streaming_simulated", "batch")

#: P1: isolated stages are measured alone in a tight loop; composed is the
#: whole pipeline. Never silently mixed.
MEASUREMENT_MODES = ("isolated", "composed")

#: Stage 1 instrumentation: hardware decode vs a silent CPU fallback can differ
#: by 2x and appears in no model-level profile.
DECODE_PATHS = ("nvdec", "v4l2", "mediacodec", "vaapi", "cpu", "none")

#: What executed a neural stage. The exported and the eager model
#: are different measurements of the same weights.
RUNTIMES = ("pytorch", "tensorrt", "onnxruntime")

ENERGY_BACKENDS = (
    "nvml",             # dev GPU / Jetson GPU rail via NVML
    "rapl",             # x86 CPU package energy
    "nvml+rapl",        # dev box, both rails
    "tegra_ina3221",    # Orin Nano on-board rails
    "usb_meter",        # inline USB-C logger (Pi 5)
    "unavailable",      # explicitly not measured — reason required
)

#: Fields that define a configuration (as opposed to a measurement outcome or
#: provenance). `config_hash` is computed over exactly these, so two runs of
#: the same configuration hash identically.
CONFIG_FIELDS = (
    "arm",
    "corpus",
    "split",
    "device",
    "device_power_mode",
    "regime",
    "batch_size",
    "paced_fps",
    "decode_path",
    "stage",
    "measurement_mode",
    "frontend_model",
    "frontend_checkpoint",
    "frontend_input",
    "frontend_precision",
    "head_model",
    "head_precision",
    "detect_runtime",
    "frontend_runtime",
    "head_runtime",
    "trt_version",
    "input_resolution",
    "resampled",
    "source_resolution",
    "temporal_stride",
    "reconstruction",
    "detector_policy",
    "beam_width",
)

#: "Non-negotiable fields ... Every one of these exists because
#: some claim in the report depends on it." Present on every row, even if null.
NON_NEGOTIABLE_FIELDS = (
    "stage",
    "regime",
    "batch_size",
    "measurement_mode",
    "del",
    "sub",
    "ins",
    "joules_per_frame",
    "input_resolution",
    "resampled",
)

#: Any of these coexisting with `resampled == true` is refused. Generous on
#: purpose — catch aliases too.
_WER_KEYS = frozenset(
    {"wer", "word_error_rate", "wer_total", "del", "sub", "ins",
     "wer_del", "wer_sub", "wer_ins", "deletions", "substitutions", "insertions"}
)

_WER_NULL_NOTE = "null because resampled=true — WER is only valid at native resolution"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SchemaError(AssertionError):
    """Base: a row violates the results contract."""


class ResampledWerError(SchemaError):
    """A row tried to attach a WER to a resampled (upscaled) input."""


class MissingFieldError(SchemaError):
    """A non-negotiable field is absent."""


class EnumError(SchemaError):
    """A controlled-vocabulary field carries an unknown value."""


class RegimeConsistencyError(SchemaError):
    """regime and batch_size / pacing disagree (P7 rule 1)."""


class EnergyError(SchemaError):
    """Energy is null without a stated reason, or zero where that is implausible."""


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Optional[Path]:
    """Nearest ancestor containing `.git`, searching from this file then cwd.

    Resolved by search rather than by a fixed number of `.parent` hops, so
    moving this module within the package cannot silently detach every row's
    provenance from the repository it was measured in. `.git` may be a
    directory or a file (worktrees and submodules), so both count.
    """
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists():
                return candidate
    return None


def _git_sha(repo: Optional[Path] = None) -> Optional[str]:
    """Current HEAD sha, or None if unavailable (e.g. no commits yet).

    `CSLR_GIT_SHA` wins when set, and it has to: in a git **worktree** `.git`
    is a *file* pointing at `<main repo>/.git/worktrees/<name>`, a path that is
    not inside the container's bind mount. `git rev-parse` there fails, and
    every row silently loses the one field that ties it to the code that
    produced it. `scripts/dev.sh` resolves both on the host and passes them in.
    """
    env = os.environ.get("CSLR_GIT_SHA")
    if env:
        return env.strip() or None
    repo = repo or _repo_root()
    if repo is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(repo: Optional[Path] = None) -> Optional[bool]:
    """True if the working tree has uncommitted changes. None if undeterminable.

    A measurement taken from a dirty tree is not reproducible from its git_sha,
    so the row should say so rather than imply a clean provenance.

    `CSLR_GIT_DIRTY` wins when set, for the worktree reason in `_git_sha`.
    """
    env = os.environ.get("CSLR_GIT_DIRTY")
    if env:
        return env.strip().lower() in ("1", "true", "yes")
    repo = repo or _repo_root()
    if repo is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def new_run_id() -> str:
    """`2026-09-14T10:22:01Z_a3f9c2` — sortable timestamp plus a collision tag."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


def config_hash(row: Mapping[str, Any]) -> str:
    """Stable 8-hex-char hash over CONFIG_FIELDS only.

    Deliberately excludes measurements and provenance so the same configuration
    measured twice hashes the same — which is what makes the extraction cache and run deduplication possible.
    """
    payload = {k: row.get(k) for k in CONFIG_FIELDS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# The row
# ---------------------------------------------------------------------------

@dataclass
class ResultRow:
    """One measurement: (run, stage, device, condition).

    Python cannot name a field `del` (reserved word), so the deletion count is
    `del_` in Python and serialises to `"del"` in JSON. `sub`/`ins` are fine as
    they are; `del_` is the only asymmetry and `to_dict`/`from_dict` handle it.

    Construct, then call `.validate()` — or hand it to a `JsonlWriter`, which
    validates on every write.
    """

    # --- identity / provenance -------------------------------------------
    stage: str
    regime: str
    measurement_mode: str
    batch_size: int
    input_resolution: str
    resampled: bool

    run_id: str = field(default_factory=new_run_id)
    config_hash: Optional[str] = None          # filled by __post_init__
    git_sha: Optional[str] = field(default_factory=_git_sha)
    git_dirty: Optional[bool] = field(default_factory=_git_dirty)
    harness_version: str = HARNESS_VERSION

    # --- what was run ----------------------------------------------------
    arm: Optional[str] = None                  # "B" (pose-only) | "D" (skeleton+RGB) | "P" (profiling)
    corpus: Optional[str] = None
    split: Optional[str] = None
    clip_id: Optional[str] = None
    n_clips: Optional[int] = None

    # --- where it ran ----------------------------------------------------
    device: Optional[str] = None               # "orin-nano-8gb", "rtx3060", ...
    device_power_mode: Optional[str] = None    # P4: pin clocks, record the mode
    decode_path: Optional[str] = None

    # --- regime detail ---------------------------------------------------
    paced_fps: Optional[float] = None          # P7 rule 2: 25 fps => 40 ms budget

    # --- model configuration ---------------------------------------------
    frontend_model: Optional[str] = None
    frontend_checkpoint: Optional[str] = None  # which weights, not which family
    frontend_input: Optional[str] = None       # "256x192" / "384x288" -- a cost regime
    frontend_precision: Optional[str] = None
    head_model: Optional[str] = None
    head_precision: Optional[str] = None
    detect_runtime: Optional[str] = None       # pytorch | tensorrt | onnxruntime
    frontend_runtime: Optional[str] = None
    head_runtime: Optional[str] = None
    trt_version: Optional[str] = None          # engine provenance
    detect_engine: Optional[str] = None        # engine file sha256, first 16 hex
    frontend_engine: Optional[str] = None
    head_engine: Optional[str] = None
    source_resolution: Optional[str] = None
    temporal_stride: Optional[int] = None
    reconstruction: Optional[str] = None
    detector_policy: Optional[str] = None
    beam_width: Optional[int] = None

    # --- accuracy (null whenever resampled) ------------------------------
    wer: Optional[float] = None
    del_: Optional[float] = None
    sub: Optional[float] = None
    ins: Optional[float] = None
    _wer_note: Optional[str] = None

    # --- latency (P5: median and p95, never mean) -----------------------
    latency_ms_median: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    first_gloss_latency_ms: Optional[float] = None
    first_gloss_latency_note: Optional[str] = None

    # --- energy (P1, P2) ------------------------------------------------
    joules_per_frame: Optional[float] = None
    joules_per_frame_incremental: Optional[float] = None   # over idle
    joules_per_video: Optional[float] = None
    idle_watts: Optional[float] = None
    energy_backend: Optional[str] = None
    energy_unavailable_reason: Optional[str] = None

    # --- isolation bookkeeping (P1) --------------------------------------
    iterations: Optional[int] = None
    window_s: Optional[float] = None
    residual_pct: Optional[float] = None       # composed vs sum-of-isolated

    # --- throughput / memory ---------------------------------------------
    seconds_per_video: Optional[float] = None
    sustained_fps_10min: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    #: PyTorch's allocator high-water mark. Misses TensorRT, cuDNN workspaces
    #: and the CUDA context; kept for the rows that already carry it.
    peak_vram_mb: Optional[float] = None
    #: The process's device-memory footprint (see `timing.MemorySnapshot`), and
    #: how it was read: "nvmap-process" or "nvml-largest-process".
    peak_device_mem_mb: Optional[float] = None
    device_mem_source: Optional[str] = None

    # --- streaming overrun ledger (P7 rule 2) ----------------------------
    overrun_frames: Optional[int] = None
    overrun_ms_total: Optional[float] = None

    # --- thermal (P3) ----------------------------------------------------
    warmup_s: Optional[float] = None
    thermal_c_start: Optional[float] = None
    thermal_c_end: Optional[float] = None

    # --- was the machine ours while we measured? (P2) --------------------
    #: None = not checked. "clean" = quiet throughout. Anything else describes
    #: what was sharing the machine, and means the energy figures on this row
    #: are not defensible. Structural rather than a note, so contaminated rows
    #: can be filtered out of a table months later instead of being read as
    #: results.
    contention: Optional[str] = None

    #: None = not checked. Otherwise "stable"/"unstable" with the drift, from
    #: `IdleBaseline.status()`. The companion to `contention`, and needed for
    #: the same reason: `joules_per_frame_incremental` is a *difference*, so it
    #: is only as good as the baseline subtracted from it. A row could
    #: previously read `contention: clean` while its bracket had drifted
    #: 16.6 -> 46.4 W, with nothing recording the fact.
    idle_baseline: Optional[str] = None

    # --- quality by-product ----------------------------------------------
    kp_failure_rate: Optional[float] = None    # Gap 7, free to log

    notes: str = ""

    def __post_init__(self) -> None:
        if self.config_hash is None:
            self.config_hash = config_hash(self.to_dict(validate=False))
        # A resampled row can never carry WER; annotate it so the null is
        # self-explaining in the results file rather than looking like an
        # omission.
        if self.resampled and self._wer_note is None:
            self._wer_note = _WER_NULL_NOTE

    # -- serialisation -----------------------------------------------------

    def to_dict(self, validate: bool = True) -> dict[str, Any]:
        """Flat JSON-ready dict. `del_` -> `del`."""
        out: dict[str, Any] = {}
        for f in fields(self):
            key = "del" if f.name == "del_" else f.name
            out[key] = getattr(self, f.name)
        if validate:
            validate_row(out)
        return out

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ResultRow":
        """Inverse of `to_dict`. Unknown keys are rejected loudly."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        unknown: list[str] = []
        for k, v in row.items():
            name = "del_" if k == "del" else k
            if name in known:
                kwargs[name] = v
            else:
                unknown.append(k)
        if unknown:
            raise SchemaError(
                f"unknown field(s) {unknown} — the schema is a contract; add the "
                "field to ResultRow deliberately rather than smuggling it in a row"
            )
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)

    def validate(self) -> "ResultRow":
        validate_row(self.to_dict(validate=False))
        return self


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def assert_no_wer_on_resampled(row: Mapping[str, Any]) -> None:
    """Refuse a WER value on a resampled input.

    THE safety interlock. The resolution ladder upscales PHOENIX-2014 (native
    210x260); compute depends on tensor dimensions, so the ladder is exact for
    cost and worthless for accuracy. A WER measured there would be
    indefensible, and it is an easy mistake once the ladder and the native
    corpora flow through the same code path.

    Note the asymmetry: *presence of a non-null value* is the violation, not
    presence of the key. Non-negotiable fields must appear on every row, so
    `"del": null` is required, while `"del": 3.1` alongside resampled=true is
    refused.
    """
    if not row.get("resampled", False):
        return
    offenders = sorted(
        k for k in row
        if k.lower() in _WER_KEYS and row.get(k) is not None
    )
    if offenders:
        raise ResampledWerError(
            "REFUSED to emit result: WER on a resampled input is indefensible "
            f"(upscaling adds no information). Offending field(s): {offenders}. "
            f"resampled=True, source_resolution={row.get('source_resolution', 'unknown')}, "
            f"input_resolution={row.get('input_resolution', 'unknown')}. "
            "Accuracy is only ever measured at each corpus's native resolution."
        )


def _assert_non_negotiable_present(row: Mapping[str, Any]) -> None:
    missing = [k for k in NON_NEGOTIABLE_FIELDS if k not in row]
    if missing:
        raise MissingFieldError(
            f"missing non-negotiable field(s): {missing}. Every one exists "
            "because some claim in the report depends on it; they must be "
            "present even when null."
        )
    # These specifically may never be null — a row without them is uninterpretable.
    for key in ("stage", "regime", "measurement_mode", "batch_size", "resampled"):
        if row.get(key) is None:
            raise MissingFieldError(
                f"'{key}' may not be null — a measurement without it cannot be "
                "compared against any other row."
            )


def _assert_enums(row: Mapping[str, Any]) -> None:
    for key, vocab in (
        ("stage", STAGES),
        ("regime", REGIMES),
        ("measurement_mode", MEASUREMENT_MODES),
    ):
        val = row.get(key)
        if val not in vocab:
            raise EnumError(f"{key}={val!r} not in {vocab}")

    decode_path = row.get("decode_path")
    if decode_path is not None and decode_path not in DECODE_PATHS:
        raise EnumError(f"decode_path={decode_path!r} not in {DECODE_PATHS}")

    backend = row.get("energy_backend")
    if backend is not None and backend not in ENERGY_BACKENDS:
        raise EnumError(f"energy_backend={backend!r} not in {ENERGY_BACKENDS}")

    for key in ("detect_runtime", "frontend_runtime", "head_runtime"):
        runtime = row.get(key)
        if runtime is not None and runtime not in RUNTIMES:
            raise EnumError(f"{key}={runtime!r} not in {RUNTIMES}")


def _assert_regime_consistency(row: Mapping[str, Any]) -> None:
    """P7 rule 1: streaming means batch 1, paced. No 'effective batch'."""
    regime, batch_size = row.get("regime"), row.get("batch_size")

    if regime == "streaming_simulated":
        if batch_size != 1:
            raise RegimeConsistencyError(
                f"regime=streaming_simulated requires batch_size=1, got {batch_size}. "
                "No exceptions, no 'effective batch', no padding several clips "
                "together. Quoting a batched throughput figure as streaming fps is "
                "a common reporting error."
            )
        if row.get("paced_fps") is None:
            raise RegimeConsistencyError(
                "regime=streaming_simulated requires paced_fps (the pacing rate the "
                "loop enforced, e.g. 25). An unpaced loop measures throughput, not "
                "whether the pipeline keeps up — which is the deployment question."
            )
    elif regime == "batch":
        if isinstance(batch_size, int) and batch_size < 1:
            raise RegimeConsistencyError(f"batch_size must be >= 1, got {batch_size}")
        if row.get("paced_fps") is not None:
            raise RegimeConsistencyError(
                "regime=batch must not carry paced_fps — the batch regime is "
                "explicitly unpaced. A paced batch run is neither regime."
            )


def _assert_energy_honest(row: Mapping[str, Any]) -> None:
    """P2: energy is never silently absent, and never silently zero."""
    joules = row.get("joules_per_frame")
    backend = row.get("energy_backend")
    reason = row.get("energy_unavailable_reason")

    if joules is None:
        if not reason:
            raise EnergyError(
                "joules_per_frame is null without energy_unavailable_reason. "
                "An unmeasured stage must say why (e.g. 'RAPL not readable as "
                "non-root on dev box') so a null is never mistaken for zero, and "
                "so the gap is visible when the table is assembled."
            )
        return

    if joules == 0:
        raise EnergyError(
            "joules_per_frame == 0 is never a real measurement. If the stage was "
            "not measured, set it null with an energy_unavailable_reason."
        )
    if joules < 0:
        raise EnergyError(f"joules_per_frame={joules} is negative")
    if backend is None or backend == "unavailable":
        raise EnergyError(
            f"joules_per_frame={joules} was reported with energy_backend={backend!r}. "
            "A number must name the rail it came from, or it cannot be compared "
            "across devices."
        )


def _assert_latency_sane(row: Mapping[str, Any]) -> None:
    """P5: median and p95. If both are present, p95 cannot precede the median."""
    med, p95 = row.get("latency_ms_median"), row.get("latency_ms_p95")
    if med is not None and p95 is not None and p95 < med:
        raise SchemaError(
            f"latency_ms_p95 ({p95}) < latency_ms_median ({med}) — the tail cannot "
            "be faster than the middle; check the percentile accumulator."
        )
    for key in ("latency_ms_median", "latency_ms_p95", "first_gloss_latency_ms"):
        val = row.get(key)
        if val is not None and val < 0:
            raise SchemaError(f"{key}={val} is negative")


def validate_row(row: Mapping[str, Any]) -> None:
    """Run every contract check. Raises the most specific SchemaError."""
    _assert_non_negotiable_present(row)
    _assert_enums(row)
    assert_no_wer_on_resampled(row)
    _assert_regime_consistency(row)
    _assert_energy_honest(row)
    _assert_latency_sane(row)


#: Public alias — `validate(row_or_dict)` accepts either form.
def validate(row: Any) -> Any:
    if isinstance(row, ResultRow):
        return row.validate()
    validate_row(row)
    return row

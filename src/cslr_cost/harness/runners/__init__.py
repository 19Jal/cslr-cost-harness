"""Measurement runners.

`isolation` implements the per-stage isolation protocol: each stage alone in a tight loop
at steady state, and the composed pipeline measured the same way.

`streaming` (batch 1, paced to 25 fps, overrun ledger) and `batch` (16-64,
unpaced) implement P7's two regimes. They are separate loop structures on
purpose -- P7 warns that bolting the second onto the first means rewriting the
harness, and they genuinely differ: streaming paces and asks whether the
pipeline keeps up; batch saturates and asks how much the device gets through.
"""

from cslr_cost.harness.runners.batch import (  # noqa: F401
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    BatchResult,
    run_batched,
)
from cslr_cost.harness.runners.isolation import (  # noqa: F401
    DEFAULT_MIN_SECONDS,
    DEFAULT_WARMUP_SECONDS,
    IsolationSweep,
    StageMeasurement,
    measure_all_stages_isolated,
    measure_batch_insitu,
    measure_isolation_sweep,
    measure_pipeline_composed,
    measure_stage_isolated,
)
from cslr_cost.harness.runners.paced import (  # noqa: F401
    PacedSweep,
    measure_all_stages_paced,
    measure_paced_sweep,
    measure_stage_paced,
    measure_streaming_composed,
    measure_streaming_insitu,
    stage_period_s,
)
from cslr_cost.harness.runners.sustained import (  # noqa: F401
    POST_HOLD_SETTLE_S,
    SUSTAINED_MINUTES,
    THROTTLE_NOTICEABLE_PCT,
    SustainedResult,
    run_sustained,
)
from cslr_cost.harness.runners.streaming import (  # noqa: F401
    DEFAULT_PACED_FPS,
    LOOKAHEAD_CAUSAL,
    LOOKAHEAD_FULL_CLIP,
    StreamingResult,
    run_streaming,
)

__all__ = [
    "DEFAULT_MIN_SECONDS",
    "DEFAULT_WARMUP_SECONDS",
    "StageMeasurement",
    "measure_stage_isolated",
    "measure_pipeline_composed",
    "measure_batch_insitu",
    "measure_all_stages_isolated",
    "IsolationSweep",
    "measure_isolation_sweep",
    "DEFAULT_PACED_FPS",
    "LOOKAHEAD_FULL_CLIP",
    "LOOKAHEAD_CAUSAL",
    "StreamingResult",
    "run_streaming",
    "BATCH_SIZES",
    "DEFAULT_BATCH_SIZE",
    "BatchResult",
    "run_batched",
    "PacedSweep",
    "measure_stage_paced",
    "measure_all_stages_paced",
    "measure_streaming_composed",
    "measure_streaming_insitu",
    "measure_paced_sweep",
    "stage_period_s",
    "SustainedResult",
    "run_sustained",
    "SUSTAINED_MINUTES",
    "POST_HOLD_SETTLE_S",
    "THROTTLE_NOTICEABLE_PCT",
]

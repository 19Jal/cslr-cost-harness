"""reconcile.py — isolated sum vs composed total.

    "The isolated sum should approach the composed total. Report the residual
     explicitly — it is real (scheduling, memory traffic, cache effects), and
     hiding it would defeat the point of measuring."

And the reconciliation gate:

    "per-stage times summing to within 5% of measured end-to-end in each regime"

    "If per-stage times do not sum to end-to-end, you have unaccounted
     overhead — usually data movement or synchronisation. Find it now. An
     unexplained 20% makes the per-stage table untrustworthy."

So this is not a reporting convenience — it is the gate. It is a first-class
module rather than a notebook cell because it decides whether the per-stage
accounting can be trusted at all.

**Sign convention.** `residual_pct` is `(composed - sum_isolated) / composed`:

  *positive* — the composed run costs MORE than its parts. The usual and
  expected case: scheduling, data movement between stages, synchronisation,
  cache pressure. Real overhead that belongs to the pipeline, not to any stage.

  *negative* — the parts cost MORE than the whole. This is a warning, not a
  finding. It usually means an isolated stage is being measured on an
  unrepresentatively hot cache, a stage is doing less work in isolation than in
  composition, or granularity is being mis-scaled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from cslr_cost.harness.runners.isolation import StageMeasurement

#: The reconciliation gate.
DEFAULT_TOLERANCE_PCT = 5.0


@dataclass
class Reconciliation:
    """The comparison, and whether it passes the gate."""

    per_stage_ms: dict[str, float]
    sum_isolated_ms: float
    composed_ms: float
    n_frames: int
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT

    @property
    def residual_ms(self) -> float:
        return self.composed_ms - self.sum_isolated_ms

    @property
    def residual_pct(self) -> float:
        if self.composed_ms == 0:
            return float("nan")
        return 100.0 * self.residual_ms / self.composed_ms

    @property
    def passes(self) -> bool:
        return abs(self.residual_pct) <= self.tolerance_pct

    def share(self, stage: str) -> float:
        """Percentage of the isolated sum attributable to one stage.

        This is the shape of the paper's headline table. Note it is a share of
        the *isolated sum*, not of the composed total — the residual belongs to
        the pipeline, and silently distributing it across stages would invent
        precision the measurement does not have.
        """
        if self.sum_isolated_ms == 0:
            return float("nan")
        return 100.0 * self.per_stage_ms[stage] / self.sum_isolated_ms

    def verdict(self) -> str:
        if self.passes:
            return f"PASS  residual {self.residual_pct:+.2f}% (tolerance ±{self.tolerance_pct}%)"
        # Which causes go on which branch is not obvious, and getting it
        # backwards sends you looking for the wrong bug at the exact moment the
        # gate fails. Work it out from the sign each time:
        #
        #   sum too SMALL  ->  positive residual  ->  isolation measured cheap
        #   sum too LARGE  ->  negative residual  ->  isolation measured dear
        #
        # Cache-hot replay makes an isolated stage look *cheaper* than it is,
        # so it belongs with the positive branch, not the negative one.
        direction = (
            "composed costs more than its parts. Either the pipeline has real "
            "overhead no stage is charged for (data movement between stages, "
            "synchronisation, CPU/GPU concurrency, the frame loop itself), or "
            "isolation measured too cheap — a cache-hot replay cycling a small "
            "working set, or a stage doing less work alone than it does in "
            "context"
            if self.residual_ms > 0
            else "parts cost more than the whole. Either a stage does MORE work "
            "in isolation than composed (re-deriving its input, reloading a "
            "model inside the timed region), or the per-clip scaling is wrong — "
            "a 'clip' stage multiplied by n_frames is the usual one, and it "
            "overshoots by exactly that factor"
        )
        return (
            f"FAIL  residual {self.residual_pct:+.2f}% exceeds ±{self.tolerance_pct}% — "
            f"{direction}"
        )

    def table(self) -> str:
        """Human-readable per-stage breakdown."""
        width = max((len(s) for s in self.per_stage_ms), default=5)
        lines = [f"{'stage'.ljust(width)}  {'ms/clip':>9}  {'share':>7}"]
        for stage, ms in self.per_stage_ms.items():
            lines.append(f"{stage.ljust(width)}  {ms:9.2f}  {self.share(stage):6.1f}%")
        lines.append(f"{'-' * width}  {'-' * 9}  {'-' * 7}")
        lines.append(f"{'sum'.ljust(width)}  {self.sum_isolated_ms:9.2f}")
        lines.append(f"{'composed'.ljust(width)}  {self.composed_ms:9.2f}")
        lines.append(f"{'residual'.ljust(width)}  {self.residual_ms:9.2f}  "
                     f"{self.residual_pct:+6.2f}%")
        lines.append(self.verdict())
        return "\n".join(lines)


def reconcile(
    isolated: Mapping[str, StageMeasurement],
    composed: StageMeasurement,
    n_frames: int,
    *,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> Reconciliation:
    """Compare the sum of isolated stages against the composed measurement.

    `n_frames` is required, not inferred: it is what converts a per-frame
    stage's per-call cost into its per-clip contribution, and inferring it from
    a fixture would hide a mismatch between the clip that was captured and the
    clip that was measured.
    """
    per_stage: dict[str, float] = {}
    for name, m in isolated.items():
        clip_ms = m.per_clip_ms(n_frames)
        if clip_ms is None:
            raise ValueError(
                f"stage '{name}' has no timing samples — it cannot be reconciled. "
                "A stage that produced no measurement must not be silently "
                "treated as costing zero."
            )
        per_stage[name] = clip_ms

    if composed.median_ms is None:
        raise ValueError("composed measurement has no timing samples")

    return Reconciliation(
        per_stage_ms=per_stage,
        sum_isolated_ms=sum(per_stage.values()),
        composed_ms=composed.median_ms,
        n_frames=n_frames,
        tolerance_pct=tolerance_pct,
    )

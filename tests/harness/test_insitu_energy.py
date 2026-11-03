"""Energy in the streaming regime: what the in-situ window can and cannot say.

A whole 24-configuration grid came back with the streaming rows reading
`energy_backend: unavailable, reason: "no reason recorded"`. Two separate
faults behind one symptom, and this file pins both:

  * the composed row *could* have had energy — the runner opens a window and
    attaches it to `end_to_end` — and the caller simply passed no meter;
  * the per-stage rows could not, because seven stages run interleaved inside
    one paced window and the rails integrate the window rather than the stage.

The second is physics and stays that way. What must never recur is a null
energy that does not say why: the schema's rule is that energy is never
silently absent, and "no reason recorded" is the sound of that rule being
satisfied in letter and broken in spirit.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from cslr_cost.harness.energy.base import EnergyMeter, EnergyReading
from cslr_cost.harness.runners.paced import measure_streaming_insitu
from cslr_cost.harness.stages.base import Pipeline
from cslr_cost.harness.stages.synthetic import SyntheticStage

FPS = 200.0          # 5 ms slots, so a clip is quick
N = 6


class _ConstantPowerMeter(EnergyMeter):
    backend = "nvml"

    def __init__(self, watts: float = 40.0, idle_watts: Optional[float] = None) -> None:
        self.watts = watts
        self.idle_watts = idle_watts
        self._t0 = 0.0

    @staticmethod
    def available() -> bool:
        return True

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> EnergyReading:
        dt = time.perf_counter() - self._t0
        return EnergyReading(joules=self.watts * dt, seconds=dt,
                             backend=self.backend, idle_watts=self.idle_watts)


class _Source:
    def rewind(self) -> None:
        pass


def _pipeline() -> Pipeline:
    return Pipeline([
        SyntheticStage("decode", "video", "frames", 0.3),
        SyntheticStage("pose", "frames", "keypoints", 0.3),
        SyntheticStage("head", "sequence", "gloss_logits", 1.0, granularity="clip"),
    ])


def _run(meter=None):
    return measure_streaming_insitu(
        _pipeline(), _Source(), N,
        paced_fps=FPS, min_seconds=0.25, warmup_clips=0, meter=meter,
    )


def test_a_meter_reaches_the_composed_row():
    _stages, end_to_end, _ = _run(_ConstantPowerMeter())
    assert end_to_end.energy is not None
    assert end_to_end.energy.backend == "nvml"
    assert end_to_end.joules_per_frame(N) > 0


def test_without_a_meter_the_composed_row_states_why():
    _stages, end_to_end, _ = _run(None)
    assert end_to_end.energy is None
    assert end_to_end.energy_unavailable_reason
    assert "meter" in end_to_end.energy_unavailable_reason


@pytest.mark.parametrize("meter", [None, _ConstantPowerMeter()])
def test_per_stage_rows_never_claim_energy_but_always_say_why(meter):
    """Even with a meter: one window's joules are not seven stages' joules."""
    stages, _end_to_end, _ = _run(meter)
    for name, m in stages.items():
        assert m.energy is None, f"{name} claimed per-stage energy from a shared window"
        reason = m.energy_unavailable_reason
        assert reason, f"{name} has a null energy with no stated reason"
        assert reason != "no reason recorded"
        assert "in situ" in reason and "paced isolation" in reason, (
            "the reason must say why it cannot be measured here and where the "
            f"streaming regime's per-stage energy does come from, got: {reason}"
        )


def test_the_idle_figure_carries_into_over_idle_joules():
    """A meter that already knows its idle floor reports over-idle, not totals alone."""
    _stages, end_to_end, _ = _run(_ConstantPowerMeter(watts=40.0, idle_watts=10.0))
    total = end_to_end.joules_per_frame(N)
    over_idle = end_to_end.joules_per_frame_incremental(N)
    assert over_idle is not None and 0 < over_idle < total

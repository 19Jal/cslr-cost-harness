"""Measurement harness for cslr-cost-harness.

One tool that produces every number in the paper, in one schema.

The schema (`cslr_cost.harness.schema`) is the contract between the
harness and the rest of the pipeline. It is deliberately the first module
written and the one that changes least: every downstream plot, table and claim
reads these rows.
"""

from cslr_cost.harness.schema import (  # noqa: F401
    ResultRow,
    SchemaError,
    ResampledWerError,
    MissingFieldError,
    EnumError,
    RegimeConsistencyError,
    EnergyError,
    validate,
    STAGES,
    REGIMES,
    MEASUREMENT_MODES,
)
from cslr_cost.harness.writer import JsonlWriter  # noqa: F401

__all__ = [
    "ResultRow",
    "SchemaError",
    "ResampledWerError",
    "MissingFieldError",
    "EnumError",
    "RegimeConsistencyError",
    "EnergyError",
    "validate",
    "JsonlWriter",
    "STAGES",
    "REGIMES",
    "MEASUREMENT_MODES",
]

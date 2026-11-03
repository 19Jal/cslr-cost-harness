"""Pipeline stages behind one interface.

    decode -> detect -> crop -> pose -> postproc -> head -> ctc

`base` defines the interface and the composition; `synthetic` provides
known-cost stand-ins used to validate the harness before real models attach.

The real adapters live one per module and are **not** re-exported here, on
purpose: `ctc` needs `ctcdecode`, `head` needs MSKA on `sys.path`, and
`detect`/`pose` need mmdet/mmpose -- none of which exist in the host venv. An
eager re-export would make `import cslr_cost.harness.stages` fail
outside the container, including for the schema tests that need none of them.
Import the module you want:

    from cslr_cost.harness.stages.decode   import VideoDecodeStage
    from cslr_cost.harness.stages.detect   import WholeFrameDetectStage
    from cslr_cost.harness.stages.crop     import TopdownCropStage
    from cslr_cost.harness.stages.pose     import RtmposeStage
    from cslr_cost.harness.stages.postproc import KeypointPostprocStage
    from cslr_cost.harness.stages.head     import MskaHeadStage
    from cslr_cost.harness.stages.ctc      import CtcDecodeStage
"""

from cslr_cost.harness.stages.base import (  # noqa: F401
    BOUNDARIES,
    Granularity,
    Pipeline,
    Stage,
)
from cslr_cost.harness.stages.synthetic import (  # noqa: F401
    DEFAULT_COST_MS,
    SyntheticStage,
    spin_ms,
    synthetic_pipeline,
)

__all__ = [
    "BOUNDARIES",
    "Granularity",
    "Pipeline",
    "Stage",
    "SyntheticStage",
    "DEFAULT_COST_MS",
    "spin_ms",
    "synthetic_pipeline",
]

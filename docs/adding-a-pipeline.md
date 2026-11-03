# Measuring your own pipeline

The harness measures any pipeline built from **stages**. A stage is one step of
the pipeline behind a small interface (`src/cslr_cost/harness/stages/base.py`),
so it can be timed inside the running pipeline *and* replayed alone for energy
(protocol rule P1).

## The shape every pipeline has

```
for each frame:  decode -> detect -> crop -> pose -> postproc      (granularity "frame")
assemble the per-frame outputs into one sequence
once per clip:   head -> ctc                                        (granularity "clip")
```

An RGB model fills the same slots: its crop stage resizes the frame and its pose
slot holds the per-frame CNN (see `stages/rgb_trt.py`). Keeping the slots the
same is what makes the per-stage tables comparable across families.

Stages connect through named **boundaries** (`BOUNDARIES` in `base.py`:
`video`, `frames`, `boxes`, `crops`, `keypoints_raw`, `keypoints`, `sequence`,
`gloss_logits`, `glosses`). A stage declares which boundary it reads and which it
writes; the pipeline refuses to build if consecutive stages do not connect,
because a disconnected pipeline still runs and still produces plausible numbers.

## Writing a stage

```python
from typing import Any
from cslr_cost.harness.stages.base import Stage


class MyPoseStage(Stage):
    name = "pose"
    input_key = "crops"          # what it consumes, one unit per call
    output_key = "keypoints_raw" # what it produces
    granularity = "frame"        # "frame" for decode..postproc, "clip" for head and ctc
    uses_cuda = True             # GPU work: timed with CUDA events, synchronised

    def __init__(self, onnx_path, precision="fp16"):
        self.onnx_path, self.precision = onnx_path, precision

    def config(self) -> dict[str, Any]:
        # Merged into every results row. Name exactly what ran.
        return {"frontend_model": "my-pose-net", "frontend_precision": self.precision,
                "frontend_runtime": "tensorrt"}

    def setup(self) -> None:
        # Load the model, build or load the TensorRT engine, allocate buffers.
        # Never timed.
        ...

    def run(self, x: Any) -> Any:
        # One crop in, one keypoint array out. Must not modify `x`: isolation
        # replays the same recorded inputs many times.
        ...
```

Rules that matter:

- **One unit per `run` call**: one frame for frame stages, one clip for clip
  stages. Per-clip energy is divided by the clip's frame count automatically.
- **Do the deployed work, and only that.** No training losses, no evaluation-only
  outputs inside `run`, and no work charged to the wrong stage (a pose stage
  that re-does the crop double-counts it).
- **Set `supports_batching = True` only** if you override `run_batch` with a
  genuinely batched implementation; otherwise the batch regime honestly loops.
- **Keep loading in `setup`**, and let `warmup` reach steady state; both happen
  outside the measured window.

## Running it

Compose the stages into a `Pipeline` and hand it to the runners, as
`harness/cli.py` does for the built-in pipelines (`_build_pipeline` for the
keypoint pipeline, the RGB and TwoStream-SLR builders next to it). The easiest
route is to add a builder there and a `--rgb-model`-style choice, so that your
pipeline gets the same idle bracket, contention gate, isolation, reconciliation,
paced and sustained runs, and results rows as every other.

`stages/synthetic.py` has stages with known, busy-spun cost. Build your pipeline
from them first: if the harness does not report the costs you asked for, the
problem is in the wiring, not in your model.

## Models whose code you cannot redistribute

If a model's code carries no licence (CorrNet's, for example), keep it out of
the repository. Export it to ONNX from your own clone with the same two-graph
split the RGB stages expect (a per-frame network, and a per-clip head that takes
the padded per-frame features), save them as `models/onnx/rgb/corrnet.frontend.onnx` and
`corrnet.head.onnx` (per-frame output: layer-2 maps of shape (N, 128, 28, 28)),
and run `harness ladder --rgb-model corrnet`, which goes through
`stages/rgb_trt.py`.

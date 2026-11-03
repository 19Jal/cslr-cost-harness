# cslr-cost-harness

Whole-pipeline cost measurement for continuous sign language recognition (CSLR):
time, energy and memory for **every stage between the encoded video and the
glosses**, on the device that would run them.

Most CSLR papers report the cost of the recognition network alone. For a
keypoint model that leaves out the pose estimator, which runs on every frame and
is usually the largest cost in the pipeline. This harness counts the whole
pipeline, the same way, for keypoint, RGB and two-stream models:

```
for each frame:  decode -> detect -> crop -> pose (or CNN) -> postproc
once per clip:   head -> CTC decode
```

It is the measurement tool behind *The Hidden Cost of Pose: Whole-Pipeline
Measurement of Keypoint-Based Continuous Sign Language Recognition* (see
[Citing](#citing)).

## What it measures

| | |
|---|---|
| **Per-stage time** | Each stage timed inside the running pipeline (CUDA events for GPU stages), median and p95, never the mean. Stage times must sum to the end-to-end time within 5%, or the run is flagged. |
| **Per-stage energy** | Each stage replayed alone on its real recorded inputs for ≥ 60 s, because board power monitors sample far too slowly to attribute energy within one pass. Reported both in total and above idle. |
| **Idle and contention** | Idle measured before and after every run at the same clocks; a drift above 20% withholds the over-idle figures. Any other process on the GPU, or foreign CPU above 10% in any window, marks the run as not clean. |
| **Two regimes** | *Batch* (batch 1, plus an optional batch-N row) and *live*: frames paced at 25 fps, slept not spun, no frame dropped, overruns counted; a 10-minute sustained run; and first-gloss latency, stated honestly for bidirectional heads. |
| **Memory** | Peak per-process device memory, including TensorRT's (nvmap on Jetson, NVML on desktop GPUs). |
| **Accuracy** | WER with deletions, substitutions and insertions, under both PHOENIX-2014 scoring conventions (markers removed, and the official `sclite` protocol). Never on upscaled input. |

Every measurement is one row in a flat JSONL file, with the device, power mode,
clock state, engine versions, idle bracket, contention verdict and git commit
that produced it. The rules are in [docs/protocol.md](docs/protocol.md).

## Supported devices and pipelines

**Devices:** NVIDIA Jetson Orin (INA3221 power rails) and x86 desktops with an
NVIDIA GPU (NVML for the GPU, Intel RAPL for the CPU package). See
[docs/devices.md](docs/devices.md).

**Pipelines measured in the paper** (released checkpoints, TensorRT FP32/FP16):

| Family | Pipeline | Front end | How to get it |
|---|---|---|---|
| Keypoint | MSKA | HRNet-w48 DARK, 384×288, flip test | `export/export_hrnet_onnx.py`, `export/export_mska_ensemble_onnx.py` (needs a clone of MSKA) |
| Keypoint | TwoStream-SLR keypoint stream | HRNet-w48 + flip | `export/export_twostream_onnx.py` (needs a clone of TwoStream-SLR) |
| RGB | VAC, VAC + SMKD | ResNet18 | `export/export_rgb_cslr_onnx.py` (VAC is ported here, Apache-2.0) |
| RGB | CorrNet | ResNet18 + correlation | bring your own ONNX export ([docs/adding-a-pipeline.md](docs/adding-a-pipeline.md)); its code carries no licence, so it is not included |
| RGB + keypoint | TwoStream-SLR | HRNet-w48 + flip, S3D | `export/export_twostream_onnx.py` |
| Pose sweep | DWPose t/s/m/l (RTMPose), INT8 PTQ | — | `export/export_rtmpose_onnx.py`, `export/quantize_rtmpose_int8.py` |

Third-party model code is never copied in, except where its licence allows (see
[NOTICE](NOTICE)). The adapters import it from your own clone, and checkpoints
are downloaded from their authors.

## Quick start (desktop GPU)

```bash
docker build -t cslr-cost-harness:dev .
scripts/dev.sh pytest                     # unit tests; GPU tests skip without data
scripts/dev.sh python -m cslr_cost.harness.cli rails
```

`rails` reports which energy rails the machine exposes, whether clocks are
pinned, and whether any other process holds GPU memory. Fix what it reports
before measuring.

Then, with the data laid out as in [docs/data.md](docs/data.md):

```bash
# 1. Build the 1080-line cost input from one PHOENIX-2014 clip
scripts/resample_ladder.sh scripts/clips_paper1.txt \
    data/phoenix2014/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train \
    data/res_ladder/out

# 2. Export the networks to ONNX (once; engines are built on each device)
scripts/dev.sh python -m cslr_cost.export.export_rgb_cslr_onnx --model vac \
    --weights data/VAC_SMKD/<checkpoint>.pt

# 3. One cost session: every pipeline, back to back, on one device
scripts/dev.sh scripts/run_session.sh
```

On a Jetson, the harness runs natively (no container); see
[docs/devices.md](docs/devices.md).

## Measuring your own pipeline

A pipeline is a list of stages behind one small interface. Wrap your detector,
pose estimator, CNN or recognition head as a `Stage`, and the harness times,
isolates, paces and reconciles it like the built-in ones.
[docs/adding-a-pipeline.md](docs/adding-a-pipeline.md) walks through it.

## Layout

| Path | What it is |
|---|---|
| `src/cslr_cost/harness/` | The harness: stages, runners (isolation, batch, paced, sustained), energy backends, contention gate, schema, CLI |
| `src/cslr_cost/harness/stages/` | Stage adapters: decode, detect, crop, pose (RTMPose/HRNet), postproc, head (MSKA), CTC, RGB and TwoStream-SLR |
| `src/cslr_cost/eval/` | WER with del/sub/ins; the official PHOENIX-2014 `sclite` wrapper; COCO-WholeBody AP for pose engines |
| `src/cslr_cost/export/` | ONNX exports with parity checks against PyTorch; INT8 post-training quantisation |
| `src/cslr_cost/rgb_cslr/` | VAC port, data loading and WER for the RGB and two-stream models |
| `src/cslr_cost/ladder/` | Resolution-ladder cataloguing and the guard that refuses WER on resampled input |
| `scripts/` | Container wrapper, ladder generation, example session |
| `docs/` | Protocol, devices, data layout, adding a pipeline |
| `tests/` | Unit tests (synthetic stages with known cost), plus GPU and data tests that skip when absent |

## Status

This is a first public release of the harness used for the paper. Not yet
included:

- the keypoint-extraction pipeline and its WER driver (MSKA's own evaluation
  script can produce the hypotheses meanwhile; score them with `eval/wer.py` and
  `eval/official_phoenix.py`);
- a temporal-stride option (frames skipped before pose); every stage here runs
  on every frame.

## Citing

If you use the harness, please cite the paper (details in
[CITATION.cff](CITATION.cff)):

```bibtex
@article{veda2026hiddencost,
  title   = {The Hidden Cost of Pose: Whole-Pipeline Measurement of
             Keypoint-Based Continuous Sign Language Recognition},
  author  = {Veda, Jalu and Cahyafazri, Muhammad Rafi and Leu, Jenq-Shiou},
  journal = {under review},
  year    = {2026}
}
```

## Licence

Apache License 2.0 ([LICENSE](LICENSE)). Code adapted from other projects keeps
its own notice ([NOTICE](NOTICE)).

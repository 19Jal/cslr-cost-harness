# Data, checkpoints and exports

Nothing third-party is redistributed here. Download each item from its authors,
under their terms, and place it under `data/` (or point `CSLR_DATA` at another
directory). Inside the container, `scripts/dev.sh` mounts that directory at the
same absolute path.

```
data/
  phoenix2014/phoenix2014-release/phoenix-2014-multisigner/   PHOENIX-2014 (RWTH; research licence)
      features/fullFrame-210x260px/{train,dev,test}/...         PNG frames
      annotations/manual/{train,dev,test}.corpus.csv
      evaluation/                                               evaluatePhoenix2014.sh + .stm
  res_ladder/out/{360p,480p,720p,1080p}/<clip>.mp4              built by scripts/resample_ladder.sh
  rtmpose/        DWPose / RTMPose whole-body checkpoints and configs (mmpose model zoo)
  rtmdet/         RTMDet-nano person detector (optional; --detector rtmdet)
  coco-wholebody/ HRNet-w48 DARK 384x288 COCO-WholeBody checkpoint (mmpose model zoo)
  VAC_SMKD/       VAC + SMKD released checkpoint
  TwoStream-SLRT/ TwoStream-SLR code and released checkpoints
  onnx/           ONNX exports (written by src/cslr_cost/export/)
  refs/           markers-removed references (below)
MSKA/             a clone of MSKA, with its released checkpoint (for its scorer and head)
models/onnx/      exports the CLI reads by default (MSKA ensemble, RGB, TwoStream-SLR)
models/engines/   TensorRT engines, built on first use, per device
```

## Cost input

Cost is measured on one 128-frame PHOENIX-2014 clip (`scripts/clips_paper1.txt`)
upscaled to 1,080 lines (872×1080) and encoded as H.264, because native
PHOENIX video (210×260) is far below what a camera delivers. The pose stage sees
a fixed-size crop and does not depend on this; decode and resize do. Rows from
upscaled input are marked `resampled: true`, and no WER is ever attached to
them.

## Exports

Each export script checks parity against PyTorch and writes a sidecar JSON with
the checkpoint it came from and the parity it reached. TensorRT engines are not
portable between GPUs, so they are built on each device from these ONNX files
the first time they are needed (`harness prebuild` builds them ahead of a run).

| Network | Script |
|---|---|
| HRNet-w48 DARK (MSKA's and TwoStream-SLR's front end) | `python -m cslr_cost.export.export_hrnet_onnx` |
| MSKA four-head ensemble | `python -m cslr_cost.export.export_mska_ensemble_onnx` |
| RTMPose / DWPose t, s, m, l | `python -m cslr_cost.export.export_rtmpose_onnx` |
| RTMPose INT8 (post-training quantisation, PHOENIX train crops) | `python -m cslr_cost.export.quantize_rtmpose_int8` |
| RTMDet-nano | `python -m cslr_cost.export.export_rtmdet_onnx` |
| VAC, VAC + SMKD | `python -m cslr_cost.export.export_rgb_cslr_onnx --model vac` |
| TwoStream-SLR, and its keypoint stream | `python -m cslr_cost.export.export_twostream_onnx` |

Released checkpoints are loaded through a restricted unpickler
(`cslr_cost/safe_load.py`) that accepts only the types a state dict needs, so a
checkpoint cannot run code when it is loaded.

## References for the markers-removed convention

`rgb_cslr/evaluate.py` scores RGB and two-stream hypotheses against MSKA's
cleaned references, matched by clip key. Provide them as
`data/refs/phoenix-2014.{split}.mska-refs.jsonl` (one `{"key", "ref"}` per line;
override with `CSLR_MSKA_REFS`). The official-protocol score needs only the
dataset's own `evaluation/` folder.

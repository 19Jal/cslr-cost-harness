"""quantize_rtmpose_int8.py — RTMPose FP32 ONNX -> INT8 QDQ ONNX.

Post-training quantisation for the pose-estimator sweep. The output is an ONNX graph carrying explicit
QuantizeLinear/DequantizeLinear nodes with calibrated scales, which TensorRT
compiles in *explicit* quantization mode.

--------------------------------------------------------------------------
WHY QDQ AND NOT TENSORRT'S CALIBRATOR
--------------------------------------------------------------------------
TensorRT can calibrate INT8 itself (`IInt8EntropyCalibrator2`, still present in
10.3). We do not use it, for three reasons:

  * **Portability.** An engine is not portable; an ONNX file is. A QDQ export
    made here is the same artifact the Orin compiles, exactly as the FP32 and
    FP16 arms already work (`engines.py`). A calibrator would have to re-run on
    every device, against whatever calibration data happened to be there.
  * **It outlives TensorRT 10.** Implicit quantization is deprecated in 10 and
    removed in 11.
  * **QAT produces this format too.** INT8-QAT produces QDQ graphs. Using
    the same path for PTQ means the precision grid's cells differ by their
    weights, not by their toolchain.

--------------------------------------------------------------------------
SYMMETRIC, OR TENSORRT WILL NOT PARSE IT
--------------------------------------------------------------------------
TensorRT's explicit-quantization parser accepts only zero zero-points
("TensorRT only supports symmetric quantization"). ONNX Runtime quantizes
*activations* asymmetrically by default even with `QInt8`, so the first export
passed ONNX Runtime's own parity check and then failed to build on every scale
(2026-09-30, at `/head/gau/ln/Constant_3`). `ActivationSymmetric` is therefore
set, and `asymmetric_zero_points` checks the written graph and refuses it
before anything downstream -- the Orin compiles this same file -- can find out
the hard way.

--------------------------------------------------------------------------
CALIBRATION DATA COMES FROM THE TRAIN SPLIT
--------------------------------------------------------------------------
Activation ranges are fitted on real crops, and they are fitted on **train**.
Calibrating on dev or test leaks the evaluation set into the model — quietly,
because nothing downstream would ever raise. The crops are produced by the
harness's own `decode` -> `detect` -> `crop` stages, so what the calibrator
sees is what deployment feeds the extractor, down to the affine warp.

Preprocessing must match `pose.py::_preprocess` exactly — BGR->RGB, the pinned
mean/std, NCHW float32. Calibrating on differently-scaled inputs produces
ranges that are wrong everywhere and a model that is merely bad rather than
broken, which is the hardest failure to notice.

--------------------------------------------------------------------------
THIS RUNS ON CPU
--------------------------------------------------------------------------
Deliberately. ONNX Runtime collects activation ranges on the CPU provider, and
the crop pipeline is CPU-only (whole-frame `detect` does no work, `crop` is a
`warpAffine`). So the export can run while *another tenant* holds the GPU.

**It must NOT run while our own measurement grid is measuring.** CPU-only does
not mean contention-free: the grid times CPU stages (`decode`, `crop`,
`postproc`, `ctc`), integrates RAPL package energy, and its contention check
flags foreign CPU above 10% of the window. A calibration run pins several
cores at 100% — exactly what that check exists to catch, and it can void the
configuration being measured. This was learned by doing it, on 2026-09-23.
Run exports while the grid is idle or waiting, never while it is measuring.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from cslr_cost.harness.stages.pose import RTMPOSE_INPUT_HW
from cslr_cost.harness.stages.pose_trt import RTMPOSE_MEAN, RTMPOSE_STD

#: PHOENIX-2014 as distributed, inside the container.
DEFAULT_FRAMES_ROOT = Path(
    "data/phoenix2014/phoenix2014-release/phoenix-2014-multisigner"
    "/features/fullFrame-210x260px"
)
DEFAULT_ONNX_DIR = Path("data/onnx")
DEFAULT_OUT_DIR = Path("models/onnx")

#: Enough crops to fit activation ranges without the export taking an hour.
#: Entropy calibration is a histogram method; a few hundred diverse crops is
#: the usual working range, and the clips are sampled across the split rather
#: than taken from its head so signers and lighting vary.
DEFAULT_N_CROPS = 512
DEFAULT_N_CLIPS = 32


def preprocess(imgs: list[np.ndarray]) -> np.ndarray:
    """uint8 HWC BGR crops -> normalised NCHW float32, as `pose.py` does it."""
    x = np.stack([np.ascontiguousarray(i) for i in imgs]).astype(np.float32)
    x = x[..., ::-1]                                   # BGR -> RGB
    x = x.transpose(0, 3, 1, 2)                        # NHWC -> NCHW
    mean = np.asarray(RTMPOSE_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(RTMPOSE_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray((x - mean) / std)


def collect_crops(
    split: str,
    frames_root: Path,
    n_clips: int = DEFAULT_N_CLIPS,
    n_crops: int = DEFAULT_N_CROPS,
    seed: int = 0,
    scale: str = "m",
) -> list[np.ndarray]:
    """Real crops through the deployed front end, CPU only.

    Clips are sampled evenly across the split rather than taken from its start:
    PHOENIX is ordered by date and signer, so the first N clips are not a
    sample of the corpus.
    """
    from cslr_cost.harness.stages.crop import TopdownCropStage
    from cslr_cost.harness.stages.decode import open_video
    from cslr_cost.harness.stages.detect import WholeFrameDetectStage

    split_root = frames_root / split
    clips = sorted(p for p in split_root.iterdir() if p.is_dir())
    if not clips:
        raise SystemExit(f"no clips under {split_root}")

    idx = np.linspace(0, len(clips) - 1, min(n_clips, len(clips))).astype(int)
    chosen = [clips[i] for i in dict.fromkeys(idx.tolist())]
    per_clip = max(1, n_crops // len(chosen))

    from cslr_cost.harness.stages.pose import crop_size_for

    detect, crop = WholeFrameDetectStage(), TopdownCropStage(input_size=crop_size_for(scale))
    detect.setup()
    crop.setup()

    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for clip in chosen:
        src_dir = clip / "1" if (clip / "1").is_dir() else clip
        source = open_video(src_dir)
        try:
            n = len(source)
            want = sorted(rng.choice(n, size=min(per_clip, n), replace=False).tolist())
            for i in range(n):
                frame = source.read()
                if i in want:
                    out.append(crop.run(detect.run(frame)).img)
                if len(out) >= n_crops:
                    break
        finally:
            source.close()
        if len(out) >= n_crops:
            break

    if not out:
        raise SystemExit("collected no crops")
    return out


class CropCalibrationReader:
    """Feeds calibration batches to onnxruntime's static quantizer."""

    def __init__(self, crops: list[np.ndarray], input_name: str, batch: int = 8) -> None:
        self.input_name = input_name
        self._batches = [
            preprocess(crops[i : i + batch]) for i in range(0, len(crops), batch)
        ]
        self._it: Optional[Iterator[np.ndarray]] = None

    def get_next(self) -> Optional[dict[str, np.ndarray]]:
        if self._it is None:
            self._it = iter(self._batches)
        nxt = next(self._it, None)
        return None if nxt is None else {self.input_name: nxt}

    def rewind(self) -> None:
        self._it = None


def _input_name(onnx_path: Path) -> str:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    initialisers = {i.name for i in model.graph.initializer}
    names = [i.name for i in model.graph.input if i.name not in initialisers]
    if len(names) != 1:
        raise SystemExit(f"{onnx_path.name} has inputs {names}; expected exactly one")
    return names[0]


def parity(fp32: Path, int8: Path, crops: list[np.ndarray]) -> dict[str, float]:
    """Decode both graphs on held-out crops and compare keypoints.

    A quantized extractor is *supposed* to move keypoints — that movement is
    the thing the precision sweep measures. What this checks is that it moved by an
    amount consistent with quantization rather than with a broken export: a
    wrong preprocessing or a mis-set axis shows up as tens of pixels, not
    fractions of one.
    """
    import onnxruntime as ort

    from cslr_cost.harness.stages.pose_trt import decode_simcc

    x = preprocess(crops)
    name = _input_name(fp32)

    def run(path: Path) -> tuple[np.ndarray, np.ndarray]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outs = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, {name: x}))}
        return decode_simcc(outs["simcc_x"], outs["simcc_y"])

    ref_k, ref_s = run(fp32)
    got_k, got_s = run(int8)

    d = np.linalg.norm(got_k - ref_k, axis=-1)          # canvas pixels
    return {
        "n_crops": int(x.shape[0]),
        "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)),
        "max_px": float(d.max()),
        "score_median_abs_delta": float(np.median(np.abs(got_s - ref_s))),
    }


def quantize(args: argparse.Namespace) -> Path:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    onnx_path = Path(args.onnx_dir) / f"rtmpose-{args.scale}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rtmpose-{args.scale}.int8{args.tag}.qdq.onnx"

    print(f"collecting {args.n_crops} crops from the {args.split} split "
          f"({args.n_clips} clips)", file=sys.stderr, flush=True)
    crops = collect_crops(
        split=args.split,
        frames_root=Path(args.frames_root),
        n_clips=args.n_clips,
        n_crops=args.n_crops,
        seed=args.seed,
        scale=args.scale,
    )
    from cslr_cost.harness.stages.pose import input_hw_for

    h, w = input_hw_for(args.scale)
    if crops[0].shape[:2] != (h, w):
        raise SystemExit(f"crop is {crops[0].shape[:2]}, expected {(h, w)}")
    print(f"  {len(crops)} crops, {crops[0].shape} {crops[0].dtype}", file=sys.stderr)

    name = _input_name(onnx_path)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        # ORT wants shape inference and a cleaned graph before static
        # quantization; skipping it produces silently unquantized nodes.
        prepped = Path(tmp) / "prepped.onnx"
        quant_pre_process(str(onnx_path), str(prepped), skip_symbolic_shape=False)

        quantize_static(
            model_input=str(prepped),
            model_output=str(out),
            calibration_data_reader=CropCalibrationReader(crops, name, args.batch),
            quant_format=QuantFormat.QDQ,
            calibrate_method=getattr(CalibrationMethod, args.calibrate),
            per_channel=True,
            # TensorRT's INT8 layout: quantize the inputs and weights of the
            # compute ops and nothing else. Quantizing LayerNorm constants,
            # residual adds and the SimCC outputs cost accuracy for no speed.
            op_types_to_quantize=args.op_types.split(",") if args.op_types else None,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            # Symmetric on both sides: TensorRT refuses a non-zero zero-point.
            extra_options={
                "AddQDQPairToWeight": True,
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                # TensorRT dequantizes only INT8/FP8/INT4; an INT32 bias fails
                # to parse. Biases stay float and TensorRT fuses them.
                "QuantizeBias": False,
                "CalibPercentile": args.percentile,
            },
        )
    asym = asymmetric_zero_points(out)
    if asym:
        raise SystemExit(
            f"REFUSED: {len(asym)} Q/DQ node(s) carry a non-zero zero-point "
            f"(first: {asym[0]}); TensorRT's explicit quantization would not parse "
            "this graph."
        )
    elapsed = time.perf_counter() - started

    # Parity on crops the calibration did not see.
    held_out = collect_crops(
        split=args.split, frames_root=Path(args.frames_root),
        n_clips=8, n_crops=32, seed=args.seed + 1, scale=args.scale,
    )
    stats = parity(onnx_path, out, held_out)

    manifest = {
        "schema": "int8-export/1",
        "source_onnx": str(onnx_path),
        "scale": args.scale,
        "quant_format": "QDQ",
        "calibrate_method": args.calibrate,
        "calib_percentile": args.percentile if args.calibrate == "Percentile" else None,
        "op_types_to_quantize": args.op_types or "all",
        "quantize_bias": False,
        "per_channel": True,
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "activation_symmetric": True,
        "weight_symmetric": True,
        "calibration_source": "phoenix",
        "calibration_split": args.split,
        "calibration_crops": len(crops),
        "calibration_clips": args.n_clips,
        "input": f"{h}x{w}",
        "export_seconds": round(elapsed, 1),
        "parity_vs_fp32": stats,
    }
    with out.with_suffix(".json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n{out}  ({out.stat().st_size / 2**20:.1f} MB, {elapsed:.0f}s)")
    print(f"  keypoint delta vs fp32 on {stats['n_crops']} held-out crops: "
          f"median {stats['median_px']:.2f} px, p95 {stats['p95_px']:.2f}, "
          f"max {stats['max_px']:.1f}")
    print(f"  score delta: median {stats['score_median_abs_delta']:.4f}")
    if stats["median_px"] > args.max_median_px:
        raise SystemExit(
            f"REFUSED: median keypoint shift {stats['median_px']:.2f} px exceeds "
            f"{args.max_median_px} px. That is not quantization noise — check the "
            "preprocessing against pose.py::_preprocess before trusting this export."
        )
    return out


def asymmetric_zero_points(onnx_path: Path) -> list[str]:
    """Names of Q/DQ nodes whose zero-point is not all zeros.

    TensorRT's explicit quantization accepts only symmetric scales; see the
    module docstring. An absent zero-point input means zero.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=False)
    inits = {t.name: t for t in model.graph.initializer}
    consts = {
        n.output[0]: a.t for n in model.graph.node if n.op_type == "Constant"
        for a in n.attribute if a.name == "value"
    }
    bad = []
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear") or len(node.input) < 3:
            continue
        zp_name = node.input[2]
        tensor = inits.get(zp_name) or consts.get(zp_name)
        if tensor is not None and np.any(numpy_helper.to_array(tensor) != 0):
            bad.append(node.name or zp_name)
    return bad


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cslr_cost.export.quantize_rtmpose_int8",
        description="Calibrate and export an INT8 QDQ RTMPose graph (CPU only).",
    )
    p.add_argument("--scale", default="m", choices=("t", "s", "m"))
    p.add_argument("--split", default="train",
                   help="calibration split; train, always — dev/test would leak")
    p.add_argument("--frames-root", default=str(DEFAULT_FRAMES_ROOT))
    p.add_argument("--onnx-dir", default=str(DEFAULT_ONNX_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--n-crops", type=int, default=DEFAULT_N_CROPS)
    p.add_argument("--n-clips", type=int, default=DEFAULT_N_CLIPS)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibrate", default="Percentile", choices=("Entropy", "Percentile", "MinMax"))
    p.add_argument("--percentile", type=float, default=99.999)
    p.add_argument("--op-types", default="Conv,MatMul,Gemm",
                   help="comma-separated op types to quantize; empty = every supported op")
    p.add_argument("--tag", default="",
                   help="suffix for a trial export: rtmpose-{scale}.int8{tag}.qdq.onnx")
    p.add_argument("--max-median-px", type=float, default=4.0,
                   help="refuse the export above this median shift; a broken "
                        "preprocessing shows up as tens of pixels")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    quantize(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

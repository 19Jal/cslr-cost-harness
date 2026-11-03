"""cli.py — run the harness.

    python -m cslr_cost.harness.cli rails
    python -m cslr_cost.harness.cli ladder --rungs 360p,720p --seconds 60

The `ladder` command runs the whole protocol: for each
resolution rung it runs the full protocol (idle bracket, every stage
alone, the composed pipeline, reconciliation) and then P7's two regimes, and
appends every measurement to `results/results.jsonl`.

--------------------------------------------------------------------------
THE DEFAULTS ARE THE DEFENSIBLE ONES
--------------------------------------------------------------------------
`--seconds` defaults to P1's 60 s floor rather than something convenient. A
shorter window still produces valid *timing* -- and the runner stamps a note on
every row saying it is not valid for energy integration -- but the default must
be the one you would defend, not the one that finishes during a coffee break.

Likewise `--resampled` defaults to true for ladder input, so WER cannot be
attached to a rung by forgetting a flag. The ladder upscales 210x260 source;
accuracy measured there is indefensible, and the schema refuses the row.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

from cslr_cost.harness.runners.sustained import SUSTAINED_MINUTES

def _shared_data() -> Path:
    """Where checkpoints live, on both sides of the container boundary.

    `scripts/dev.sh` mounts the shared corpus tree at the *same absolute path*
    inside the container and exports `CSLR_DATA`. Falling back to
    `Path.home()` is not enough: the container runs as the host uid with no
    home directory of its own, so `Path.home()` is `/` and a `$HOME/datasets`
    default silently resolves to `/datasets` -- which does not exist, and fails
    much later inside a checkpoint loader.
    """
    env = os.environ.get("CSLR_DATA")
    if env:
        return Path(env)
    return Path("data")


DEFAULT_LADDER_ROOT = Path("data/res_ladder/out")
DEFAULT_RESULTS = Path("results/results.jsonl")
DEFAULT_MSKA = Path("MSKA")


def _mmpose_config_dir() -> Path:
    import mmpose

    return (
        Path(os.path.dirname(mmpose.__file__))
        / ".mim/configs/wholebody_2d_keypoint/rtmpose/ubody"
    )


# ---------------------------------------------------------------------------
# rails
# ---------------------------------------------------------------------------

def cmd_rails(args: argparse.Namespace) -> int:
    """What this machine can actually measure, and why not when it cannot."""
    from cslr_cost.harness.energy import auto_meter, describe_rails

    from cslr_cost.harness.device import describe_device

    dev = describe_device()
    if dev["tegra"]:
        print(f"device {dev['device']}  ({dev['model']})")
        print(f"power  {dev['power_mode']}")
        print(f"clocks {dev['clocks']}")
        from cslr_cost.harness.energy.contention import read_nvmap_clients

        try:
            own = os.getpid()
            others = [c for c in read_nvmap_clients() if c.pid != own and c.size_kb > 0]
            if not others:
                print("gpu attribution  nvmap, readable; no other process holds GPU memory\n")
            else:
                # Already holding GPU memory before anything runs, so every row
                # of a grid started now would read SHARED MACHINE. Said here,
                # because the first time it was "3 clients readable" and a whole
                # INT8 grid was measured behind GDM's login screen (2026-10-01).
                held = ", ".join(f"{c.process} ({c.size_kb / 1024:.1f} MB)" for c in others)
                print(f"gpu attribution  nvmap -- WARNING: {len(others)} other process(es) "
                      f"already hold GPU memory: {held}. Every row measured now reads "
                      "SHARED MACHINE. On a Jetson this is usually the desktop: "
                      "`sudo systemctl isolate multi-user.target` (docs/devices.md)\n")
        except Exception as exc:                      # noqa: BLE001
            print(f"gpu attribution  UNAVAILABLE ({exc.__class__.__name__}: {str(exc)[:60]}) "
                  "-- contention will read PARTIALLY CHECKED; see docs/devices.md\n")

    rails = describe_rails()
    if not rails:
        print("no energy rails readable here")
    for name, detail in rails.items():
        print(f"{name:6s} {detail}")

    meter = auto_meter()
    print(f"\nbackend: {meter.backend}")
    if args.idle > 0:
        from cslr_cost.harness.energy import (
            close_idle_baseline,
            measure_idle_baseline,
        )

        print(f"measuring idle ({args.settle:.0f}s settle + {args.idle:.0f}s window, twice)...")
        baseline = measure_idle_baseline(meter, seconds=args.idle, settle_seconds=args.settle)
        baseline = close_idle_baseline(baseline, meter)
        print(baseline.note())
    return 0


# ---------------------------------------------------------------------------
# ladder
# ---------------------------------------------------------------------------

def _mska_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """MSKA's root, config and weights -- required only where they are read.

    The PyTorch head loads both. The TensorRT head reads neither: its vocabulary
    comes from the engine's output shape, and the decoder needs only that size.
    Demanding a 657 MB checkpoint the run never opens is what kept the ladder
    from starting on the Orin, which has the ONNX exports and no MSKA checkout.
    """
    mska = Path(args.mska).resolve()
    config = mska / "configs/phoenix-2014_s2g.yaml"
    weights = mska / "pretrained_models/best.pth"
    if getattr(args, "runtime", "pytorch") == "pytorch":
        for p in (config, weights):
            if not p.exists():
                raise SystemExit(
                    f"missing {p}. The PyTorch head needs MSKA's config and "
                    "weights; run with --mska pointing at the submodule, or use "
                    "--runtime tensorrt, which needs neither."
                )
    return mska, config, weights


def head_precision_for(frontend_precision: str, explicit: Optional[str] = None) -> str:
    """The head's precision in a cell whose front end runs `frontend_precision`.

    INT8 is a front-end precision only: MSKA has no INT8 export, so
    an INT8 cell pairs the INT8 front end with an FP16 head -- the FP16 cell
    with one variable changed, the front end's precision. Rows carry both
    `frontend_precision` and `head_precision`, so the pairing is never implicit.
    """
    if explicit:
        return explicit
    return "fp16" if frontend_precision == "int8" else frontend_precision


def _pose_label(scale: str) -> str:
    """How a pose model is named in a printed banner."""
    return "hrnet-w48-dark" if scale == "hrnet-w48" else f"rtmpose-{scale}"


def _build_rgb_pipeline(args: argparse.Namespace, source: Any):
    """An RGB CSLR model (VAC, VAC + SMKD, CorrNet) in the same seven slots.
    See `stages/rgb_trt.py` for what each slot holds."""
    from cslr_cost.harness.stages import rgb_trt
    from cslr_cost.harness.stages.base import Pipeline
    from cslr_cost.harness.stages.ctc import CtcDecodeStage
    from cslr_cost.harness.stages.decode import VideoDecodeStage
    from cslr_cost.harness.stages.detect import WholeFrameDetectStage

    if args.runtime != "tensorrt":
        raise SystemExit("--rgb-model runs on TensorRT only; use --runtime tensorrt")
    if args.precision not in ("fp32", "fp16"):
        raise SystemExit("--rgb-model supports fp32 and fp16")
    if args.detector != "whole-frame":
        raise SystemExit("an RGB model has no detector; use --detector whole-frame")
    name = args.rgb_model
    onnx_dir = Path(args.rgb_onnx_dir)
    engine_dir = Path(args.engine_dir or f"models/engines/{args.device}")
    if name in ("twostream-slr", "twostream-kp"):
        return _build_twostream_pipeline(args, source, engine_dir)
    feat = (512,) if name in ("vac", "vac-smkd") else (128, 28, 28)
    # the head's sequence includes the 12-16 padding frames
    padded = -(-args.frames // 4) * 4 + 2 * rgb_trt.LEFT_PAD
    # The profile's maximum sizes the engine's activations. 320 covers every PHOENIX
    # clip; CorrNet's FP32 head does not build at 320 on the 8 GB Orin (its
    # correlation affinities are ~0.8 GB each there), so `--rgb-head-max-frames`
    # sizes it to the measured clip instead. The engine hash in the row says which.
    max_frames = args.rgb_head_max_frames or max(320, padded)
    if max_frames < padded:
        raise SystemExit(f"--rgb-head-max-frames {max_frames} is below the padded clip ({padded})")
    head = rgb_trt.TrtRgbHeadStage(onnx_dir / f"{name}.head.onnx", engine_dir, name, feat,
                                   precision=args.precision, opt_frames=padded,
                                   max_frames=max_frames)
    head.setup()
    stages = [
        VideoDecodeStage(accel=args.decode_accel, source=source),
        WholeFrameDetectStage(),
        rgb_trt.RgbResizeCropStage(),
        rgb_trt.TrtRgbFrontendStage(onnx_dir / f"{name}.frontend.onnx", engine_dir, name,
                                    precision=args.precision, opt_batch=max(2, args.batch),
                                    max_batch=max(64, args.batch)),
        rgb_trt.IdentityPostprocStage(),
        head,
        CtcDecodeStage(vocab_size=head.vocab_size, beam_width=args.beam, cutoff_top_n=40),
    ]
    pipeline = Pipeline(stages, assemble=rgb_trt.assemble_features, stride=1, reconstruct=None)
    for stage in stages:
        if stage is not head:
            stage.setup()
    return pipeline, head


def _build_twostream_pipeline(args: argparse.Namespace, source: Any, engine_dir: Path):
    """TwoStream-SLR (RGB + HRNet keypoints), or its keypoint stream alone
    (`twostream-kp`); see `stages/twostream_trt.py`. HRNet runs FP32 with flip test, as
    TwoStream's published keypoints were made; `--precision` is the TwoStream network's."""
    from cslr_cost.harness.stages import twostream_trt as ts
    from cslr_cost.harness.stages.base import Pipeline
    from cslr_cost.harness.stages.ctc import CtcDecodeStage
    from cslr_cost.harness.stages.decode import VideoDecodeStage
    from cslr_cost.harness.stages.detect import WholeFrameDetectStage

    if args.frames % 4:
        raise SystemExit("TwoStream-SLR takes clips of a multiple of 4 frames")
    # as the RGB heads: --rgb-head-max-frames sizes the engine to the measured clip
    # where 300 frames does not fit (the 8 GB Orin)
    max_frames = args.rgb_head_max_frames or max(300, args.frames)
    if max_frames < args.frames:
        raise SystemExit(f"--rgb-head-max-frames {max_frames} is below the clip ({args.frames})")
    kp_only = args.rgb_model == "twostream-kp"
    onnx = Path(args.twostream_kp_onnx if kp_only else args.twostream_onnx)
    head = ts.TrtTwoStreamHeadStage(onnx, engine_dir, precision=args.precision,
                                    opt_frames=args.frames, max_frames=max_frames, keypoint_only=kp_only)
    head.setup()
    stages = [
        VideoDecodeStage(accel=args.decode_accel, source=source),
        WholeFrameDetectStage(),
        ts.TwoStreamCropStage(),
        ts.TwoStreamPoseStage(Path(args.onnx_dir), engine_dir, precision="fp32",
                              opt_batch=max(2, args.batch), max_batch=max(64, args.batch), rgb=not kp_only),
        ts.TwoStreamPostprocStage(frame_wh=source.wh),
        head,
        CtcDecodeStage(vocab_size=head.vocab_size, beam_width=args.beam),
    ]
    pipeline = Pipeline(stages, assemble=ts.assemble_twostream, stride=1, reconstruct=None)
    for stage in stages:
        if stage is not head:
            stage.setup()
    return pipeline, head


def _build_pipeline(args: argparse.Namespace, source: Any):
    if getattr(args, "rgb_model", None):
        return _build_rgb_pipeline(args, source)
    from cslr_cost.harness.stages.pose import crop_size_for
    from cslr_cost.harness.stages.base import Pipeline
    from cslr_cost.harness.stages.crop import TopdownCropStage
    from cslr_cost.harness.stages.ctc import CtcDecodeStage
    from cslr_cost.harness.stages.decode import VideoDecodeStage
    from cslr_cost.harness.stages.detect import (
        RtmdetPersonStage,
        WholeFrameDetectStage,
    )
    from cslr_cost.harness.stages.head import MskaHeadStage
    from cslr_cost.harness.stages.pose import RtmposeStage
    from cslr_cost.harness.stages.postproc import (
        KeypointPostprocStage,
        assemble_sequence,
    )

    mska, config, weights = _mska_paths(args)

    engine_dir = Path(args.engine_dir or f"models/engines/{args.device}")

    if args.detector == "rtmdet" and args.runtime == "tensorrt":
        from cslr_cost.harness.stages.detect_trt import TrtRtmdetPersonStage

        detect = TrtRtmdetPersonStage(
            Path(args.detector_onnx), engine_dir,
            precision=head_precision_for(args.precision),
            opt_batch=max(2, args.batch),
            max_batch=max(64, args.batch),
        )
    elif args.detector == "rtmdet":
        det_dir = Path(args.rtmdet_dir)
        detect = RtmdetPersonStage(
            det_dir / "rtmdet_nano_320-8xb32_coco-person.py",
            det_dir / "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
            precision=args.precision,
        )
    else:
        detect = WholeFrameDetectStage()

    # The head is built first: the decoder needs its vocabulary size, and
    # building it twice would load 657 MB of weights for nothing.
    if args.runtime == "tensorrt":
        # Engines are compiled on this device from the ONNX exports
        # and cached by configuration; the first run of each config pays the
        # build, outside every measurement window. The head profile is tuned at
        # the clip length actually measured, the pose batch profile at the
        # batch actually run -- see engines.py on why opt shapes matter.
        from cslr_cost.harness.stages.head_trt import (
            MSKA_PE_FRAMES,
            TrtMskaHeadStage,
        )
        from cslr_cost.harness.stages.pose_trt import make_trt_pose_stage

        # The ceiling is the model's, not a tuning choice: MSKA's positional
        # encoding is a fixed buffer, so neither runtime can serve a longer
        # clip. Taken from the stage rather than repeated here, and refused
        # rather than clamped -- a silently shortened clip is a wrong
        # measurement, not a smaller one.
        if args.frames > MSKA_PE_FRAMES:
            raise SystemExit(
                f"--frames {args.frames} exceeds MSKA's fixed positional encoding "
                f"({MSKA_PE_FRAMES}); the head cannot process a longer clip in any "
                "runtime. No PHOENIX clip is that long (train max 299)."
            )
        head = TrtMskaHeadStage(
            Path(args.head_onnx), engine_dir,
            precision=head_precision_for(args.precision, args.head_precision),
            opt_frames=args.frames,
        )
        head.setup()
        pose = make_trt_pose_stage(
            Path(args.onnx_dir), args.pose_scale, args.precision, engine_dir,
            opt_batch=max(2, args.batch), max_batch=max(64, args.batch),
        )
    else:
        if args.pose_scale == "hrnet-w48":
            raise SystemExit("hrnet-w48 (MSKA's front end) runs on TensorRT only; use --runtime tensorrt")
        if args.pose_scale == "l":
            raise SystemExit("rtmpose-l (DWPose-l, 384x288) runs on TensorRT only; use --runtime tensorrt")
        if args.precision == "int8":
            raise SystemExit("int8 is a TensorRT engine built from a QDQ export; "
                             "use --runtime tensorrt (there is no int8 PyTorch path)")
        head = MskaHeadStage(mska, config, weights,
                             precision=head_precision_for(args.precision, args.head_precision))
        head.setup()

        cfg_dir = _mmpose_config_dir()
        pose = RtmposeStage(
            cfg_dir / f"rtmpose-{args.pose_scale}_8xb64-270e_coco-ubody-wholebody-256x192.py",
            Path(args.rtmpose_dir) / f"rtmpose-{args.pose_scale}_wholebody_256x192.pth",
            scale=args.pose_scale,
            precision=args.precision,
        )

    stages = [
        # Bound to the opened source so the row reports the decode path that
        # was granted, not the one requested.
        VideoDecodeStage(accel=args.decode_accel, source=source),
        detect,
        TopdownCropStage(input_size=crop_size_for(args.pose_scale)),
        pose,
        # Normalisation divides by the frame's own dimensions, never a
        # hardcoded corpus constant -- a 720p rung's keypoints are in 720p
        # coordinates, and 210x260 would put them far outside [-1, 1].
        KeypointPostprocStage(frame_wh=source.wh),
        head,
        CtcDecodeStage(vocab_size=head.vocab_size, beam_width=args.beam),
    ]
    pipeline = Pipeline(
        stages, assemble=assemble_sequence,
    )
    for stage in stages:
        if stage is not head:
            stage.setup()
    return pipeline, head


def _run_one_rung(args: argparse.Namespace, rung: str, clip: Path, writer) -> dict[str, Any]:
    from cslr_cost.harness.energy import auto_meter
    from cslr_cost.harness.fixtures import FixtureStore
    from cslr_cost.harness.reconcile import reconcile
    from cslr_cost.harness.rows import (
        RunContext,
        row_from_batch,
        row_from_streaming,
        row_from_sustained,
        rows_from_isolation,
    )
    from cslr_cost.harness.energy import ContentionProbe, contention_verdict
    from cslr_cost.harness.runners import (
        measure_batch_insitu,
        measure_isolation_sweep,
        measure_paced_sweep,
        measure_streaming_insitu,
        run_batched,
        run_streaming,
        run_sustained,
    )
    from cslr_cost.harness.stages.decode import open_video

    source = open_video(clip, accel=args.decode_accel)
    n_frames = min(args.frames, source.n_frames) if source.n_frames else args.frames
    pipeline, _head = _build_pipeline(args, source)

    print(f"\n{'=' * 68}\n{rung}  {clip.name}  "
          f"{source.width}x{source.height}  {n_frames} frames  "
          f"{args.rgb_model or _pose_label(args.pose_scale)}/{args.precision}/{args.runtime}"
          f"\n{'=' * 68}")

    ctx = RunContext(
        device=args.device,
        input_resolution=f"{source.width}x{source.height}",
        resampled=args.resampled,
        corpus=args.corpus,
        split=args.split,
        clip_id=clip.stem,
        n_clips=1,
        arm=args.arm,
        device_power_mode=args.power_mode,
        source_resolution=args.source_resolution,
    )

    # --- capture: not a measurement, runs cold, deliberately untimed -------
    source.rewind()
    store = FixtureStore.capture(
        pipeline, source, n_frames,
        provenance={"clip": clip.stem, "rung": rung, "pose": args.rgb_model or args.pose_scale},
    )

    # --- warm-up, outside every window ------------------------------------
    for _ in range(args.warmup_clips):
        source.rewind()
        pipeline.run(source, n_frames)

    # --- P1 ---------------------------------------------------------------
    meter = auto_meter() if args.energy else None
    source.rewind()
    sweep = measure_isolation_sweep(
        pipeline, store, source, n_frames,
        meter=meter,
        idle_seconds=args.idle if args.energy else 0.0,
        idle_settle_seconds=args.settle,
        min_seconds=args.seconds,
        warmup_seconds=args.warmup,
    )
    rec = reconcile(sweep.isolated, sweep.composed, n_frames=n_frames)
    print("--- batch regime, isolation (P1: energy per stage; residual reported, not the gate) ---")
    print(rec.table())
    print(sweep.notes())
    if meter is not None and not sweep.energy_is_defensible:
        print("  ^ over-idle energy from this run is NOT defensible", file=sys.stderr)

    contention = sweep.contention_verdict
    idle_status = None if sweep.idle is None else sweep.idle.status()
    rows = rows_from_isolation(
        sweep.isolated, sweep.composed,
        ctx=ctx, pipeline=pipeline, n_frames=n_frames, reconciliation=rec,
        regime="batch", batch_size=1, contention=contention,
        idle_baseline=idle_status,
    )

    # --- the batch regime's reconcilable decomposition ---------------------
    # Isolation serialises what the pipeline overlaps. On the Orin, FFmpeg's
    # decode threads run ahead while pose is on the GPU, and at FP16 the stages
    # timed alone overshot the composed clip by up to 8.9%. Timing each stage in
    # place is the decomposition that adds up, so it is the gate; isolation still
    # supplies per-stage energy, and its residual is still reported (P1).
    source.rewind()
    insitu_probe = ContentionProbe().start()
    b_insitu, b_insitu_total = measure_batch_insitu(
        pipeline, source, n_frames,
        min_seconds=args.seconds, warmup_seconds=args.warmup,
        contention=insitu_probe,
    )
    insitu_report = insitu_probe.stop()
    batch_rec = reconcile(b_insitu, b_insitu_total, n_frames=n_frames)
    composed_ms = sweep.composed.median_ms
    timing_cost_pct = (
        None if not composed_ms or b_insitu_total.median_ms is None
        else 100.0 * (b_insitu_total.median_ms - composed_ms) / composed_ms
    )
    print("\n--- batch regime, per stage in situ (the gate) ---")
    print(batch_rec.table())
    if timing_cost_pct is not None:
        print(f"  in-situ clip vs composed clip: {timing_cost_pct:+.2f}% (the cost of timing each stage)")
    rows.extend(rows_from_isolation(
        b_insitu, b_insitu_total,
        ctx=ctx, pipeline=pipeline, n_frames=n_frames, reconciliation=batch_rec,
        regime="batch", batch_size=1,
        contention=contention_verdict(insitu_report),
    ))

    # --- P1 again, inside P7's streaming regime --------------------------
    # The exit criterion asks for reconciliation *in each regime*, and the
    # unpaced numbers are not the paced ones: a frame costs ~1.8x more when fed
    # one every 40 ms than back to back.
    paced_rec = None
    if args.paced_isolation:
        source.rewind()
        # One meter across both paced windows: `measure_paced_sweep` closes an
        # idle bracket on it, so the in-situ window that follows inherits the
        # settled idle figure and its row can report over-idle joules rather
        # than totals alone.
        paced_meter = auto_meter() if args.energy else None
        paced, paced_results = measure_paced_sweep(
            pipeline, store, source, n_frames,
            paced_fps=args.paced_fps,
            meter=paced_meter,
            idle_seconds=args.idle if args.energy else 0.0,
            idle_settle_seconds=args.settle,
            min_seconds=args.seconds,
            warmup_seconds=args.warmup,
        )
        paced_contention = paced.contention_verdict

        # The reconcilable decomposition. Paced *isolation* does not sum: each
        # stage alone pays its own post-idle wake-up, and the pipeline pays one
        # per slot. Timing stages in place charges that ramp once, to whichever
        # stage actually triggers it.
        # Its own contention probe: this is a separate window from the sweep
        # above, and a tenant who arrives during it would otherwise be recorded
        # against the wrong one.
        from cslr_cost.harness.energy.contention import ContentionProbe

        insitu_probe = ContentionProbe().start()
        source.rewind()
        insitu, insitu_total, insitu_results = measure_streaming_insitu(
            pipeline, source, n_frames,
            paced_fps=args.paced_fps, min_seconds=args.seconds,
            warmup_clips=1,
            meter=paced_meter,
            contention=insitu_probe,
        )
        insitu_report = insitu_probe.stop()
        insitu_contention = contention_verdict(insitu_report)
        paced_rec = reconcile(insitu, insitu_total, n_frames=n_frames)
        print("\n--- streaming regime, per stage in situ ---")
        print(paced_rec.table())

        rows.extend(rows_from_isolation(
            insitu, insitu_total,
            ctx=ctx, pipeline=pipeline, n_frames=n_frames,
            reconciliation=paced_rec,
            regime="streaming_simulated", batch_size=1,
            paced_fps=args.paced_fps, contention=insitu_contention,
            idle_baseline=None if paced.idle is None else paced.idle.status(),
        ))

        # Paced isolation alongside it: not additive, but it is what says
        # `pose` pays a 14 ms GPU clock ramp after every idle slot.
        iso_rec = reconcile(paced.isolated, paced.composed, n_frames=n_frames)
        print("\n--- streaming regime, paced isolation (NOT additive) ---")
        print(iso_rec.table())
        print(paced.notes())
        rows.extend(rows_from_isolation(
            paced.isolated, paced.composed,
            ctx=ctx, pipeline=pipeline, n_frames=n_frames,
            reconciliation=None,
            regime="streaming_simulated", batch_size=1,
            paced_fps=args.paced_fps, contention=paced_contention,
            idle_baseline=None if paced.idle is None else paced.idle.status(),
        ))
        streaming = insitu_results[-1]
        streaming_idle = None if paced.idle is None else paced.idle.status()
    else:
        source.rewind()
        streaming = run_streaming(pipeline, source, n_frames, paced_fps=args.paced_fps)
        insitu_contention = None
        streaming_idle = None

    print("\n" + streaming.table())
    rows.append(row_from_streaming(
        streaming, ctx=ctx, pipeline=pipeline,
        contention=insitu_contention, idle_baseline=streaming_idle,
    ))

    # P7's batch regime at batch N -- inside an energy window when there is a
    # meter, because the cost table reports joules/video here, and the streaming-vs-
    # batch energy gap is this row against the paced composed row above.
    source.rewind()
    if not getattr(args, "batch_sweep", True):
        print(f"\nbatch-{args.batch} sweep skipped (--no-batch-sweep); engines still sized for it")
    elif args.energy:
        from cslr_cost.harness.runners.batch import measure_batch_sweep

        batch_sweep = measure_batch_sweep(
            pipeline, source, n_frames,
            batch_size=args.batch,
            min_seconds=args.seconds,
            meter=auto_meter(),
            idle_seconds=args.idle,
            idle_settle_seconds=args.settle,
        )
        batched = batch_sweep.last
        print("\n" + batched.table())
        if not batch_sweep.energy_is_defensible:
            print("  ^ batch-regime over-idle energy is NOT defensible", file=sys.stderr)
        rows.append(row_from_batch(
            batched, ctx=ctx, pipeline=pipeline, sweep=batch_sweep,
            contention=batch_sweep.contention_verdict,
            idle_baseline=None if batch_sweep.idle is None else batch_sweep.idle.status(),
        ))
    else:
        batched = run_batched(pipeline, source, n_frames, batch_size=args.batch)
        print("\n" + batched.table())
        rows.append(row_from_batch(batched, ctx=ctx, pipeline=pipeline))

    # --- P3: warm to plateau, then hold ----------------------------------
    # Last, deliberately: it leaves the device hot, and every measurement above
    # wants a device that is merely warm rather than soaked.
    sustained = None
    if args.sustained_minutes > 0:
        sustained = run_sustained(
            pipeline, source, n_frames,
            minutes=args.sustained_minutes,
            paced_fps=args.paced_fps if args.sustained_paced else None,
            meter=auto_meter() if args.energy else None,
            idle_seconds=args.idle if args.energy else 0.0,
            idle_settle_seconds=args.settle,
        )
        print("\n" + sustained.table())
        rows.append(row_from_sustained(
            sustained, ctx=ctx, pipeline=pipeline,
            contention=sustained.contention_verdict,
            idle_baseline=None if sustained.idle is None else sustained.idle.status(),
        ))

    written = writer.write_all(rows)
    print(f"\nwrote {written} rows")

    pipeline.teardown()
    source.close()
    return {
        "rung": rung,
        "clip": clip.stem,
        "residual_pct": batch_rec.residual_pct,
        "passes": batch_rec.passes,
        "iso_residual_pct": rec.residual_pct,
        "timing_cost_pct": timing_cost_pct,
        "paced_residual_pct": None if paced_rec is None else paced_rec.residual_pct,
        "paced_passes": None if paced_rec is None else paced_rec.passes,
        "rows": written,
        "energy_defensible": sweep.energy_is_defensible if meter else None,
        "kept_up": streaming.kept_up,
        "sustained_fps": None if sustained is None else sustained.sustained_fps,
        "throttling": None if sustained is None else sustained.throttling,
    }


def _dev_gpu_slug() -> str:
    """The dev box's historical `--device` value, only when it is the dev box."""
    try:
        import pynvml

        pynvml.nvmlInit()
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        name = name.decode() if isinstance(name, bytes) else name
    except Exception:
        name = ""
    if "RTX 3060" in name:
        return "rtx3060"
    raise SystemExit(
        f"--device auto: not a Jetson, and not the dev box's RTX 3060 (NVML says {name!r}). "
        "Pass --device explicitly rather than let rows claim to be another machine."
    )


def resolve_provenance(args: argparse.Namespace) -> None:
    """Fill `--device` / `--power-mode` from the board, and enforce P4.

    Both used to default to the dev box (`rtx3060`, `default`), so an Orin run
    that forgot the flags would have written rows claiming to be a desktop
    GPU. On a Jetson they are read from the device tree and nvpmodel, and a run
    with dynamic clocks is refused: `jetson_clocks` does not survive a reboot
    while the power mode does, so "right mode, unpinned" is the normal state
    of a board that restarted overnight -- and it moves idle by ~0.8 W.
    """
    from cslr_cost.harness import device

    if device.is_tegra():
        if args.device == "auto":
            args.device = device.device_slug()
            if args.device is None:
                raise SystemExit(
                    f"this Jetson's model ({device.board_model()!r}) was not recognised; "
                    "pass --device explicitly"
                )
        if args.power_mode == "auto":
            mode = device.nvpmodel_mode()
            if mode is None:
                raise SystemExit(
                    "nvpmodel mode unreadable (/var/lib/nvpmodel/status); pass --power-mode explicitly"
                )
            args.power_mode = mode.name
        clocks = device.clock_state()
        if not clocks.pinned:
            if not getattr(args, "allow_unpinned", False):
                raise SystemExit(
                    f"clocks {clocks.detail}.\n"
                    "P4: pin them before measuring -- `sudo jetson_clocks` (it does not "
                    "survive a reboot) -- or pass --allow-unpinned, which marks every "
                    "row's device_power_mode with '+unpinned'."
                )
            args.power_mode = f"{args.power_mode}+unpinned"
        return

    if args.device == "auto":
        args.device = _dev_gpu_slug()
    if args.power_mode == "auto":
        args.power_mode = "default"   # desktop GPU: no power modes to pin


def cmd_ladder(args: argparse.Namespace) -> int:
    from cslr_cost.harness.writer import JsonlWriter

    resolve_provenance(args)
    print(f"device {args.device}  power mode {args.power_mode}")

    root = Path(args.ladder_root)
    if not root.is_dir():
        raise SystemExit(f"ladder root {root} not found (is dataset/ mounted?)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    with JsonlWriter(out) as writer:
        for rung in args.rungs.split(","):
            clips = sorted((root / rung).glob("*.mp4"))[: args.clips]
            if not clips:
                print(f"warning: no clips under {root / rung}", file=sys.stderr)
                continue
            for clip in clips:
                summaries.append(_run_one_rung(args, rung, clip, writer))

    print(f"\n{'=' * 68}\nSUMMARY  ->  {out}\n{'=' * 68}")
    print(f"{'rung':<8}{'clip':<38}{'batch':>9}{'':>6}{'streaming':>11}{'':>6}"
          f"{'isolation':>11}")
    for s in summaries:
        paced = s.get("paced_residual_pct")
        paced_txt = "  n/a" if paced is None else f"{paced:>10.2f}%"
        paced_gate = "" if s.get("paced_passes") is None else (
            "PASS" if s["paced_passes"] else "FAIL")
        iso = s.get("iso_residual_pct")
        iso_txt = "" if iso is None else f"{iso:>10.2f}%"
        print(f"{s['rung']:<8}{s['clip'][:36]:<38}{s['residual_pct']:>8.2f}%"
              f"{'PASS' if s['passes'] else 'FAIL':>6}{paced_txt}{paced_gate:>6}{iso_txt}")
    print("batch and streaming are the in-situ decompositions -- the 5% gate. "
          "isolation is P1's residual, reported, not gating.")

    failed = [s for s in summaries
              if not s["passes"] or s.get("paced_passes") is False]
    if failed:
        print(f"\n{len(failed)}/{len(summaries)} runs outside the 5% gate", file=sys.stderr)
        return 1
    print(f"\nall {len(summaries)} runs within the 5% reconciliation gate")
    return 0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# prebuild
# ---------------------------------------------------------------------------
def cmd_prebuild(args: argparse.Namespace) -> int:
    """Build every engine the grid will need, before any of it is timed.

    Engine construction already sits outside the measurement windows -- it
    happens in `setup()` -- so this changes no number. What it buys is failing
    fast: a missing export or an opset TensorRT cannot parse should surface in
    the first minute of a ten-hour grid, not in its fourth hour, and the first
    configuration of each shape should not be the one that pays a five-minute
    build while the machine is otherwise idle and available.
    """
    from cslr_cost.harness.engines import trt_version
    from cslr_cost.harness.stages.head_trt import TrtMskaHeadStage
    from cslr_cost.harness.stages.pose_trt import make_trt_pose_stage

    version = trt_version()
    if version is None:
        raise SystemExit("TensorRT is not installed in this environment")
    # Same device resolution and clock check as `ladder`. TensorRT picks tactics
    # by timing them during the build, so an engine compiled on dynamic clocks
    # is not guaranteed to be the one a pinned measurement would have chosen --
    # and the engine directory is named after the device, so it must be right.
    resolve_provenance(args)
    engine_dir = Path(args.engine_dir or f"models/engines/{args.device}")
    print(f"TensorRT {version}  ->  {engine_dir}")

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]

    stages: list[tuple[str, Any]] = []
    for precision in precisions:
        hp = head_precision_for(precision)
        stages.append((f"head/{hp}", TrtMskaHeadStage(
            Path(args.head_onnx), engine_dir,
            precision=hp, opt_frames=args.frames)))
        for scale in scales:
            stages.append((f"pose-{scale}/{precision}", make_trt_pose_stage(
                Path(args.onnx_dir), scale, precision, engine_dir,
                opt_batch=max(2, args.batch), max_batch=max(64, args.batch))))
        if args.detector == "rtmdet":
            from cslr_cost.harness.stages.detect_trt import TrtRtmdetPersonStage

            stages.append((f"detect/{hp}", TrtRtmdetPersonStage(
                Path(args.detector_onnx), engine_dir, precision=hp,
                opt_batch=max(2, args.batch), max_batch=max(64, args.batch))))

    built = cached = 0
    for name, stage in stages:
        stage.setup()
        info = stage.engine_info
        if info.built_now:
            built += 1
            print(f"  {name:20} built   {info.build_s:6.1f}s  {info.short_hash}  {info.path.name}")
        else:
            cached += 1
            print(f"  {name:20} cached          {info.short_hash}  {info.path.name}")
        stage.teardown()

    print(f"\n{built} built, {cached} already cached")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("rails", help="what energy rails this machine exposes")
    r.add_argument("--idle", type=float, default=0.0,
                   help="also measure a bracketed idle baseline, seconds per window")
    r.add_argument("--settle", type=float, default=20.0,
                   help="seconds to let clocks wind down before an idle window")
    r.set_defaults(func=cmd_rails)

    l = sub.add_parser("ladder", help="the full per-stage sweep across resolution rungs")
    l.add_argument("--ladder-root", default=str(DEFAULT_LADDER_ROOT))
    l.add_argument("--rungs", default="360p,480p,720p,1080p")
    l.add_argument("--clips", type=int, default=1, help="clips per rung")
    l.add_argument("--frames", type=int, default=128)
    l.add_argument("--out", default=str(DEFAULT_RESULTS))

    l.add_argument("--pose-scale", default="t", choices=("t", "s", "m", "l", "hrnet-w48"),
                   help="DWPose t/s/m/l, or hrnet-w48: MSKA's own front end (HRNet-w48 DARK "
                        "384x288 with flip test; TensorRT only)")
    l.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "int8"),
                   help="the front end's precision; int8 needs --runtime tensorrt")
    l.add_argument("--head-precision", default=None, choices=("fp32", "fp16"),
                   help="default: the front end's, or fp16 when it is int8 (head_precision_for)")
    l.add_argument("--runtime", default="pytorch", choices=("pytorch", "tensorrt"),
                   help="pose + head runtime. tensorrt builds engines from --onnx-dir / "
                        "--head-onnx on first use; detect stays PyTorch")
    l.add_argument("--onnx-dir", default=str(_shared_data() / "onnx"),
                   help="rtmpose-{t,s,m}.onnx")
    l.add_argument("--head-onnx", default="models/onnx/mska-phoenix2014-ensemble.onnx",
                   help="the four-head ensemble export; see stages/head_trt.py")
    l.add_argument("--detector-onnx", default="models/onnx/rtmdet-nano-person-320.onnx",
                   help="RTMDet export, used only with --detector rtmdet --runtime tensorrt")
    l.add_argument("--engine-dir", default=None,
                   help="engine cache (default models/engines/<device>)")
    l.add_argument("--detector", default="whole-frame", choices=("whole-frame", "rtmdet"))
    l.add_argument("--rgb-model", default=None, choices=("vac", "vac-smkd", "corrnet", "twostream-slr", "twostream-kp"),
                   help="measure an RGB CSLR model instead of the keypoint pipeline "
                        "(TensorRT; --precision fp32|fp16; --beam 10 is what it deploys)")
    l.add_argument("--rgb-onnx-dir", default="models/onnx/rgb")
    l.add_argument("--twostream-onnx", default="models/onnx/twostream/twostream-slr.onnx")
    l.add_argument("--twostream-kp-onnx", default="models/onnx/twostream/twostream-kp.onnx",
                   help="TwoStream-SLR's keypoint stream alone (--rgb-model twostream-kp)")
    l.add_argument("--rgb-head-max-frames", type=int, default=None,
                   help="the RGB head engine's longest clip (padded); default 320")
    l.add_argument("--decode-accel", default="software",
                   choices=("software", "hardware", "any"))
    l.add_argument("--beam", type=int, default=5)

    # P1 windows. The default IS the floor, not a convenience.
    l.add_argument("--seconds", type=float, default=60.0,
                   help="isolation window per stage (P1 floor: 60)")
    l.add_argument("--warmup", type=float, default=5.0)
    l.add_argument("--warmup-clips", type=int, default=2)
    l.add_argument("--energy", dest="energy", action="store_true", default=True)
    l.add_argument("--no-energy", dest="energy", action="store_false")
    l.add_argument("--idle", type=float, default=10.0)
    l.add_argument("--settle", type=float, default=20.0)

    # P7
    l.add_argument("--paced-fps", type=float, default=25.0)
    # On by default: "within 5% in each regime" is half the exit criterion, and
    # an opt-in check is one nobody runs.
    l.add_argument("--paced-isolation", dest="paced_isolation",
                   action="store_true", default=True)
    l.add_argument("--no-paced-isolation", dest="paced_isolation",
                   action="store_false",
                   help="skip streaming-regime reconciliation (roughly halves runtime)")
    l.add_argument("--batch", type=int, default=32)
    # Off only where batch N does not fit: TwoStream-SLR on the 8 GB Orin (HRNet-w48 + flip
    # at batch 32 beside its head). --batch still sizes the engines, so they are unchanged.
    l.add_argument("--no-batch-sweep", dest="batch_sweep", action="store_false", default=True,
                   help="skip the batch-N row (P7); batch 1, streaming and the hold still run")

    # P3. Zero skips it; the default is the protocol's window, because a
    # shortened hold quoted as sustained throughput is the error P3 names.
    l.add_argument("--sustained-minutes", type=float, default=SUSTAINED_MINUTES,
                   help="P3 hold after warm-to-plateau (0 to skip)")
    l.add_argument("--sustained-paced", action="store_true", default=False,
                   help="hold at --paced-fps and watch overruns drift as it heats, "
                        "instead of measuring maximum sustained throughput")

    # provenance
    l.add_argument("--device", default="auto",
                   help="auto: read from the board (Jetson) or the dev GPU; never guessed")
    l.add_argument("--power-mode", default="auto",
                   help="auto: the active nvpmodel mode on Jetson, 'default' elsewhere")
    l.add_argument("--allow-unpinned", action="store_true", default=False,
                   help="run on a Jetson with dynamic clocks; rows are marked '+unpinned'")
    l.add_argument("--corpus", default="phoenix-2014")
    l.add_argument("--split", default=None)
    l.add_argument("--arm", default="P", help="free-form label stored in the row (default P: profiling)")
    l.add_argument("--source-resolution", default="210x260")
    l.add_argument("--resampled", dest="resampled", action="store_true", default=True)
    l.add_argument("--not-resampled", dest="resampled", action="store_false",
                   help="native-resolution input; only then may a row carry WER")

    l.add_argument("--mska", default=str(DEFAULT_MSKA))
    l.add_argument("--rtmpose-dir", default=str(_shared_data() / "rtmpose"))
    l.add_argument("--rtmdet-dir", default=str(_shared_data() / "rtmdet"))
    l.set_defaults(func=cmd_ladder)

    b = sub.add_parser("prebuild", help="build every TensorRT engine the grid needs")
    b.add_argument("--scales", default="t,s,m")
    b.add_argument("--precisions", default="fp32,fp16")
    b.add_argument("--frames", type=int, default=128,
                   help="the clip length the head's profile is tuned for; must match the grid's")
    b.add_argument("--batch", type=int, default=32)
    b.add_argument("--detector", default="whole-frame", choices=("whole-frame", "rtmdet"))
    b.add_argument("--device", default="auto")
    b.add_argument("--power-mode", default="auto")
    b.add_argument("--allow-unpinned", action="store_true", default=False)
    b.add_argument("--engine-dir", default=None)
    b.add_argument("--onnx-dir", default=str(_shared_data() / "onnx"))
    b.add_argument("--head-onnx", default="models/onnx/mska-phoenix2014-ensemble.onnx")
    b.add_argument("--detector-onnx", default="models/onnx/rtmdet-nano-person-320.onnx")
    b.set_defaults(func=cmd_prebuild)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

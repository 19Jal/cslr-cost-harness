"""engines.py — TensorRT engines built from ONNX, on the device that runs them.

ONNX is portable; a TensorRT engine is not. It is compiled for one GPU
architecture, one TensorRT version and one CUDA version, so an RTX 3060 engine
(SM 8.6) will not load on an Orin (SM 8.7). Engines are therefore built here,
on first use, from the ONNX files the export step produced ("ONNX -> engine, on each device"), and cached under a key that covers
everything that makes two engines non-interchangeable:

    onnx sha256     a re-export must never silently reuse the old engine
    precision       fp32 / fp16 select different kernels at build time
    tf32            always off -- see below
    profiles        TensorRT tunes kernels for each profile's `opt` shape
    trt version     engines do not load across TensorRT versions
    GPU + SM        nor across architectures

The engine file's own sha256 is what goes into a results row
(`frontend_engine`, `head_engine`). Tactic selection is timing-driven, so
rebuilding the same configuration on the same box can pick different kernels
and produce a different engine. "Which engine produced this row" has to stay
answerable in month four, and the cache key alone does not answer it.

--------------------------------------------------------------------------
"FP32" MEANS FP32: TF32 IS CLEARED
--------------------------------------------------------------------------
On Ampere (the 3060 and the Orin alike) TensorRT enables TF32 by default, as
torch 2.0 does for cuDNN convolutions. TF32 keeps 32-bit storage but does the
matmul/conv arithmetic with 24-bit mantissas, so an "fp32" engine is quietly a
third precision. It is not noise: exporting MSKA's head, the PyTorch reference
disagreed with ONNX Runtime's strict-FP32 CPU path by ~8e-3 in log-probability
with TF32 on, and by ~3e-5 with it off. The FP32 export floor has to
mean export error alone, and the precision axis has to have two values, not
two labels over three behaviours -- so every engine is built with TF32
cleared, and the FP16 arm falls back to FP32, never to TF32.

--------------------------------------------------------------------------
THE MEMORY IS TORCH'S
--------------------------------------------------------------------------
I/O buffers are torch CUDA tensors whose pointers are handed to TensorRT, and
inference is enqueued on torch's current stream. No pycuda, no second
allocator, no host round-trip. The harness's CUDA-event timing and explicit
synchronise keep working unchanged, because to them this is simply more work
on the same stream.

Output tensors are allocated per call rather than reused. Reusing them would
be marginally cheaper, but fixtures captured during a composed pass hold on to
stage outputs, and a reused buffer would be overwritten by the next call --
every captured fixture silently becoming the last frame's.

--------------------------------------------------------------------------
ONE EXECUTION CONTEXT PER OPTIMISATION PROFILE
--------------------------------------------------------------------------
An engine tuned for batch 32 and timed at batch 1 is measuring a kernel choice
nobody would deploy for streaming, and the reverse understates the batch
regime. So a model that serves both regimes is built with one profile per
regime, each profile gets its own execution context, and every call is routed
to the first profile whose [min, max] range admits its shapes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

Shape = tuple[int, ...]

#: One optimisation profile: input name -> (min, opt, max) shapes.
Profile = Mapping[str, tuple[Shape, Shape, Shape]]

PRECISIONS = ("fp32", "fp16", "int8")


class EngineBuildError(RuntimeError):
    """The ONNX file could not be parsed or compiled by this TensorRT."""


class EngineLoadError(RuntimeError):
    """A serialised engine would not deserialise on this device."""


def _trt() -> Any:
    # Deferred: importing the harness must not require TensorRT, which the
    # PyTorch rows and the whole test suite run without.
    #
    # Two import names for one library. JetPack and NVIDIA's `tensorrt`
    # meta-package provide `tensorrt`; the image installs the concrete
    # `tensorrt-cu11-bindings` wheel instead (the meta-package is an sdist that
    # shells out to pip and will not build there), and that wheel's module is
    # `tensorrt_bindings`. Same bindings, same API.
    try:
        import tensorrt
    except ImportError:
        import tensorrt_bindings as tensorrt

    return tensorrt


def trt_version() -> Optional[str]:
    """TensorRT's version, or None when it is not installed."""
    try:
        return str(_trt().__version__)
    except ImportError:
        return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def gpu_tag() -> str:
    """e.g. `nvidia-geforce-rtx-3060-sm86`."""
    name = torch.cuda.get_device_name().lower().replace(" ", "-")
    major, minor = torch.cuda.get_device_capability()
    return f"{name}-sm{major}{minor}"


def _profile_spec(profiles: Sequence[Profile]) -> list[dict[str, list[list[int]]]]:
    return [
        {name: [list(s) for s in shapes] for name, shapes in sorted(p.items())}
        for p in profiles
    ]


@dataclass(frozen=True)
class EngineInfo:
    """What a results row needs to say which engine produced it."""

    path: Path
    sha256: str
    onnx_sha256: str
    precision: str
    trt_version: str
    gpu: str
    profiles: list[dict[str, list[list[int]]]]
    built_now: bool
    build_s: Optional[float]

    @property
    def short_hash(self) -> str:
        return self.sha256[:16]


def _has_qdq_nodes(network: Any) -> bool:
    """Does this parsed network carry explicit quantization nodes?

    TensorRT accepts `BuilderFlag.INT8` on a graph with no QDQ nodes and
    quietly falls back to implicit quantization with no calibrator, producing
    an engine whose scales came from nowhere. That engine runs, and its rows
    would be labelled int8. So the flag is only set when the graph earns it.
    """
    trt = _trt()
    quantize_ops = {trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE}
    return any(network.get_layer(i).type in quantize_ops for i in range(network.num_layers))


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    profiles: Sequence[Profile],
    workspace_mb: int = 4096,
    fp32_layers: Sequence[str] = (),
) -> float:
    """Compile `onnx_path` into `engine_path`. Returns build seconds.

    `fp32_layers`: substrings of layer names to pin to FP32 inside an FP16 or
    INT8 engine (TensorRT names layers after their ONNX nodes). For overflow
    -prone reductions -- see `TrtRtmposeStage.FP32_NORM_LAYERS`. A pattern
    that matches nothing is an error: a pin that silently missed would leave
    the overflow it exists to prevent.
    """
    trt = _trt()
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise EngineBuildError(
            f"TensorRT {trt.__version__} could not parse {onnx_path}: {errors}"
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    config.clear_flag(trt.BuilderFlag.TF32)   # see "FP32 MEANS FP32" above
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise EngineBuildError(
                "this GPU has no fast FP16 path; an fp16 engine here would "
                "measure a fallback, not FP16"
            )
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        # EXPLICIT quantization: the scales come from the QDQ nodes the export
        # already carries (export/quantize_rtmpose_int8.py), so no calibrator
        # and no per-device calibration cache. TensorRT reads the graph.
        #
        # An ONNX without QDQ nodes would still build under this flag and
        # would silently be an implicitly-quantized engine calibrated on
        # nothing -- a different model wearing the same label -- so the graph
        # is checked below rather than trusted.
        if not builder.platform_has_fast_int8:
            raise EngineBuildError(
                "this GPU has no fast INT8 path; an int8 engine here would "
                "measure a fallback, not INT8"
            )
        if not _has_qdq_nodes(network):
            raise EngineBuildError(
                f"{onnx_path.name} carries no QuantizeLinear/DequantizeLinear "
                "nodes, so there is nothing to build an INT8 engine from. "
                "Export it first: python -m "
                "cslr_cost.export.quantize_rtmpose_int8 --scale <t|s|m>"
            )
        config.set_flag(trt.BuilderFlag.INT8)
        # FP16 stays available for layers INT8 cannot serve; the alternative
        # fallback is FP32, which would make an "int8" engine quietly mixed
        # with a much slower precision than the label implies.
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

    if fp32_layers and precision != "fp32":
        hits = {pat: 0 for pat in fp32_layers}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for pat in fp32_layers:
                if pat in layer.name:
                    layer.precision = trt.float32
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.float32)
                    hits[pat] += 1
        missed = [p for p, n in hits.items() if n == 0]
        if missed:
            raise EngineBuildError(f"fp32_layers matched no layer in {onnx_path.name}: {missed}")
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    inputs = {network.get_input(i).name for i in range(network.num_inputs)}
    for prof in profiles:
        missing, unknown = inputs - set(prof), set(prof) - inputs
        if missing or unknown:
            raise EngineBuildError(
                f"profile names {sorted(prof)} do not match the network's inputs "
                f"{sorted(inputs)} (missing {sorted(missing)}, unknown {sorted(unknown)}). "
                "Export pruned an unused input, or the profile is for another model."
            )
        p = builder.create_optimization_profile()
        for name, (lo, opt, hi) in prof.items():
            p.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(p)

    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if blob is None:
        raise EngineBuildError(f"TensorRT failed to build {onnx_path}; see its log above")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = engine_path.with_name(engine_path.name + ".partial")
    tmp.write_bytes(bytes(blob))
    tmp.replace(engine_path)  # never leave a truncated engine under the real name
    return build_s


def ensure_engine(
    onnx_path: Path,
    cache_dir: Path,
    precision: str,
    profiles: Sequence[Profile],
    workspace_mb: int = 4096,
    fp32_layers: Sequence[str] = (),
) -> EngineInfo:
    """Return the cached engine for this exact configuration, building it if absent."""
    version = trt_version()
    if version is None:
        raise EngineBuildError(
            "TensorRT is not installed in this environment. Rebuild the image "
            "(see the TensorRT layer in the Dockerfile)."
        )
    onnx_path = Path(onnx_path)
    onnx_sha = sha256_file(onnx_path)
    spec = {
        "onnx": onnx_sha,
        "precision": precision,
        "tf32": False,
        "profiles": _profile_spec(profiles),
        "trt": version,
        "gpu": gpu_tag(),
    }
    # Only when used, so every engine built without pins keeps its cache key.
    if fp32_layers and precision != "fp32":
        spec["fp32_layers"] = sorted(fp32_layers)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"{onnx_path.stem}.{precision}.{digest}.engine"
    meta = path.with_suffix(".json")

    built_now, build_s = False, None
    if not path.exists():
        build_s = build_engine(onnx_path, path, precision, profiles, workspace_mb, fp32_layers)
        built_now = True

    sha = sha256_file(path)
    if built_now or not meta.exists():
        meta.write_text(json.dumps(
            {**spec, "onnx_path": str(onnx_path), "engine_sha256": sha, "build_s": build_s},
            indent=2,
        ))

    return EngineInfo(
        path=path,
        sha256=sha,
        onnx_sha256=onnx_sha,
        precision=precision,
        trt_version=version,
        gpu=spec["gpu"],
        profiles=spec["profiles"],
        built_now=built_now,
        build_s=build_s,
    )


def _torch_dtype(trt_dtype: Any) -> torch.dtype:
    trt = _trt()
    table = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.uint8: torch.uint8,
        trt.bool: torch.bool,
    }
    try:
        return table[trt_dtype]
    except KeyError:
        raise TypeError(f"no torch dtype for TensorRT dtype {trt_dtype}") from None


class TrtModule:
    """A deserialised engine, callable with torch tensors: `outputs = m(input=x)`."""

    def __init__(self, engine_path: Path, device: str = "cuda") -> None:
        trt = _trt()
        self.path = Path(engine_path)
        self.device = torch.device(device)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise EngineLoadError(
                f"{self.path.name} did not deserialise. Engines are not portable: "
                "it was built by another TensorRT version or for another GPU."
            )

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [n for n in names if n not in self.input_names]

        # One context per profile, created the first time its profile is used,
        # with memory for that profile alone (ON_PROFILE_CHANGE). TensorRT's
        # default (STATIC) gives every context the largest profile's activation
        # memory, so a batch-1 run held memory for the batch profile -- 6.3 GB
        # for HRNet-w48 with flip test on the dev box (2026-10-18). Allocation
        # happens at warm-up, outside every timed window; what runs is unchanged.
        self._contexts: list = [None] * self.engine.num_optimization_profiles
        self._context(0)

    def _context(self, idx: int):
        ctx = self._contexts[idx]
        if ctx is None:
            trt = _trt()
            strategy = getattr(trt, "ExecutionContextAllocationStrategy", None)
            ctx = (self.engine.create_execution_context(strategy.ON_PROFILE_CHANGE)
                   if strategy is not None else self.engine.create_execution_context())
            stream = torch.cuda.current_stream(self.device)
            ctx.set_optimization_profile_async(idx, stream.cuda_stream)
            stream.synchronize()
            self._contexts[idx] = ctx
        return ctx

    def _profile_for(self, shapes: Mapping[str, Shape]) -> int:
        for idx in range(len(self._contexts)):
            fits = True
            for name, shape in shapes.items():
                lo, _opt, hi = self.engine.get_tensor_profile_shape(name, idx)
                if len(lo) != len(shape) or any(
                    s < a or s > b for s, a, b in zip(shape, lo, hi)
                ):
                    fits = False
                    break
            if fits:
                return idx
        raise ValueError(
            f"input shapes {dict(shapes)} fall outside every optimisation profile "
            f"of {self.path.name}"
        )

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        missing = set(self.input_names) - set(inputs)
        if missing:
            raise ValueError(f"{self.path.name} needs inputs {sorted(missing)}")

        shapes = {n: tuple(inputs[n].shape) for n in self.input_names}
        ctx = self._context(self._profile_for(shapes))

        bound: list[torch.Tensor] = []  # keep converted inputs alive until enqueued
        for name in self.input_names:
            t = inputs[name]
            want = _torch_dtype(self.engine.get_tensor_dtype(name))
            if t.device != self.device or t.dtype != want or not t.is_contiguous():
                t = t.to(self.device, want).contiguous()
            bound.append(t)
            if not ctx.set_input_shape(name, tuple(t.shape)):
                raise ValueError(f"TensorRT rejected shape {tuple(t.shape)} for {name}")
            ctx.set_tensor_address(name, t.data_ptr())

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(ctx.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(
                    f"output {name} has a data-dependent shape {shape}; this runner "
                    "only supports shapes computable from the inputs"
                )
            out = torch.empty(
                shape, dtype=_torch_dtype(self.engine.get_tensor_dtype(name)), device=self.device
            )
            ctx.set_tensor_address(name, out.data_ptr())
            outputs[name] = out

        if not ctx.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError(f"TensorRT execution failed for {self.path.name}")
        del bound
        return outputs

    def output_shape(self, name: str) -> Shape:
        """The engine's declared output shape (-1 on dynamic axes)."""
        return tuple(self.engine.get_tensor_shape(name))

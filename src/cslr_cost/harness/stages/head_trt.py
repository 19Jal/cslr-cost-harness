"""head_trt.py — the MSKA head from its ONNX export, run as a TensorRT engine (stage 6).

The deployment form of `head.py`. Same contract: one clip's assembled keypoint
sequence (N=1, C=3, T, V=133) in, ensemble gloss log-scores (N, T//4, vocab)
out, once per clip.

--------------------------------------------------------------------------
WHICH GRAPH: THE FOUR-HEAD ENSEMBLE, NOT THE FUSE HEAD
--------------------------------------------------------------------------
MSKA's reported WER is decoded from `ensemble_last_gloss_logits`: the log of
the sum of all four visual heads' softmax (`recognition.py`, `forward`), and
`head.py` reproduces exactly that. The first MSKA export
(`mska-phoenix2014.onnx`) returns the *fuse* head's logits
alone -- the backbone plus one head of four. It decodes to different glosses,
and it is cheaper by three heads. Measured as "the head", it would understate
the head's share of system joules.

So `export/export_mska_ensemble_onnx.py` exports the backbone and all four
heads. The fuse-only graph still loads, because what the three extra heads cost
is worth one row of its own, but it is labelled `mska-s2g-fuse`, never
`mska-s2g`: the label follows the graph's outputs, not the file name.

--------------------------------------------------------------------------
WHY THE ENSEMBLE ARITHMETIC RUNS OUTSIDE THE ENGINE
--------------------------------------------------------------------------
softmax -> sum -> log is unsafe in an FP16 engine: float16's smallest
subnormal is ~6e-8, so most of a 1,000-class softmax rounds to exactly zero
and its log to -inf, and the beam search then decodes from a distribution with
most of its mass deleted. The engine therefore returns the four logit tensors,
and those three ops run here, in float32, on the GPU -- charged to this stage,
since they are the head's work in any deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from cslr_cost.harness.stages.base import Stage

#: Ensemble export's outputs, in `Recognition.forward`'s summation order.
ENSEMBLE_OUTPUTS = ("logits_left", "logits_right", "logits_body", "logits_fuse")

#: The first export's single output (fuse head only).
FUSE_OUTPUT = "gloss_logits"

#: MSKA consumes the full whole-body layout.
HEAD_JOINTS = 133

#: MSKA cannot process a clip longer than this, in ANY runtime: its positional
#: encoding is a fixed buffer of 400 (`recognition.py`, `PositionalEncoding`),
#: and PyTorch raises at 401 just as an engine would. Measured, not assumed --
#: the PyTorch head raises "size of tensor a (480) must match tensor b (400)".
#: A profile built past it produces a TensorRT warning ("Profile kMAX values
#: are not self-consistent ... 512 != 400") and an engine that cannot serve its
#: own declared maximum, so the cap is enforced here instead.
MSKA_PE_FRAMES = 400

#: Sequence-length profile. TensorRT tunes kernels for `opt`, so it sits at the
#: ladder's clip length (`--frames 128`); min/max bound what an engine accepts.
DEFAULT_MIN_FRAMES = 16
DEFAULT_OPT_FRAMES = 128
DEFAULT_MAX_FRAMES = MSKA_PE_FRAMES


class TrtMskaHeadStage(Stage):
    """MSKA keypoint head as a TensorRT engine."""

    name = "head"
    input_key = "sequence"
    output_key = "gloss_logits"
    granularity = "clip"
    uses_cuda = True

    def __init__(
        self,
        onnx_path: Path,
        engine_dir: Path,
        device: str = "cuda",
        precision: str = "fp32",
        min_frames: int = DEFAULT_MIN_FRAMES,
        opt_frames: int = DEFAULT_OPT_FRAMES,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(
                f"precision must be fp32 or fp16, got {precision!r}. INT8 needs a "
                "TensorRT calibration set, not a builder flag."
            )
        if not min_frames <= opt_frames <= max_frames:
            raise ValueError(f"need min <= opt <= max frames, got {min_frames}, {opt_frames}, {max_frames}")
        if max_frames > MSKA_PE_FRAMES:
            raise ValueError(
                f"max_frames={max_frames} exceeds MSKA's fixed positional encoding "
                f"({MSKA_PE_FRAMES} frames). The engine would build with a maximum it "
                "cannot serve, and TensorRT reports that only as a warning."
            )
        self.onnx_path = Path(onnx_path)
        self.engine_dir = Path(engine_dir)
        self.device = device
        self.precision = precision
        self.min_frames = min_frames
        self.opt_frames = opt_frames
        self.max_frames = max_frames
        self.module = None
        self.engine_info = None
        self._ensemble: Optional[bool] = None
        self._vocab: Optional[int] = None

    def config(self) -> dict[str, Any]:
        info = self.engine_info
        return {
            "head_model": {True: "mska-s2g", False: "mska-s2g-fuse", None: None}[self._ensemble],
            "head_precision": self.precision,
            "head_runtime": "tensorrt",
            "head_engine": None if info is None else info.short_hash,
            "trt_version": None if info is None else info.trt_version,
        }

    @property
    def vocab_size(self) -> Optional[int]:
        return self._vocab

    # -- lifecycle -----------------------------------------------------------

    def setup(self) -> None:
        from cslr_cost.harness.engines import TrtModule, ensure_engine

        profiles = [{
            "keypoint": (
                (1, 3, self.min_frames, HEAD_JOINTS),
                (1, 3, self.opt_frames, HEAD_JOINTS),
                (1, 3, self.max_frames, HEAD_JOINTS),
            ),
        }]
        self.engine_info = ensure_engine(self.onnx_path, self.engine_dir, self.precision, profiles)
        self.module = TrtModule(self.engine_info.path, device=self.device)

        outputs = set(self.module.output_names)
        if outputs == set(ENSEMBLE_OUTPUTS):
            self._ensemble = True
        elif outputs == {FUSE_OUTPUT}:
            self._ensemble = False
        else:
            raise ValueError(
                f"{self.onnx_path.name} emits {sorted(outputs)}; expected the four-head "
                f"ensemble {ENSEMBLE_OUTPUTS} or the fuse-only '{FUSE_OUTPUT}'"
            )
        first = ENSEMBLE_OUTPUTS[0] if self._ensemble else FUSE_OUTPUT
        self._vocab = int(self.module.output_shape(first)[-1])

    def teardown(self) -> None:
        self.module = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- the timed unit ------------------------------------------------------

    @torch.inference_mode()
    def run(self, sequence: Any) -> torch.Tensor:
        """(N=1, C=3, T, V) keypoints -> (N, T//4, vocab) ensemble log-scores."""
        if self.module is None:
            raise RuntimeError("setup() must run before run(); engine loading is not timed")

        kp = sequence if torch.is_tensor(sequence) else torch.as_tensor(sequence)
        if kp.dim() == 3:            # (C, T, V) -> add batch
            kp = kp.unsqueeze(0)
        kp = kp.to(self.device, torch.float32)

        out = self.module(keypoint=kp)
        if not self._ensemble:
            return out[FUSE_OUTPUT]

        # Recognition.forward's ensemble, in float32 (see module docstring).
        probs = None
        for name in ENSEMBLE_OUTPUTS:
            p = out[name].float().softmax(-1)
            probs = p if probs is None else probs + p
        return probs.log()

    @torch.inference_mode()
    def run_all_heads(self, sequence: Any) -> dict[str, torch.Tensor]:
        """The five streams `eval.run` scores, as `MskaHeadStage.run_all_heads`
        returns them: each head's logits, plus the ensemble. Never timed.

        This is what lets WER be measured *through the engine* -- on the Orin,
        or at FP16 -- rather than through the PyTorch head the engine was built
        from."""
        if self.module is None:
            raise RuntimeError("setup() must run before run_all_heads()")
        if not self._ensemble:
            raise RuntimeError(f"{self.onnx_path.name} is the fuse-only export; it has one stream")
        kp = sequence if torch.is_tensor(sequence) else torch.as_tensor(sequence)
        if kp.dim() == 3:
            kp = kp.unsqueeze(0)
        out = self.module(keypoint=kp.to(self.device, torch.float32))
        streams = {name.removeprefix("logits_"): out[name].float() for name in ENSEMBLE_OUTPUTS}
        streams["ensemble_last"] = sum(v.softmax(-1) for v in streams.values()).log()
        return streams

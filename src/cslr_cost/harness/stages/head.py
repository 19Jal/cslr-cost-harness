"""head.py — the MSKA CSLR head (stage 6).

Consumes the assembled keypoint sequence for a clip and emits gloss logits.
Runs **once per clip**, not per frame: `granularity = "clip"`.

--------------------------------------------------------------------------
WHY THIS DOES NOT CALL `Recognition.forward()`
--------------------------------------------------------------------------
MSKA's `Recognition.forward` is a *training* forward. It unconditionally
computes the CTC loss for all four heads:

    outputs['recognition_loss'] = loss_left + loss_right + loss_fuse + loss_body

which requires `src_input['gloss_input']['gloss_labels']` — labels we do not
have at inference, and work a deployed system would never do. Measuring it as
"the head's cost" would inflate the head's share of system energy, which is
precisely the quantity a whole-pipeline account has to get right: the head's
share of the pipeline's cost.

So this adapter runs the inference path only: the DSTA keypoint backbone,
followed by the four visual heads, ensembled exactly as `forward` does. The
ensemble arithmetic is copied deliberately rather than reused, because reusing
`forward` is what would drag the loss back in.

The output matches `ensemble_last_gloss_logits`, which is what
`Recognition.decode` consumes — so the `ctc` stage sees the same tensor it
would in the original pipeline.

--------------------------------------------------------------------------
A 400-FRAME CEILING, AND IT IS THE MODEL'S, NOT THE HARNESS'S
--------------------------------------------------------------------------
MSKA's positional encoding is a fixed buffer of 400, so a clip of 401 frames
raises here (`recognition.py`: "The size of tensor a (480) must match the size
of tensor b (400)"), in PyTorch, before any export is involved. It bounds every
runtime equally — `stages/head_trt.py` refuses to build an engine claiming
more — and it bounds the corpus: a few PHOENIX clips are longer than that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

from cslr_cost.harness.stages.base import Stage


def _add_mska_to_path(mska_root: Path) -> None:
    """MSKA's modules import each other flatly (`from recognition import ...`),
    so its root has to be on sys.path. Prepended rather than appended: several
    of its module names are generic enough to collide."""
    p = str(mska_root.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


class MskaHeadStage(Stage):
    """MSKA keypoint head, inference only."""

    name = "head"
    input_key = "sequence"
    output_key = "gloss_logits"
    granularity = "clip"
    uses_cuda = True

    def __init__(
        self,
        mska_root: Path,
        config_path: Path,
        checkpoint: Optional[Path] = None,
        device: str = "cuda",
        precision: str = "fp32",
    ) -> None:
        self.mska_root = Path(mska_root)
        self.config_path = Path(config_path)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device = device
        self.precision = precision
        self.model = None
        self._vocab: Optional[int] = None

    def config(self) -> dict[str, Any]:
        return {
            "head_model": "mska-s2g",
            "head_precision": self.precision,
            "head_runtime": "pytorch",
        }

    # -- lifecycle -----------------------------------------------------------

    def setup(self) -> None:
        import yaml

        _add_mska_to_path(self.mska_root)
        from recognition import Recognition  # noqa: E402  (needs sys.path first)

        with self.config_path.open() as fh:
            cfg = yaml.safe_load(fh)

        model_cfg = cfg["model"]["RecognitionNetwork"]

        # MSKA resolves these relative to its own root, and its config ships
        # with a Windows absolute path for img_path — irrelevant for the
        # keypoint arm, but the gloss vocabulary must resolve or the head is
        # built with the wrong class count.
        gloss_file = self.mska_root / cfg["gloss"]["gloss2id_file"]
        model_cfg["GlossTokenizer"]["gloss2id_file"] = str(gloss_file)

        if self.checkpoint is not None:
            model_cfg["pretrained_path"] = str(self.checkpoint)

        args = _MinimalArgs()
        self.model = Recognition(cfg=model_cfg, args=args).to(self.device).eval()
        self._vocab = len(self.model.gloss_tokenizer)

        if self.precision == "fp16":
            self.model = self.model.half()

        for p in self.model.parameters():
            p.requires_grad_(False)

    def teardown(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def vocab_size(self) -> Optional[int]:
        return self._vocab

    # -- the timed unit ------------------------------------------------------

    @torch.inference_mode()
    def run(self, sequence: Any) -> torch.Tensor:
        """One clip's keypoints -> ensemble gloss logits.

        `sequence` is the assembled per-clip keypoint tensor, shaped
        (N=1, C=3, T, V): batch, (x, y, confidence), frames, joints.
        """
        if self.model is None:
            raise RuntimeError("setup() must run before run(); model loading is not timed")

        src_input = self._as_src_input(sequence)

        fuse, left, right, body = self.model.visual_backbone_keypoint(src_input)
        mask = src_input["mask"]
        lengths = src_input["new_src_lengths"]

        heads = {
            "body": self.model.body_visual_head(x=body, mask=mask, valid_len_in=lengths),
            "fuse": self.model.fuse_visual_head(x=fuse, mask=mask, valid_len_in=lengths),
            "left": self.model.left_visual_head(x=left, mask=mask, valid_len_in=lengths),
            "right": self.model.right_visual_head(x=right, mask=mask, valid_len_in=lengths),
        }

        # Same ensemble as Recognition.forward: sum the four heads'
        # probabilities, then log. Deliberately reproduced rather than reused,
        # since calling forward() would also compute the training loss.
        ensemble = sum(h["gloss_probabilities"] for h in heads.values()).log()
        return ensemble

    # -- not the timed path --------------------------------------------------

    @torch.inference_mode()
    def run_all_heads(self, sequence: Any) -> dict[str, torch.Tensor]:
        """Every logit stream MSKA's `evaluate` decodes, not just the ensemble.

        **This is not `run`, and nothing timed may call it.** `run` is the
        deployed unit and emits the ensemble alone. This exists
        because MSKA's reported WER is not the ensemble's:
        `train.py:326` decodes *five* streams — `ensemble_last`, `fuse`, `left`,
        `right`, `body` — and reports

            evaluation_results['wer'] = min(wer_results['wer'], ...)

        the minimum over all five, selected on the evaluation set itself. MSKA's
        reported 21.24 is therefore a best-of-five, and reproducing it requires
        knowing which stream won. Returning all five costs four extra CTC
        decodes and no extra backbone work — the four head outputs are already
        computed here to build the ensemble.
        """
        if self.model is None:
            raise RuntimeError("setup() must run before run_all_heads()")

        src_input = self._as_src_input(sequence)

        fuse, left, right, body = self.model.visual_backbone_keypoint(src_input)
        mask = src_input["mask"]
        lengths = src_input["new_src_lengths"]

        heads = {
            "body": self.model.body_visual_head(x=body, mask=mask, valid_len_in=lengths),
            "fuse": self.model.fuse_visual_head(x=fuse, mask=mask, valid_len_in=lengths),
            "left": self.model.left_visual_head(x=left, mask=mask, valid_len_in=lengths),
            "right": self.model.right_visual_head(x=right, mask=mask, valid_len_in=lengths),
        }

        out = {name: h["gloss_logits"] for name, h in heads.items()}
        out["ensemble_last"] = sum(h["gloss_probabilities"] for h in heads.values()).log()
        return out

    # -- input shaping -------------------------------------------------------

    def _as_src_input(self, sequence: Any) -> dict[str, Any]:
        """Build the dict MSKA's backbone expects.

        The backbone reads `keypoint` and the visual heads read `mask` and
        `new_src_lengths`. Note the backbone downsamples time by 4, so the
        head-side lengths are T//4 — getting that wrong produces logits of the
        wrong length and a CTC decode that silently disagrees with the
        reference.
        """
        kp = sequence
        if not torch.is_tensor(kp):
            kp = torch.as_tensor(kp)
        if kp.dim() == 3:            # (C, T, V) -> add batch
            kp = kp.unsqueeze(0)

        kp = kp.to(self.device)
        if self.precision == "fp16":
            kp = kp.half()

        n, _c, t, _v = kp.shape
        out_len = t // 4
        lengths = torch.full((n,), out_len, dtype=torch.long, device=self.device)
        mask = torch.ones(n, 1, out_len, dtype=torch.bool, device=self.device)

        return {"keypoint": kp, "mask": mask, "new_src_lengths": lengths}


class _MinimalArgs:
    """MSKA's modules read a handful of attributes off an argparse Namespace.

    Supplying a small stand-in rather than importing its training argument
    parser, which would pull in wandb, distributed setup and a default config
    path — none of which belong in a measurement harness.
    """

    device = "cuda"
    local_rank = 0
    world_size = 1
    distributed = False
    batch_size = 1

    def __getattr__(self, name: str) -> None:
        return None

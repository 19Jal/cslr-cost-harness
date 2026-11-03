"""ctc.py — CTC beam-search decoding (stage 7).

Consumes the head's gloss logits for a clip and emits a gloss id sequence.
Runs **once per clip**: `granularity = "clip"`.

--------------------------------------------------------------------------
WHY NOT MSKA'S DECODER
--------------------------------------------------------------------------
MSKA decodes with `tf.nn.ctc_beam_search_decoder`. Carrying TensorFlow onto
Jetson and Android for a single beam search is not reasonable, and the
beam-width sweep (beam ∈ {1, 5, 10, 20} against WER and decode
latency) is ours to run. So this uses `ctcdecode`, which the image already
builds.

**That swap must be proven, not assumed.** The 21.24 WER anchor was produced by
the TF decoder; if the replacement decodes differently, every later WER change
is unattributable between "the decoder changed" and whatever was being
measured. `tests/fixtures/ctc_reference.json` holds TF's output on fixed
logits, captured while a TF install still existed, and the parity test checks
against it.

--------------------------------------------------------------------------
TWO BEHAVIOURS THAT MUST BE REPRODUCED EXACTLY
--------------------------------------------------------------------------
1. **Blank rotation is a TF workaround, not semantics.** MSKA's `decode()`
   moves the blank from index 0 to the end before calling TF, then adds 1 back
   to every returned id. `ctcdecode` takes `blank_id=0` directly, so neither
   step is needed — but the *resulting ids must match*, which is what makes the
   fixture worth having.

2. **MSKA collapses repeats a second time.** After CTC's own collapse it
   applies `groupby` to the output sequence:

       [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]

   So a gloss legitimately emitted twice — separated by a blank, which is
   exactly how CTC represents a genuine repetition — is merged anyway. That is
   arguably a bug in MSKA, but it is the behaviour the 21.24 anchor was
   measured with, so reproducing it is required. Decoding without it produces
   repeated glosses MSKA would never emit, and a WER that differs for reasons
   unrelated to anything being studied.
"""

from __future__ import annotations

from itertools import groupby
from typing import Any, Optional

import torch

from cslr_cost.harness.stages.base import Stage

#: Beam widths swept. 5 is what MSKA's reference decoder uses.
BEAM_WIDTHS = (1, 5, 10, 20)
DEFAULT_BEAM_WIDTH = 5


class CtcDecodeStage(Stage):
    """CTC beam search over gloss logits, via `ctcdecode`."""

    name = "ctc"
    input_key = "gloss_logits"
    output_key = "glosses"
    granularity = "clip"
    uses_cuda = False          # ctcdecode is a CPU C++ extension

    def __init__(
        self,
        vocab_size: int,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        blank_id: int = 0,
        num_processes: int = 4,
        collapse_repeats: bool = True,
        cutoff_top_n: Optional[int] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.blank_id = blank_id
        self.num_processes = num_processes
        self.collapse_repeats = collapse_repeats
        #: None: no pruning (MSKA's parity with TF, below). The RGB baselines
        #: decode with ctcdecode's default top 40, as their
        #: repositories do, so their CTC cost is what they deploy.
        self.cutoff_top_n = cutoff_top_n
        self.decoder = None

    def config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"beam_width": self.beam_width}
        if self.cutoff_top_n is not None:
            cfg["notes"] = f"ctc: top-{self.cutoff_top_n} pruning"
        return cfg

    def setup(self) -> None:
        from ctcdecode import CTCBeamDecoder

        # Labels are placeholders: we decode to ids, never to strings, so only
        # the vocabulary *size* matters. cutoff_top_n is set to the full vocab
        # and cutoff_prob to 1.0 to disable pruning — ctcdecode's defaults prune
        # to the top 40 classes, which would make it disagree with TF for
        # reasons that have nothing to do with the beam width being swept.
        self.decoder = CTCBeamDecoder(
            ["_"] * self.vocab_size,
            beam_width=self.beam_width,
            blank_id=self.blank_id,
            num_processes=self.num_processes,
            cutoff_top_n=self.cutoff_top_n or self.vocab_size,
            cutoff_prob=1.0,
            log_probs_input=True,
        )

    def teardown(self) -> None:
        self.decoder = None

    @torch.inference_mode()
    def run(self, gloss_logits: Any) -> list[list[int]]:
        """Logits (B, T, V) -> one gloss id sequence per batch item."""
        if self.decoder is None:
            raise RuntimeError("setup() must run before run(); decoder construction is not timed")

        logits = gloss_logits
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        # ctcdecode wants CPU float32; log_softmax because the head emits
        # unnormalised ensemble logits and the decoder is told log_probs_input.
        log_probs = torch.log_softmax(logits.float().cpu(), dim=-1)
        lengths = torch.full((log_probs.shape[0],), log_probs.shape[1], dtype=torch.int32)

        beams, _scores, _timesteps, out_lens = self.decoder.decode(log_probs, lengths)

        out: list[list[int]] = []
        for b in range(beams.shape[0]):
            best = beams[b][0][: out_lens[b][0]].tolist()
            out.append(self._collapse(best) if self.collapse_repeats else best)
        return out

    @staticmethod
    def _collapse(seq: list[int]) -> list[int]:
        """MSKA's second collapse — see the module docstring."""
        return [k for k, _ in groupby(seq)]

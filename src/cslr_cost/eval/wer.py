"""wer.py — WER with del/sub/ins, scored the way the 21.24 anchor was.

--------------------------------------------------------------------------
THE ANCHOR IS A BEST-OF-FIVE, AND THAT HAS TO BE SAID OUT LOUD
--------------------------------------------------------------------------
MSKA's `train.py:305-322` decodes every `*gloss_logits` stream its
`Recognition.forward` returns -- `ensemble_last`, `fuse`, `left`, `right`,
`body` -- scores each, and reports

    evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])

the **minimum over all five, chosen on the evaluation set itself**. So "21.24"
is not the ensemble's WER; it is the best of five streams, picked with the
labels in hand.

That is not a reason to score differently -- reproducing the anchor means
reproducing how it was produced -- but it is a reason to report all five and
name the winner. The deployed pipeline decodes the ensemble alone (`stages/head.py`), so if
the argmin is some other stream, the number to carry forward is the
ensemble's, and the gap between them belongs in the methods section.

--------------------------------------------------------------------------
SCORING IS MSKA'S OWN, IMPORTED, NOT REIMPLEMENTED
--------------------------------------------------------------------------
`metrics.wer_list` is pure Python Levenshtein -- no sclite -- and
`phoenix_cleanup.clean_phoenix_2014` is the PHOENIX-2014 normalisation applied
to **both** hypothesis and reference before scoring. Reimplementing either
would make a WER difference unattributable between the scorer and the thing
being measured, which is the same trap the ctcdecode swap had to avoid.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The logit streams MSKA's `evaluate` decodes, ensemble first. `head.py`'s
#: `run` -- the deployed, timed path -- emits only `ensemble_last`.
HYPOTHESIS_STREAMS = ("ensemble_last", "fuse", "left", "right", "body")

#: What the deployed pipeline actually decodes.
DEPLOYED_STREAM = "ensemble_last"


def _add_mska_to_path(mska_root: Path) -> None:
    p = str(Path(mska_root).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_mska_metrics(mska_root: Path):
    """Import MSKA's `metrics`, stubbing the one dependency WER does not use.

    `MSKA/metrics.py` is a vendored copy of sacreBLEU 1.4.2 with `wer_list` and
    `wer_single` appended, so importing it drags in sacreBLEU's whole module
    scope -- including `import portalocker`, which the image does not have.
    Portalocker is used at exactly one site (`metrics.py:1703`), inside the
    **test-set download** path, which nothing reachable from `wer_list` calls.

    So the import gets a stub rather than the image getting a rebuild, and
    rather than `wer_list` being copied into this repository. Copying it is the
    thing to avoid: a WER difference would then be unattributable between our
    scorer and the thing being measured, which is the same trap the ctcdecode
    swap had to prove its way out of.
    """
    _add_mska_to_path(mska_root)
    try:
        import portalocker  # noqa: F401
    except ImportError:
        stub = types.ModuleType("portalocker")

        def _refuse(*_args, **_kwargs):
            raise RuntimeError(
                "portalocker is stubbed for WER scoring; this call means "
                "sacreBLEU's download path ran, which WER never reaches."
            )

        stub.Lock = _refuse       # type: ignore[attr-defined]
        sys.modules["portalocker"] = stub

    import metrics                # noqa: E402

    return metrics


def score(
    references: Sequence[str],
    hypotheses: Sequence[str],
    mska_root: Path,
    corpus: str = "phoenix-2014",
) -> dict[str, float]:
    """WER and its decomposition, via MSKA's own scorer.

    Returns `wer`, `del`, `sub`, `ins` as percentages. Never returns the total
    alone: whether a change (quantisation, say) inflates deletions specifically
    is untestable without the decomposition.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"{len(references)} references against {len(hypotheses)} hypotheses"
        )

    metrics = _import_mska_metrics(mska_root)
    from phoenix_cleanup import clean_phoenix_2014    # noqa: E402

    if corpus == "phoenix-2014":
        refs = [clean_phoenix_2014(r) for r in references]
        hyps = [clean_phoenix_2014(h) for h in hypotheses]
    else:
        raise ValueError(
            f"no cleanup rule for corpus {corpus!r}. PHOENIX-2014 has one and it "
            "is applied to both sides; adding a corpus means adding its rule, "
            "not skipping the step."
        )

    res = metrics.wer_list(references=refs, hypotheses=hyps)
    return {
        "wer": float(res["wer"]),
        "del": float(res["del_rate"]),
        "sub": float(res["sub_rate"]),
        "ins": float(res["ins_rate"]),
    }


def score_split(
    references: Mapping[str, str],
    hypotheses_by_stream: Mapping[str, Mapping[str, str]],
    mska_root: Path,
    corpus: str = "phoenix-2014",
    deployed_stream: str = DEPLOYED_STREAM,
) -> dict[str, Any]:
    """Score every decoded stream, and report how MSKA would have reported it.

    `hypotheses_by_stream` maps stream name -> {clip key -> gloss string}.
    Returns per-stream scores plus `best_stream` / `best_wer` (MSKA's reported
    figure) and `deployed` (the ensemble's, which is what we actually ship).
    """
    keys = list(references)
    per_stream: dict[str, dict[str, float]] = {}

    for stream, hyps in hypotheses_by_stream.items():
        missing = [k for k in keys if k not in hyps]
        if missing:
            raise ValueError(
                f"stream {stream!r} is missing {len(missing)} clips, first "
                f"{missing[:3]}. A stream scored on a subset is not comparable "
                "with one scored on the split."
            )
        per_stream[stream] = score(
            references=[references[k] for k in keys],
            hypotheses=[hyps[k] for k in keys],
            mska_root=mska_root,
            corpus=corpus,
        )

    best = min(per_stream, key=lambda s: per_stream[s]["wer"])
    out: dict[str, Any] = {
        "n_clips": len(keys),
        "per_stream": per_stream,
        "best_stream": best,
        "best_wer": per_stream[best]["wer"],
    }
    if deployed_stream in per_stream:
        out["deployed_stream"] = deployed_stream
        out["deployed"] = per_stream[deployed_stream]
    return out


def clip_counts(refs: Sequence[str], hyps: Sequence[str], mska_root: Path):
    """Per clip (err, del, sub, ins, ref_len), after PHOENIX cleanup."""
    import numpy as np

    metrics = _import_mska_metrics(mska_root)
    from phoenix_cleanup import clean_phoenix_2014  # noqa: E402

    out = []
    for r, h in zip(refs, hyps):
        s = metrics.wer_single(r=clean_phoenix_2014(r), h=clean_phoenix_2014(h))
        out.append((s["num_err"], s["num_del"], s["num_sub"], s["num_ins"], s["num_ref"]))
    return np.asarray(out, dtype=np.float64)


def rates(c) -> dict[str, float]:
    """Corpus WER and its del/sub/ins parts, in percent, from `clip_counts`."""
    tot = c.sum(0)
    return {k: 100 * tot[i] / tot[4] for i, k in enumerate(("wer", "del", "sub", "ins"))}

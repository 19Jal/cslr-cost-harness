"""Scoring: the decomposition, the cleanup, and MSKA's best-of-five reporting.

The arithmetic belongs to MSKA and is imported, so what is checked here is that
we call it the way the 21.24 anchor was produced — cleanup on both sides, all
five streams scored, the minimum reported as MSKA's figure while the ensemble
stays identified as the deployed one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cslr_cost.eval.wer import (
    DEPLOYED_STREAM,
    HYPOTHESIS_STREAMS,
    score,
    score_split,
)

MSKA_ROOT = Path(__file__).resolve().parents[2] / "MSKA"

pytestmark = pytest.mark.skipif(
    not (MSKA_ROOT / "metrics.py").exists(), reason="MSKA submodule not checked out"
)


def test_a_perfect_hypothesis_scores_zero():
    got = score(["ICH BIN", "WETTER GUT"], ["ICH BIN", "WETTER GUT"], MSKA_ROOT)
    assert got == {"wer": 0.0, "del": 0.0, "sub": 0.0, "ins": 0.0}


def test_the_decomposition_is_always_present_and_sums_to_the_total():
    """WER is never reported as the total alone: whether a change inflates
    deletions specifically is untestable without the decomposition."""
    got = score(["ICH WETTER GUT MORGEN"], ["ICH REGEN MORGEN"], MSKA_ROOT)
    assert set(got) == {"wer", "del", "sub", "ins"}
    assert got["del"] + got["sub"] + got["ins"] == pytest.approx(got["wer"])


def test_deletions_and_insertions_are_told_apart():
    deleted = score(["ICH WETTER GUT MORGEN"], ["ICH WETTER GUT"], MSKA_ROOT)
    inserted = score(["ICH WETTER GUT"], ["ICH WETTER GUT MORGEN"], MSKA_ROOT)

    assert deleted["del"] > 0 and deleted["ins"] == 0
    assert inserted["ins"] > 0 and inserted["del"] == 0


def test_phoenix_cleanup_is_applied_to_both_sides():
    """`clean_phoenix_2014` strips `loc-`, `__PU__` and friends. Applying it to
    one side only would invent errors that MSKA's own scoring never sees."""
    got = score(["loc-HIER __PU__ WETTER"], ["HIER WETTER"], MSKA_ROOT)
    assert got["wer"] == 0.0


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="against"):
        score(["ICH", "DU"], ["ICH"], MSKA_ROOT)


def test_an_unknown_corpus_is_refused_rather_than_scored_uncleaned():
    """Silently skipping the cleanup step is how a corpus gets scored by a
    different rule than the one its published numbers used."""
    with pytest.raises(ValueError, match="no cleanup rule"):
        score(["ICH"], ["ICH"], MSKA_ROOT, corpus="isharah")


# ---------------------------------------------------------------------------
# score_split — how MSKA reports, and what we deploy
# ---------------------------------------------------------------------------

def streams(**overrides) -> dict[str, dict[str, str]]:
    """Every stream one substitution away from the reference, so a stream that
    is meant to win a test has to be made to win it explicitly.

    The glosses are words, not single letters: `clean_phoenix_2014` reads
    isolated capitals as fingerspelling and joins them ('A B C' -> 'A+B+C'),
    which would silently turn a three-word reference into one token."""
    base = {s: {"c1": "ICH WETTER REGEN", "c2": "MORGEN KALT"} for s in HYPOTHESIS_STREAMS}
    base.update(overrides)
    return base


def test_the_reported_figure_is_the_minimum_over_streams():
    """MSKA's `train.py:322` takes the min across all five decoded streams,
    chosen on the evaluation set itself. Reproducing the anchor means
    reproducing that, and naming it."""
    refs = {"c1": "ICH WETTER GUT", "c2": "MORGEN KALT"}
    hyps = streams(body={"c1": "ICH WETTER GUT", "c2": "MORGEN KALT"},       # perfect
                   fuse={"c1": "ICH WETTER", "c2": "MORGEN KALT"})         # one deletion

    got = score_split(refs, hyps, MSKA_ROOT)

    assert got["best_stream"] == "body"
    assert got["best_wer"] == 0.0
    assert got["per_stream"]["fuse"]["del"] > 0
    assert got["per_stream"]["ensemble_last"]["sub"] > 0


def test_a_tie_goes_to_the_deployed_stream():
    """`min` keeps the first minimum, and HYPOTHESIS_STREAMS puts the ensemble
    first — so when another stream merely equals it, the reported stream is the
    one actually deployed rather than an arbitrary head."""
    refs = {"c1": "ICH WETTER GUT", "c2": "MORGEN KALT"}
    perfect = {"c1": "ICH WETTER GUT", "c2": "MORGEN KALT"}
    hyps = streams(ensemble_last=perfect, body=perfect)

    got = score_split(refs, hyps, MSKA_ROOT)

    assert got["best_stream"] == DEPLOYED_STREAM


def test_the_deployed_stream_is_reported_separately_from_the_best():
    """The pipeline decodes the ensemble alone. If some other
    stream is the argmin, the number to carry forward is still the ensemble's."""
    refs = {"c1": "ICH WETTER GUT", "c2": "MORGEN KALT"}
    hyps = streams(ensemble_last={"c1": "ICH WETTER", "c2": "MORGEN KALT"},
                   body={"c1": "ICH WETTER GUT", "c2": "MORGEN KALT"})

    got = score_split(refs, hyps, MSKA_ROOT)

    assert got["deployed_stream"] == DEPLOYED_STREAM == "ensemble_last"
    assert got["deployed"]["wer"] > got["best_wer"]


def test_a_stream_missing_clips_is_refused():
    """A stream scored on a subset would look better than one scored on the
    split, and the table would compare the two as if they were the same run."""
    refs = {"c1": "ICH WETTER GUT", "c2": "MORGEN KALT"}
    hyps = streams(left={"c1": "ICH WETTER GUT"})

    with pytest.raises(ValueError, match="missing 1 clips"):
        score_split(refs, hyps, MSKA_ROOT)


def test_every_stream_is_scored_and_the_count_recorded():
    refs = {"c1": "ICH WETTER GUT", "c2": "MORGEN KALT"}
    got = score_split(refs, streams(), MSKA_ROOT)

    assert set(got["per_stream"]) == set(HYPOTHESIS_STREAMS)
    assert got["n_clips"] == 2

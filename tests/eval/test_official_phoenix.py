"""The official-protocol scorer's two pieces that run without sclite."""

from cslr_cost.eval.official_phoenix import parse_dtl, write_ctm

DTL = """
Percent Total Error       =   19.8%   (1097)

Percent Correct           =   82.9%   (4594)

Percent Substitution      =   10.2%   ( 567)
Percent Deletions         =    6.8%   ( 378)
Percent Insertions        =    2.7%   ( 152)
Percent Word Accuracy     =   80.2%

Ref. words                =           (5540)
"""


def test_counts_not_rounded_percentages():
    r = parse_dtl(DTL)
    assert r["ref_words"] == 5540 and r["errors"] == 1097
    assert round(r["wer"], 2) == 19.80 and round(r["sub"], 2) == 10.23


def test_an_empty_hypothesis_is_written_as_EMPTY(tmp_path):
    p = tmp_path / "h.ctm"
    write_ctm([{"key": "dev/clipA/1/", "hyp": "A B"}, {"key": "dev/clipB/1/", "hyp": ""}], "hyp", p)
    lines = p.read_text().splitlines()
    assert lines == ["clipA 1 0.000 0.010 A", "clipA 1 0.010 0.020 B", "clipB 1 0.000 0.010 [EMPTY]"]

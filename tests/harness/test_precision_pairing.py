"""An INT8 cell is the FP16 cell with one variable changed: the front end.

MSKA has no INT8 export, so `--precision int8` quantizes the front end only
and the head runs FP16 -- recorded in every row as `head_precision`, never
implied.
"""

from cslr_cost.harness.cli import build_parser, head_precision_for


def test_int8_front_end_pairs_with_an_fp16_head():
    assert head_precision_for("int8") == "fp16"


def test_float_cells_keep_one_precision_throughout():
    assert head_precision_for("fp32") == "fp32"
    assert head_precision_for("fp16") == "fp16"


def test_an_explicit_head_precision_wins():
    assert head_precision_for("int8", "fp32") == "fp32"


def test_the_ladder_accepts_int8_and_a_separate_head_precision():
    args = build_parser().parse_args(["ladder", "--precision", "int8", "--runtime", "tensorrt"])
    assert args.precision == "int8" and args.head_precision is None
    args = build_parser().parse_args(["ladder", "--precision", "int8", "--head-precision", "fp32"])
    assert args.head_precision == "fp32"

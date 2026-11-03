"""The INT8 export must be symmetric, or TensorRT will not parse it.

The first QDQ export passed ONNX Runtime's parity check and then failed to
build on every scale: ONNX Runtime quantizes activations asymmetrically by
default, and TensorRT's explicit quantization accepts only zero zero-points.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from cslr_cost.export.quantize_rtmpose_int8 import asymmetric_zero_points  # noqa: E402


def _qdq_model(tmp_path, zero_point: int, as_constant: bool = False):
    scale = numpy_helper.from_array(np.array(0.1, np.float32), "s")
    zp_arr = np.array(zero_point, np.int8)
    nodes, inits = [], [scale]
    if as_constant:
        nodes.append(helper.make_node("Constant", [], ["zp"], value=numpy_helper.from_array(zp_arr)))
    else:
        inits.append(numpy_helper.from_array(zp_arr, "zp"))
    nodes += [
        helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["q"], name="q0"),
        helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["y"], name="dq0"),
    ]
    g = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=inits,
    )
    path = tmp_path / f"m{zero_point}{as_constant}.onnx"
    onnx.save(helper.make_model(g), str(path))
    return path


def test_symmetric_graph_passes(tmp_path):
    assert asymmetric_zero_points(_qdq_model(tmp_path, 0)) == []


@pytest.mark.parametrize("as_constant", [False, True])
def test_a_nonzero_zero_point_is_named(tmp_path, as_constant):
    assert asymmetric_zero_points(_qdq_model(tmp_path, 3, as_constant)) == ["q0", "dq0"]

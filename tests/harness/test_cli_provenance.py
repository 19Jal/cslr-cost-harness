"""`--device` and `--power-mode` are read from the board, never defaulted.

They used to default to `rtx3060` / `default`. On the Orin that would have
produced rows claiming to be a desktop GPU with no power mode — valid under the
schema, and wrong in the two columns the device comparison is keyed on.
"""

from __future__ import annotations

import argparse

import pytest

from cslr_cost.harness import device
from cslr_cost.harness.cli import build_parser, resolve_provenance


def _args(*extra: str) -> argparse.Namespace:
    return build_parser().parse_args(["ladder", *extra])


@pytest.fixture
def orin(monkeypatch):
    monkeypatch.setattr(device, "is_tegra", lambda *a, **k: True)
    monkeypatch.setattr(device, "device_slug", lambda *a, **k: "orin-nano-8gb-super")
    monkeypatch.setattr(device, "nvpmodel_mode", lambda *a, **k: device.PowerMode(0, "15W"))
    monkeypatch.setattr(device, "clock_state",
                        lambda *a, **k: device.ClockState(True, "pinned (7 clock domains at min == max)"))
    return monkeypatch


def test_defaults_are_auto_not_the_dev_box():
    a = _args()
    assert (a.device, a.power_mode) == ("auto", "auto")


def test_a_jetson_names_itself_and_its_power_mode(orin):
    a = _args()
    resolve_provenance(a)
    assert (a.device, a.power_mode) == ("orin-nano-8gb-super", "15W")


def test_unpinned_clocks_are_refused(orin):
    """The state of a board after a reboot: nvpmodel persists, jetson_clocks does not."""
    orin.setattr(device, "clock_state",
                 lambda *a, **k: device.ClockState(False, "not pinned: gpu 306-612 MHz"))
    with pytest.raises(SystemExit, match="jetson_clocks"):
        resolve_provenance(_args())


def test_allow_unpinned_marks_every_row(orin):
    orin.setattr(device, "clock_state",
                 lambda *a, **k: device.ClockState(False, "not pinned: gpu 306-612 MHz"))
    a = _args("--allow-unpinned")
    resolve_provenance(a)
    assert a.power_mode == "15W+unpinned"


def test_explicit_values_are_kept_but_pinning_is_still_enforced(orin):
    orin.setattr(device, "clock_state",
                 lambda *a, **k: device.ClockState(False, "not pinned: cpu0 729600-1497600 kHz"))
    with pytest.raises(SystemExit):
        resolve_provenance(_args("--device", "orin-nano-8gb", "--power-mode", "15W"))


def test_unrecognised_jetson_refuses_rather_than_guessing(orin):
    orin.setattr(device, "device_slug", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="--device"):
        resolve_provenance(_args())


def test_off_jetson_without_the_dev_gpu_refuses_auto(monkeypatch):
    import sys, types

    monkeypatch.setattr(device, "is_tegra", lambda *a, **k: False)
    fake = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: object(),
        nvmlDeviceGetName=lambda h: "NVIDIA A100",
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    with pytest.raises(SystemExit, match="--device"):
        resolve_provenance(_args())


def test_the_dev_box_keeps_its_historical_labels(monkeypatch):
    """Rows already in results/ say rtx3060/default; auto must not rename them."""
    import sys, types

    monkeypatch.setattr(device, "is_tegra", lambda *a, **k: False)
    fake = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: object(),
        nvmlDeviceGetName=lambda h: b"NVIDIA GeForce RTX 3060",
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    a = _args()
    resolve_provenance(a)
    assert (a.device, a.power_mode) == ("rtx3060", "default")


def test_prebuild_resolves_the_device_and_refuses_unpinned_clocks(orin):
    """Engines are cached under models/engines/<device>, and TensorRT times
    tactics while building -- so the name and the clocks both matter here too."""
    a = build_parser().parse_args(["prebuild"])
    assert a.device == "auto"
    resolve_provenance(a)
    assert a.device == "orin-nano-8gb-super"

    orin.setattr(device, "clock_state",
                 lambda *a, **k: device.ClockState(False, "not pinned: gpu 306-612 MHz"))
    with pytest.raises(SystemExit, match="jetson_clocks"):
        resolve_provenance(build_parser().parse_args(["prebuild"]))


def test_tensorrt_runtime_needs_no_mska_checkout(tmp_path):
    """The Orin has the ONNX exports and no MSKA: the TensorRT head reads its
    vocabulary from the engine, so a missing checkpoint must not stop it."""
    from cslr_cost.harness.cli import _mska_paths

    a = build_parser().parse_args(["ladder", "--runtime", "tensorrt", "--mska", str(tmp_path / "absent")])
    _mska_paths(a)   # no SystemExit


def test_pytorch_runtime_still_requires_mska_weights(tmp_path):
    from cslr_cost.harness.cli import _mska_paths

    a = build_parser().parse_args(["ladder", "--runtime", "pytorch", "--mska", str(tmp_path / "absent")])
    with pytest.raises(SystemExit, match="best.pth|phoenix-2014_s2g.yaml"):
        _mska_paths(a)

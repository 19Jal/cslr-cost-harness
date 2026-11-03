"""The device-memory footprint: it must see allocations PyTorch's allocator
does not (TensorRT's), and say how it was read.

`peak_vram_mb` is PyTorch's allocator high-water mark. An engine's weights and
workspaces are allocated by TensorRT, outside it, so a TensorRT pipeline's
`peak_vram_mb` read 1.3-1.4 MB on the Orin (2026-10-17) -- a meaningless
number. `peak_device_mem_mb` reads the process's footprint from the driver.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cslr_cost.harness.timing import MemoryProbe, device_footprint_mb  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
def test_footprint_names_its_source():
    torch.zeros(1, device="cuda")                    # a CUDA context exists
    mb, source = device_footprint_mb()
    if source is None:
        pytest.skip("neither nvmap nor NVML readable here")
    assert source in ("nvmap-process", "nvml-largest-process")
    assert mb > 0


@cuda
def test_probe_sees_memory_the_torch_allocator_does_not_track():
    """Memory allocated outside PyTorch's allocator (as TensorRT's is) shows in
    the footprint and not in `peak_vram_mb`."""
    if device_footprint_mb()[1] is None:
        pytest.skip("neither nvmap nor NVML readable here")
    probe = MemoryProbe(track_vram=True).start()
    before = device_footprint_mb()[0]
    # a raw CUDA allocation: cudaMalloc through the driver, invisible to the allocator
    import ctypes

    cudart = ctypes.CDLL("libcudart.so")
    ptr = ctypes.c_void_p()
    assert cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(512 << 20)) == 0
    try:
        snap = probe.stop()
    finally:
        cudart.cudaFree(ptr)
    assert snap.device_mem_source is not None
    assert snap.peak_device_mem_mb >= before + 400       # the 512 MB, give or take rounding
    assert snap.peak_vram_mb < 100                       # the allocator never saw it

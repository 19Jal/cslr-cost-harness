"""Loading third-party checkpoints without running their code.

A PyTorch checkpoint is a pickle, and unpickling can execute arbitrary code.
Released CSLR checkpoints are third-party files, so they are loaded through an
unpickler that accepts only the globals a state dict needs and refuses the rest.
"""

from __future__ import annotations

import pickle
import types

#: The globals a plain state dict (tensors, OrderedDicts, NumPy arrays) needs.
SAFE_GLOBALS = {
    ("collections", "OrderedDict"), ("collections", "Counter"),
    ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
    ("torch", "FloatStorage"), ("torch", "LongStorage"), ("torch", "ByteStorage"),
    ("torch", "HalfStorage"), ("torch", "IntStorage"), ("torch", "BoolStorage"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
    ("_codecs", "encode"),
}


def restricted_pickle_module() -> types.ModuleType:
    """A `pickle_module` for `torch.load` that refuses any global not in SAFE_GLOBALS."""

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in SAFE_GLOBALS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"refused global {module}.{name} in a checkpoint")

    mod = types.ModuleType("restricted_pickle")
    mod.Unpickler = Unpickler
    mod.load = lambda f, **kw: Unpickler(f, **kw).load()
    mod.__dict__.update({k: getattr(pickle, k) for k in ("UnpicklingError", "HIGHEST_PROTOCOL")})
    return mod

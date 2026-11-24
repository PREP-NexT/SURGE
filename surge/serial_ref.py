"""Optional reference: ctypes wrapper around CWatM's serial kinematic-wave
kernel (the compiled `t5` routine), used only to validate the GPU schemes against
the exact serial solution.

The compiled library is NOT distributed with this package: it is part of CWatM
(GPL-3.0) and must be built from a CWatM source tree. Point this wrapper at it via
the GPU_KWR_T5_SO environment variable, e.g.

    export GPU_KWR_T5_SO=/path/to/cwatm/hydrological_modules/routing_reservoirs/t5_linux.so

Nothing else in surge depends on this module; the GPU ordered scheme
(TorchRouter.substep_ordered) provides a reference that agrees with it to
single-precision rounding on the cells the paper evaluates, without any external library.
"""
import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        so = os.environ.get("GPU_KWR_T5_SO")
        if not so or not os.path.exists(so):
            raise RuntimeError(
                "CWatM serial kernel not found. Set GPU_KWR_T5_SO to the path of the "
                "compiled t5 library from a CWatM build (GPL-3.0; not bundled here). "
                "The GPU ordered scheme is an equivalent reference if you cannot "
                "build CWatM.")
        _LIB = ctypes.cdll.LoadLibrary(so)
        f64 = ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        i64 = ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        _LIB.kinematic.restype = None
        _LIB.kinematic.argtypes = [f64, f64, i64, i64, i64, f64, f64,
                                   ctypes.c_double, ctypes.c_double, f64, ctypes.c_int]
    return _LIB


def kinematic_serial(qold, sideflow, net):
    """One routing substep, identical to CWatM's serial `kinematic` call."""
    qnew = np.zeros_like(qold)
    _lib().kinematic(np.ascontiguousarray(qold, np.float64),
                     np.ascontiguousarray(sideflow, np.float64),
                     net.dirDown, net.dirupLen, net.dirupID, qnew,
                     net.channelAlpha, net.beta, net.dtRouting,
                     net.chanLength, net.lendirDown)
    return qnew

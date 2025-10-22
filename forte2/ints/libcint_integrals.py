from math import *
import numpy as np
import ctypes

from .libcint_utils import *

try:
    _cint = ctypes.cdll.LoadLibrary("libcint.6.1.1.dylib")
    LIBCINT_AVAILABLE = True
except Exception as e:
    LIBCINT_AVAILABLE = False


def int1e_ovlp(system):
    if not LIBCINT_AVAILABLE:
        raise ImportError("libcint library not available.")

    _cint.CINTcgto_spheric.restype = ctypes.c_int
    _cint.CINTcgto_spheric.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
    ]
    di = _cint.CINTcgto_spheric(0, system.cint_bas)
    dj = _cint.CINTcgto_spheric(1, system.cint_bas)

    _cint.cint1e_ovlp.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.double, ndim=3),
        (ctypes.c_int * 2),
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
    ]
    buf = np.empty((di, dj), order="F")
    _cint.cint1e_ovlp(
        buf,
        (ctypes.c_int * 2)(0, 1),
        system.cint_atm,
        system.cint_atm.shape[0],
        system.cint_bas,
        system.cint_bas.shape[1],
        system.cint_env,
    )
    return buf

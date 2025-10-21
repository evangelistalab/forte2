from math import *
import numpy
import ctypes

from .libcint_utils import atom_basis_to_bas


def int1e_ipnuc_sph(system):
    bas = atom_basis_to_bas(system.atom_basis)
    _cint = ctypes.cdll.LoadLibrary('libcint.6.1.1.dylib')

    _cint.CINTcgto_spheric.restype = ctypes.c_int
    _cint.CINTcgto_spheric.argtypes = [ctypes.c_int, numpy.ctypeslib.ndpointer(dtype=numpy.intc, ndim=2)]
    di = _cint.CINTcgto_spheric(0, bas)
    dj = _cint.CINTcgto_spheric(1, bas)

    _cint.cint1e_ipnuc_sph.argtypes = [
        numpy.ctypeslib.ndpointer(dtype=numpy.double, ndim=3),
        (ctypes.c_int * 2),
        numpy.ctypeslib.ndpointer(dtype=numpy.intc, ndim=2),
        ctypes.c_int,
        numpy.ctypeslib.ndpointer(dtype=numpy.intc, ndim=2),
        ctypes.c_int,
        numpy.ctypeslib.ndpointer(dtype=numpy.double, ndim=1)
    ]
    buf = numpy.empty((di, dj, 3), order='F')
    _cint.cint1e_ipnuc_sph(buf, (ctypes.c_int * 2)(0, 1), atm, atm.shape[0], bas, bas.shape[1], env)
    print(buf)
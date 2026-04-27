from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


@overload
def compute_T1_block(t1: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], ei: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ea: Annotated[NDArray[numpy.float64], dict(shape=(None,))], flow_param: float) -> None:
    """
    Computes the renormalized T1 amplitudes for a given block: T1_renorm = T1 * (1 - exp(-s*denom^2)) / denom
    """

@overload
def compute_T1_block(t1: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))], ei: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ea: Annotated[NDArray[numpy.float64], dict(shape=(None,))], flow_param: float) -> None: ...

@overload
def compute_T2_block(t2: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))], ei: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ej: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ea: Annotated[NDArray[numpy.float64], dict(shape=(None,))], eb: Annotated[NDArray[numpy.float64], dict(shape=(None,))], flow_param: float) -> None:
    """
    Computes the renormalized T2 amplitudes for a given block: T2_renorm = T2 * (1 - exp(-s*denom^2)) / denom
    """

@overload
def compute_T2_block(t2: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))], ei: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ej: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ea: Annotated[NDArray[numpy.float64], dict(shape=(None,))], eb: Annotated[NDArray[numpy.float64], dict(shape=(None,))], flow_param: float) -> None: ...

@overload
def renormalize_V_block(v: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))], ei: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ej: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ea: Annotated[NDArray[numpy.float64], dict(shape=(None,))], eb: Annotated[NDArray[numpy.float64], dict(shape=(None,))], flow_param: float) -> None:
    """
    Renormalizes a block of two-electron integrals: V_renorm = V * (1 + exp(-s*denom^2))
    """

@overload
def renormalize_V_block(v: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))], ei: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ej: Annotated[NDArray[numpy.float64], dict(shape=(None,))], ea: Annotated[NDArray[numpy.float64], dict(shape=(None,))], eb: Annotated[NDArray[numpy.float64], dict(shape=(None,))], flow_param: float) -> None: ...

@overload
def renormalize_3index(v: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None))], ep: float, eq: Annotated[NDArray[numpy.float64], dict(shape=(None,))], er: Annotated[NDArray[numpy.float64], dict(shape=(None,))], es: Annotated[NDArray[numpy.float64], dict(shape=(None,))], flow_param: float) -> None:
    """
    Renormalizes a block of three-index intermediates: V_renorm = V * (1 + exp(-s*denom^2)) * (1 - exp(-s*denom^2)) / denom
    """

@overload
def renormalize_3index(v: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None))], ep: float, eq: Annotated[NDArray[numpy.float64], dict(shape=(None,))], er: Annotated[NDArray[numpy.float64], dict(shape=(None,))], es: Annotated[NDArray[numpy.float64], dict(shape=(None,))], flow_param: float) -> None: ...

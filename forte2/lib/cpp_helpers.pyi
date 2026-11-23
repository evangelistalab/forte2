"""C++ utilities"""

from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def pair_index_geq(arg0: int, arg1: int, /) -> int: ...

def pair_index_gt(arg0: int, arg1: int, /) -> int: ...

def inv_pair_index_gt(arg: int, /) -> tuple[int, int]: ...

def triplet_index_gt(arg0: int, arg1: int, arg2: int, /) -> int: ...

def triplet_index_aab(arg0: int, arg1: int, arg2: int, arg3: int, /) -> int: ...

def triplet_index_abb(arg0: int, arg1: int, arg2: int, arg3: int, /) -> int: ...

@overload
def packed_tensor4_to_tensor4(m: Annotated[NDArray[numpy.float64], dict(shape=(None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]: ...

@overload
def packed_tensor4_to_tensor4(m: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]:
    """Expand a packed 4D tensor stored as a 2D matrix into a full 4D tensor"""

@overload
def tensor4_to_packed_tensor4(t: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def tensor4_to_packed_tensor4(t: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]:
    """
    Pack a 4D tensor antisymmetric in its first and last index pairs into a 2D matrix over the pairs p > q and r > s
    """

@overload
def packed_tensor6_to_tensor6(m: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], n: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]: ...

@overload
def packed_tensor6_to_tensor6(m: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))], n: int) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]:
    """
    Expand a 6D tensor packed over the triplets p > q > r and s > t > u into a full 6D tensor
    """

@overload
def tensor6_to_packed_tensor6(t: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def tensor6_to_packed_tensor6(t: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]:
    """
    Pack a 6D tensor antisymmetric in its first and last index triplets into a 2D matrix over the triplets p > q > r and s > t > u
    """

@overload
def packed_aab_tensor6_to_tensor6(m: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]: ...

@overload
def packed_aab_tensor6_to_tensor6(m: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]:
    """
    Expand an alpha-alpha-beta 6D tensor packed over the pairs p > q and s > t into a full 6D tensor
    """

@overload
def tensor6_to_packed_aab_tensor6(t: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]: ...

@overload
def tensor6_to_packed_aab_tensor6(t: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]:
    """
    Pack an alpha-alpha-beta 6D tensor over the pairs p > q and s > t, giving shape (npair, n, npair, n)
    """

@overload
def packed_abb_tensor6_to_tensor6(m: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]: ...

@overload
def packed_abb_tensor6_to_tensor6(m: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]:
    """
    Expand an alpha-beta-beta 6D tensor packed over the pairs q > r and t > u into a full 6D tensor
    """

@overload
def tensor6_to_packed_abb_tensor6(t: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]: ...

@overload
def tensor6_to_packed_abb_tensor6(t: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]:
    """
    Pack an alpha-beta-beta 6D tensor over the pairs q > r and t > u, giving shape (n, npair, n, npair)
    """

def set_log_level(arg: int, /) -> None:
    """
    Set the logging verbosity level (0=NONE, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG)
    """

def get_log_level() -> int:
    """Get the current logging verbosity level"""

def get_num_threads() -> int: ...

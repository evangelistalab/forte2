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
def packed_tensor4_to_tensor4(m: Annotated[NDArray[numpy.float64], dict(shape=(None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
    """Expand a packed 4D tensor stored as a 2D matrix into a full 4D tensor"""

@overload
def packed_tensor4_to_tensor4(m: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]: ...

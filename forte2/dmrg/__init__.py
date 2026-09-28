import importlib.util
import os
import sys

# forte2 builds block2 itself and vendors it here (see USE_BLOCK2 in
# forte2/CMakeLists.txt), so that it links the same BLAS and OpenMP runtime as
# the rest of forte2. Put that directory on sys.path so `pyblock2` and the
# `block2` extension next to it resolve to this build.
_BLOCK2_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_block2")
if os.path.isdir(_BLOCK2_DIR) and _BLOCK2_DIR not in sys.path:
    sys.path.insert(0, _BLOCK2_DIR)

# find_spec locates pyblock2 without importing it, so `import forte2` does not
# pull in the block2 extension; that happens on the first DMRGDriver instead.
BLOCK2_AVAILABLE = importlib.util.find_spec("pyblock2") is not None

from .dmrg import DMRGSolver, RelDMRGSolver

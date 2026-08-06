try:
    # See forte2/__init__.py for why RTLD_GLOBAL is set process-wide before
    # any extension module (including this one) is imported.
    import pyblock2  # noqa: F401
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes  # noqa: F401

    BLOCK2_AVAILABLE = True
except ImportError:
    BLOCK2_AVAILABLE = False

from .dmrg import DMRG, DMRGSolver, RelDMRG, RelDMRGSolver

__all__ = [
    "DMRG",
    "DMRGSolver",
    "RelDMRG",
    "RelDMRGSolver",
    "BLOCK2_AVAILABLE",
]

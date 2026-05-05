__version__ = "2026.6.4"
__author__ = "Forte2 Developers"

from .integrals import integrals
from .system import System, ModelSystem, HubbardModel
from .state import State, RelState, MOSpace
from .scf import RHF, ROHF, UHF, CUHF, GHF
from .ci import CI, RelCI, CISolver, RelCISolver
from .x2c import x2c
from .orbitals import (
    AVAS,
    CubeGenerator,
    Cube,
    ASET,
    write_orbital_cubes,
    SpinorUpcaster,
)
from .mcopt import MCOptimizer
from .optimize import GeometryOptimizer
from .props import get_1e_property, mulliken_population
from .helpers import logger, set_verbosity_level, comparisons
from .dsrg import (
    DSRG_MRPT2,
    RelDSRG_MRPT2,
    SparseMRDSRG,
    SparseMRDSRG2,
    SparseMRDSRGExcitation,
    WickdDSRG,
    WickdDSRGData,
    enumerate_mrdsrg_excitations,
    solve_sparse_mrdsrg,
    solve_sparse_mrdsrg2,
    solve_sparse_mrdsrg3,
    solve_sparse_mrdsrg4,
    solve_wickd_dsrg,
    solve_wickd_dsrg2,
    solve_wickd_dsrg3,
    solve_wickd_dsrg4,
    wickd_dsrg_data_from_rhf,
)

from .mods_manager import load_mods, enable_mod

# Automatically load any mods in the mods/ or ~/.forte2 directory
load_mods()

import numpy as np
from pathlib import Path
from forte2.orbitals.cube import Cube

from forte2 import System, RHF, MCOptimizer, ASET, CI, State, AVAS
from forte2.helpers.comparisons import approx

xyz = """
F 0.0 0.0 0.0
F 0.0 0.0 1.0
H 2.0 0.0 0.0
H 2.0 0.0 0.74
"""
system = System(
    xyz=xyz,
    basis_set="cc-pVDZ",
    auxiliary_basis_set="def2-universal-JKFIT",
    reorient=False,
)

rhf = RHF(charge=0, econv=1e-12)(system)
avas = AVAS(
    subspace=["F(2p)"],
    selection_method="separate",
    num_active_docc=5,
    num_active_uocc=1,
)(rhf)
aset = ASET(
    fragment=["F"],
    cutoff_method="threshold",
)(avas)
# aset.run()
# cube = Cube()
# cube.run(system, aset.C[0])
mc = MCOptimizer(
    State(nel=20, multiplicity=1, ms=0.0),
    optimize_frozen_orbs=False,
    gconv=2e-5,
)(aset)
ci = CI(State(system=system, multiplicity=1, ms=0.0))(mc)
ci.run()
# cube = Cube()
# cube.run(system, ci.C[0])

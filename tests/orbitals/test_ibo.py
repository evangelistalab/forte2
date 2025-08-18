import numpy as np
import pytest

from forte2 import System, RHF
from forte2.orbitals.iao import IBO
from forte2.helpers.comparisons import approx


def test_ibo_water():
    xyz = """
    O
    H 1 1.1
    H 1 1.1 2 104.5
    """

    system = System(xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    C_occ = rhf.C[0][:, : rhf.ndocc]
    ibo = IBO(system, C_occ)
    D_ibo = np.einsum("pi,qi->pq", ibo.C_ibo, ibo.C_ibo)
    # IBO should be an equivalent representation of the occupied orbitals
    E = np.einsum("pq,pq->", D_ibo, rhf.F[0] + system.ints_hcore())

    assert E + system.nuclear_repulsion == approx(rhf.E)

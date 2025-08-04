import numpy as np
import pytest

from forte2 import System, RHF
from forte2.orbitals.iao import IBO
from forte2.helpers.comparisons import approx


@pytest.mark.slow
def test_ibo_acrylic_acid():
    # geometry from http://www.iboview.org/bin/ibo-ref.20141030.tar.bz2
    xyz = """
    C                 1.9750101       -0.0359056        0.0000000
    C                 0.8155445        0.6290773        0.0000000
    C                -0.4769136       -0.0990005        0.0000000
    O                -0.6246729       -1.3065879        0.0000000
    O                -1.5290139        0.7740207        0.0000000
    H                 1.9795702       -1.1269710        0.0000000
    H                 2.9318898        0.4848390        0.0000000
    H                 0.7696874        1.7186370        0.0000000
    H                -2.3334997        0.2190058        0.0000000
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

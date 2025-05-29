import numpy as np

import forte2

from forte2.scf import RHF
from forte2.jkbuilder.jkbuilder import DFFockBuilder
from forte2 import ints


def parse_state(state: dict) -> tuple:
    if "nel" not in state:
        raise KeyError(
            "The state dictionary must contain the number of electrons ('nel')"
        )
    nel = state["nel"]

    if "multiplicity" in state:
        multiplicity = state["multiplicity"]
    elif "multp" in state:
        multiplicity = state["multp"]
    else:
        raise KeyError(
            "The state dictionary must contain either 'multiplicity' or 'multp'"
        )

    if "ms" not in state:
        raise KeyError("The state dictionary must contain 'ms'")
    ms = state["ms"]
    twoms = round(ms * 2)

    # nel = na + nb
    # 2 ms = na - nb
    na = (nel + twoms) // 2
    nb = nel - na

    return (na, nb, multiplicity, twoms)


class Integrals:
    def __init__(self, method, orbitals, core=None):
        self.orbitals = orbitals
        self.method = method
        self.core = core

    def run(self, system):
        jkbuilder = DFFockBuilder(system)
        C = self.method.C[0][:, self.orbitals]

        self.basis = system.basis
        T = ints.kinetic(system.basis, system.basis)
        V = ints.nuclear(system.basis, system.basis, system.atoms)
        self.H = np.einsum("mi,mn,nj->ij", C, T + V, C)
        if self.core:
            Ccore = self.method.C[0][:, self.core]
            J, K = jkbuilder.build_JK([Ccore])
            Jcore = J[0]
            Kcore = K[0]
            self.Ecore = forte2.ints.nuclear_repulsion(system.atoms)
            self.Ecore += 2.0 * np.einsum("mi,mn,ni->", Ccore, T + V, Ccore)
            self.Ecore += np.einsum("mi,mn,ni->", Ccore, 2 * Jcore - Kcore, Ccore)

            self.H += np.einsum("mi,mn,nj->ij", C, 2 * Jcore - Kcore, C)

        self.V = jkbuilder.two_electron_integrals_block(C)


class SelectedCI:
    def __init__(self, method, orbitals, core, state, nroot):
        self.method = method
        self.orbitals = orbitals
        self.core = core
        self.norb = len(self.orbitals)
        na, nb, multiplicity, twoms = parse_state(state)
        self.na = na
        self.nb = nb
        self.multiplicity = multiplicity
        self.twoms = twoms
        self.nroot = nroot

    def run(self, system):
        print("\nSelected configuration interaction")

        state = forte2.SparseState()
        dets = forte2.hilbert_space(self.norb, self.na, self.nb)
        ints = Integrals(self.method, self.orbitals, self.core)
        ints.run(system)
        print(ints.Ecore + 2 * ints.H[0, 0] + ints.V[0, 0, 0, 0])

        H = forte2.sparse_operator_hamiltonian(self.norb, ints.Ecore, ints.H, ints.V)
        Hmat = H.matrix(dets)
        print("Hamiltonian matrix size:", Hmat)
        eig = np.linalg.eigh(Hmat)[0]
        print("Eigenvalues:", eig)


def test_sci1():
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.econv = 1e-12
    scf.run()

    sci = SelectedCI(
        method=scf,
        orbitals=[4, 5],
        core=range(4),
        state={"nel": 2, "multiplicity": 1, "ms": 0.0},
        nroot=1,
    )
    sci.run(system)


if __name__ == "__main__":
    test_sci1()

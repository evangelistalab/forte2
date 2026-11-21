import numpy as np
import pytest

from forte2 import CI, CISolver, GHF, RelCISolver, RHF, State, System
from forte2.base_classes import X2CParams
from forte2.helpers import random_unitary
from forte2.helpers.comparisons import approx, is_diagonal_matrix
from forte2.orbitals import NaturalOrbitals
from forte2.orbitals.ci_rotation import rotate_ci_vectors


def _water(x2c=None):
    xyz = """
    O 0.0 0.0 0.0
    H 0.0  0.757 0.587
    H 0.0 -0.757 0.587
    """
    return System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=x2c,
    )


def _gas_natural_orbitals(ci):
    """Active natural-orbital rotation of the ground state, per GAS space."""
    space = ci.mo_space
    natural = NaturalOrbitals(space)
    natural.make_natural_orbitals(
        g1_act=ci.make_rdm(0, order=1, spin_type="sf"),
        C_contig=ci.mos.C[0][:, space.orig_to_contig],
    )
    return natural.Uactv


def test_rotate_ci_vectors_matches_resolve():
    """
    Rotating the CI vectors of a converged CASCI gives the same wavefunctions
    as solving it again in the rotated orbitals, for every root of every
    state. The rotations are the ground-state natural orbitals, a signed
    permutation, and a random orthogonal matrix.
    """
    system = _water()
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        states=[
            State(nel=10, multiplicity=1, ms=0.0),
            State(nel=10, multiplicity=3, ms=1.0),
        ],
        nroots=[2, 1],
        core_orbitals=[0, 1, 2],
        active_orbitals=[3, 4, 5, 6],
    )
    ci = CI(ci_solver=ci_solver)(rhf)
    ci.run()
    C = rhf.mos.C[0].copy()

    _, U_natural = np.linalg.eigh(ci.make_rdm(0, order=1, spin_type="sf"))
    U_natural = U_natural[:, ::-1]
    U_permutation = np.eye(4)[:, [2, 0, 3, 1]] * [1.0, -1.0, -1.0, 1.0]
    U_random = random_unitary(
        4, cmplx=False, rng=np.random.default_rng(3), rotation=False
    )

    for label, U in (
        ("natural", U_natural),
        ("permutation", U_permutation),
        ("random", U_random),
    ):
        rotated = rotate_ci_vectors(ci_solver, U)

        C_rotated = C.copy()
        C_rotated[:, 3:7] = C[:, 3:7] @ U
        rhf.mos.C[0] = C_rotated
        ci_solver_resolved = CISolver(
            states=[
                State(nel=10, multiplicity=1, ms=0.0),
                State(nel=10, multiplicity=3, ms=1.0),
            ],
            nroots=[2, 1],
            core_orbitals=[0, 1, 2],
            active_orbitals=[3, 4, 5, 6],
        )
        ci_resolved = CI(ci_solver=ci_solver_resolved)(rhf)
        ci_resolved.run()

        for state, sub_solver in enumerate(ci_solver_resolved.sub_solvers):
            for root in range(sub_solver.nroot):
                overlap = sub_solver.evecs[:, root] @ rotated[state][:, root]
                assert abs(overlap) == approx(
                    1.0
                ), f"{label}, state {state}, root {root}"
        if label == "natural":
            g1 = ci_resolved.make_rdm(0, order=1, spin_type="sf")
            assert is_diagonal_matrix(g1)


def test_rotate_ci_vectors_two_component_matches_resolve():
    """
    Two-component version of the comparison with a re-solve, for the ground
    state and complex rotations.
    """
    x2c = X2CParams(x2c_type="so", x2c_model="1e")
    system = _water(x2c=x2c)
    ghf = GHF(charge=0, e_tol=1e-12)(system)
    ci_solver = RelCISolver(
        nel=10, core_orbitals=list(range(6)), active_orbitals=list(range(6, 14))
    )
    ci = CI(ci_solver=ci_solver)(ghf)
    ci.run()
    C = ghf.mos.C[0].copy()

    _, U_natural = np.linalg.eigh(ci.make_rdm(0, order=1, spin_type="so"))
    U_natural = U_natural[:, ::-1]
    phases = np.exp(1j * np.linspace(0.3, 2.9, 8))
    U_permutation = np.eye(8)[:, [3, 0, 5, 1, 7, 2, 6, 4]] * phases
    U_random = random_unitary(
        8, cmplx=True, rng=np.random.default_rng(3), rotation=False
    )

    for label, U in (
        ("natural", U_natural),
        ("permutation", U_permutation),
        ("random", U_random),
    ):
        rotated = rotate_ci_vectors(ci_solver, U)

        C_rotated = C.copy()
        C_rotated[:, 6:14] = C[:, 6:14] @ U
        ghf.mos.C[0] = C_rotated
        ci_solver_resolved = RelCISolver(
            nel=10, core_orbitals=list(range(6)), active_orbitals=list(range(6, 14))
        )
        ci_resolved = CI(ci_solver=ci_solver_resolved)(ghf)
        ci_resolved.run()

        resolved = ci_solver_resolved.sub_solvers[0].evecs[:, 0]
        overlap = np.vdot(resolved, rotated[0][:, 0])
        assert abs(overlap) == approx(1.0), label


def test_rotate_ci_vectors_gas():
    """
    A rotation within each GAS space keeps a GASCI space closed, so natural
    orbitals computed space by space rotate its CI vector. A rotation that
    couples the spaces raises.
    """
    system = _water()
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[3, 0], gas_max=[4, 1]),
        core_orbitals=[0, 1, 2],
        active_orbitals=[[3, 4], [5, 6]],
    )
    ci = CI(ci_solver=ci_solver)(rhf)
    ci.run()

    U = _gas_natural_orbitals(ci)
    rotated = rotate_ci_vectors(ci_solver, U)

    C_rotated = rhf.mos.C[0].copy()
    C_rotated[:, 3:7] = C_rotated[:, 3:7] @ U
    rhf.mos.C[0] = C_rotated
    ci_solver_resolved = CISolver(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[3, 0], gas_max=[4, 1]),
        core_orbitals=[0, 1, 2],
        active_orbitals=[[3, 4], [5, 6]],
    )
    ci_resolved = CI(ci_solver=ci_solver_resolved)(rhf)
    ci_resolved.run()

    resolved = ci_solver_resolved.sub_solvers[0].evecs[:, 0]
    assert abs(resolved @ rotated[0][:, 0]) == approx(1.0)
    U_coupling = random_unitary(4, cmplx=False, rng=np.random.default_rng(3))
    with pytest.raises(ValueError):
        rotate_ci_vectors(ci_solver, U_coupling)

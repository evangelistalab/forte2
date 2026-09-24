import numpy as np
import pytest

from forte2 import CI, CISolver, GHF, MCOptimizer, RelCISolver, RHF, State, System
from forte2.base_classes import X2CParams
from forte2.helpers import random_unitary
from forte2.helpers.comparisons import approx, approx_abs
from forte2.orbitals import ci_overlap, mo_overlap


def _water(scale, x2c=None):
    xyz = f"""
    O 0.0 0.0 0.0
    H 0.0  {0.757 * scale} {0.587 * scale}
    H 0.0 {-0.757 * scale} {0.587 * scale}
    """
    return System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=x2c,
    )


def test_ci_overlap_invariant_under_orbital_rotations():
    """
    Rotations within the core and within the active space leave a CASCI
    wavefunction unchanged, so its overlap with the rotated representation is
    1 in magnitude. With two active electrons of each spin, off-diagonal
    determinant pairs contribute, which pins the determinant sign convention.
    """
    system = _water(1.0)
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver_1 = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2],
        active_orbitals=[3, 4, 5, 6],
    )
    ci_1 = CI(ci_solver=ci_solver_1)(rhf)
    ci_1.run()

    rng = np.random.default_rng(7)
    C = rhf.mos.C[0].copy()
    for block in (slice(0, 3), slice(3, 7)):
        n = block.stop - block.start
        U = random_unitary(n, cmplx=False, rng=rng, rotation=False)
        C[:, block] = C[:, block] @ U
    rhf.mos.C[0] = C

    ci_solver_2 = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2],
        active_orbitals=[3, 4, 5, 6],
    )
    ci_2 = CI(ci_solver=ci_solver_2)(rhf)
    ci_2.run()

    assert ci_2.E_ci[0] == approx(ci_1.E_ci[0])
    assert abs(ci_overlap(ci_1, ci_2)) == approx(1.0)


def test_ci_overlap_displaced_casscf():
    """CASSCF(4,4) ground states of water at two O-H bond lengths."""
    mcs = []
    for scale in (1.0, 1.1):
        system = _water(scale)
        rhf = RHF(charge=0, e_tol=1e-12)(system)
        ci_solver = CISolver(
            State(nel=10, multiplicity=1, ms=0.0),
            core_orbitals=[0, 1, 2],
            active_orbitals=[3, 4, 5, 6],
        )
        mc = MCOptimizer(ci_solver=ci_solver)(rhf)
        mc.run()
        mcs.append(mc)

    S_12 = ci_overlap(mcs[0], mcs[1])
    S_21 = ci_overlap(mcs[1], mcs[0])
    assert S_12 == approx(S_21)
    # the overlap is first order in the MCSCF convergence error
    assert abs(S_12) == approx_abs(0.989052518228039, 1e-6)


def test_ci_overlap_single_determinant():
    """
    A closed-shell single determinant has overlap det(S_occ)^2, however its
    occupied orbitals are split between core and active.
    """
    rhfs = []
    for scale in (1.0, 1.1):
        system = _water(scale)
        rhfs.append(RHF(charge=0, e_tol=1e-12)(system))

    for core, active in (
        ([0, 1, 2, 3, 4], []),
        ([0, 1, 2, 3], [4]),
        ([], [0, 1, 2, 3, 4]),
    ):
        cis = []
        for rhf in rhfs:
            ci_solver = CISolver(
                State(nel=10, multiplicity=1, ms=0.0),
                core_orbitals=core,
                active_orbitals=active,
            )
            ci = CI(ci_solver=ci_solver)(rhf)
            ci.run()
            cis.append(ci)

        S_occ = mo_overlap(
            rhfs[0].C[0][:, :5], rhfs[0].system, rhfs[1].C[0][:, :5], rhfs[1].system
        )
        expected = np.linalg.det(S_occ) ** 2
        S = ci_overlap(*cis)
        assert abs(S) == approx(expected), f"core={core}, active={active}"


def test_ci_overlap_two_component_invariant_under_orbital_rotations():
    """
    Two-component version of the rotation test. Spin-orbit coupling makes the
    CI vectors complex, and a complex core rotation gives the overlap a phase.
    """
    x2c = X2CParams(x2c_type="so", x2c_model="1e")
    system = _water(1.0, x2c=x2c)
    ghf = GHF(charge=0, e_tol=1e-12)(system)
    ci_solver_1 = RelCISolver(
        nel=10, core_orbitals=list(range(6)), active_orbitals=list(range(6, 14))
    )
    ci_1 = CI(ci_solver=ci_solver_1)(ghf)
    ci_1.run()

    rng = np.random.default_rng(7)
    C = ghf.mos.C[0].copy()
    for block in (slice(0, 6), slice(6, 14)):
        n = block.stop - block.start
        U = random_unitary(n, cmplx=True, rng=rng, rotation=False)
        C[:, block] = C[:, block] @ U
    ghf.mos.C[0] = C

    ci_solver_2 = RelCISolver(
        nel=10, core_orbitals=list(range(6)), active_orbitals=list(range(6, 14))
    )
    ci_2 = CI(ci_solver=ci_solver_2)(ghf)
    ci_2.run()

    assert ci_2.E_ci[0] == approx(ci_1.E_ci[0])
    assert abs(ci_overlap(ci_1, ci_2)) == approx(1.0)


def test_ci_overlap_two_component_matches_nonrelativistic():
    """
    Without spin-orbit coupling, CASCI in 8 spinors is the nonrelativistic
    CAS(4,4) wavefunction, so both give the same overlap between geometries.
    """
    cis_1c, cis_2c = [], []
    for scale in (1.0, 1.1):
        system_1c = _water(scale)
        rhf = RHF(charge=0, e_tol=1e-12)(system_1c)
        ci_solver_1c = CISolver(
            State(nel=10, multiplicity=1, ms=0.0),
            core_orbitals=[0, 1, 2],
            active_orbitals=[3, 4, 5, 6],
        )
        ci_1c = CI(ci_solver=ci_solver_1c)(rhf)
        ci_1c.run()
        cis_1c.append(ci_1c)

        # GHF sets two_component on its System, so it can't share system_1c
        system_2c = _water(scale)
        ghf = GHF(charge=0, e_tol=1e-12)(system_2c)
        ci_solver_2c = RelCISolver(
            nel=10, core_orbitals=list(range(6)), active_orbitals=list(range(6, 14))
        )
        ci_2c = CI(ci_solver=ci_solver_2c)(ghf)
        ci_2c.run()
        cis_2c.append(ci_2c)

        assert ci_2c.E_ci[0] == approx(ci_1c.E_ci[0])

    S_1c = ci_overlap(cis_1c[0], cis_1c[1])
    S_12 = ci_overlap(cis_2c[0], cis_2c[1])
    S_21 = ci_overlap(cis_2c[1], cis_2c[0])
    assert abs(S_12) == approx(abs(S_1c))
    assert S_12 == approx(np.conj(S_21))
    with pytest.raises(ValueError):
        ci_overlap(cis_1c[0], cis_2c[0])


def test_ci_overlap_two_component_single_determinant():
    """
    A single determinant of singly occupied spinors has overlap |det(S_occ)|,
    however its spinors are split between core and active.
    """
    x2c = X2CParams(x2c_type="so", x2c_model="1e")
    ghfs = []
    for scale in (1.0, 1.1):
        system = _water(scale, x2c=x2c)
        ghfs.append(GHF(charge=0, e_tol=1e-12)(system))

    for core, active in (
        (list(range(10)), []),
        (list(range(8)), [8, 9]),
        ([], list(range(10))),
    ):
        cis = []
        for ghf in ghfs:
            ci_solver = RelCISolver(nel=10, core_orbitals=core, active_orbitals=active)
            ci = CI(ci_solver=ci_solver)(ghf)
            ci.run()
            cis.append(ci)

        S_occ = mo_overlap(
            ghfs[0].C[0][:, :10], ghfs[0].system, ghfs[1].C[0][:, :10], ghfs[1].system
        )
        expected = abs(np.linalg.det(S_occ))
        S = ci_overlap(*cis)
        assert abs(S) == approx(expected), f"core={core}, active={active}"

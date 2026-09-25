import numpy as np
import pytest

from forte2 import CI, CISolver, GHF, MCOptimizer, RelCISolver, RHF, State, System
from forte2.base_classes import X2CParams
from forte2.helpers import random_unitary
from forte2.helpers.comparisons import approx, approx_abs
from forte2.orbitals import ci_overlap, mo_overlap
from forte2.orbitals.wavefunction_overlap import biorthogonalize_casscf_orbitals

ALGORITHMS = ["naive", "biorthogonal"]


def _water(scale, x2c=None, symmetry=False):
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
        symmetry=symmetry,
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_ci_overlap_invariant_under_orbital_rotations(algorithm):
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
    S = ci_overlap(ci_1, ci_2, algorithm=algorithm)
    assert abs(S) == approx(1.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_ci_overlap_displaced_casscf(algorithm):
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

    S_12 = ci_overlap(mcs[0], mcs[1], algorithm=algorithm)
    S_21 = ci_overlap(mcs[1], mcs[0], algorithm=algorithm)
    assert S_12 == approx(S_21)
    # the overlap is first order in the MCSCF convergence error
    assert abs(S_12) == approx_abs(0.989052518228039, 1e-6)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_ci_overlap_single_determinant(algorithm):
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
        S = ci_overlap(*cis, algorithm=algorithm)
        assert abs(S) == approx(expected), f"core={core}, active={active}"


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_ci_overlap_two_component_invariant_under_orbital_rotations(
    algorithm,
):
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
    S = ci_overlap(ci_1, ci_2, algorithm=algorithm)
    assert abs(S) == approx(1.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_ci_overlap_two_component_matches_nonrelativistic(algorithm):
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

    S_1c = ci_overlap(cis_1c[0], cis_1c[1], algorithm=algorithm)
    S_12 = ci_overlap(cis_2c[0], cis_2c[1], algorithm=algorithm)
    S_21 = ci_overlap(cis_2c[1], cis_2c[0], algorithm=algorithm)
    assert abs(S_12) == approx(abs(S_1c))
    assert S_12 == approx(np.conj(S_21))
    with pytest.raises(ValueError):
        ci_overlap(cis_1c[0], cis_2c[0], algorithm=algorithm)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_ci_overlap_two_component_single_determinant(algorithm):
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
        S = ci_overlap(*cis, algorithm=algorithm)
        assert abs(S) == approx(expected), f"core={core}, active={active}"


def test_ci_overlap_algorithms_agree():
    """
    The two algorithms agree for every root pair of a two-root CASCI at two
    geometries, where the second calculation orders its core and active
    orbitals differently. The reordering puts the biorthogonalizing rotations
    far from the identity.
    """
    cis = []
    for scale, order in ((1.0, [0, 1, 2, 3, 4, 5, 6]), (1.1, [1, 2, 0, 6, 3, 5, 4])):
        system = _water(scale)
        rhf = RHF(charge=0, e_tol=1e-12)(system)
        rhf.run()
        rhf.mos.C[0][:, :7] = rhf.mos.C[0][:, order]
        ci_solver = CISolver(
            State(nel=10, multiplicity=1, ms=0.0),
            core_orbitals=[0, 1, 2],
            active_orbitals=[3, 4, 5, 6],
            nroots=2,
        )
        ci = CI(ci_solver=ci_solver)(rhf)
        ci.run()
        cis.append(ci)

    for root_1 in range(2):
        for root_2 in range(2):
            S_naive = ci_overlap(cis[0], cis[1], root_1, root_2, algorithm="naive")
            S_bio = ci_overlap(cis[0], cis[1], root_1, root_2, algorithm="biorthogonal")
            assert S_bio == approx(S_naive), f"roots ({root_1}, {root_2})"


def test_ci_overlap_two_component_algorithms_agree():
    """Two-component version of the agreement test, for the ground state."""
    x2c = X2CParams(x2c_type="so", x2c_model="1e")
    order_2 = [2, 0, 1, 5, 3, 4, 13, 6, 11, 8, 9, 10, 7, 12]
    cis = []
    for scale, order in ((1.0, list(range(14))), (1.1, order_2)):
        system = _water(scale, x2c=x2c)
        ghf = GHF(charge=0, e_tol=1e-12)(system)
        ghf.run()
        ghf.mos.C[0][:, :14] = ghf.mos.C[0][:, order]
        ci_solver = RelCISolver(
            nel=10, core_orbitals=list(range(6)), active_orbitals=list(range(6, 14))
        )
        ci = CI(ci_solver=ci_solver)(ghf)
        ci.run()
        cis.append(ci)

    S_naive = ci_overlap(*cis, algorithm="naive")
    S_bio = ci_overlap(*cis, algorithm="biorthogonal")
    assert S_bio == approx(S_naive)


@pytest.mark.parametrize("cmplx", [False, True])
def test_biorthogonalize_casscf_orbitals(cmplx):
    """
    The transforms biorthonormalize two random orbital sets, keep active
    character out of the core, and are assembled from the returned factors.
    """
    rng = np.random.default_rng(0)
    nbf = 20
    for ndocc, nactv in ((3, 4), (0, 5), (5, 1)):
        n = ndocc + nactv
        C_X = random_unitary(nbf, cmplx=cmplx, rng=rng)[:, :n]
        C_Y = random_unitary(nbf, cmplx=cmplx, rng=rng)[:, :n]
        A = rng.standard_normal((nbf, nbf))
        S = C_X.conj().T @ (A @ A.T + nbf * np.eye(nbf)) @ C_Y

        bio = biorthogonalize_casscf_orbitals(S, ndocc, nactv)

        core, actv = slice(0, ndocc), slice(ndocc, n)
        assert bio.M.conj().T @ S @ bio.M_prime == approx_abs(np.eye(n), 1e-10)
        for M in (bio.M, bio.M_prime):
            assert M[actv, core] == approx_abs(np.zeros((nactv, ndocc)), 0)
        assert bio.M[core, core] == approx_abs(bio.U_C, 1e-14)
        assert bio.M_prime[core, core] == approx_abs(bio.V_C / bio.d_C, 1e-14)
        assert bio.M[actv, actv] == approx_abs(bio.U_A, 1e-14)
        assert bio.M_prime[actv, actv] == approx_abs(bio.V_A / bio.d_A, 1e-14)
        for U in (bio.U_C, bio.V_C, bio.U_A, bio.V_A):
            assert U.conj().T @ U == approx_abs(np.eye(len(U)), 1e-12)
        assert np.all(bio.d_C > 0) and np.all(bio.d_A > 0)


def test_ci_overlap_biorthogonal_rejects_unsupported_expansions():
    """
    The biorthogonal algorithm needs matching core and active spaces and the
    complete CAS determinant space. The naive algorithm handles both cases.
    """
    rhf = RHF(charge=0, e_tol=1e-12)(_water(1.0))
    ci_solver_44 = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2],
        active_orbitals=[3, 4, 5, 6],
    )
    ci_44 = CI(ci_solver=ci_solver_44)(rhf)
    ci_44.run()
    ci_solver_65 = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1],
        active_orbitals=[2, 3, 4, 5, 6],
    )
    ci_65 = CI(ci_solver=ci_solver_65)(rhf)
    ci_65.run()

    with pytest.raises(ValueError):
        ci_overlap(ci_44, ci_65, algorithm="biorthogonal")
    assert 0.0 < abs(ci_overlap(ci_44, ci_65, algorithm="naive")) < 1.0

    rhf_sym = RHF(charge=0, e_tol=1e-12)(_water(1.0, symmetry=True))
    ci_solver_sym = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2],
        active_orbitals=[3, 4, 5, 6],
    )
    ci_sym = CI(ci_solver=ci_solver_sym)(rhf_sym)
    ci_sym.run()

    with pytest.raises(ValueError):
        ci_overlap(ci_sym, ci_sym, algorithm="biorthogonal")
    assert abs(ci_overlap(ci_sym, ci_sym, algorithm="naive")) == approx(1.0)

    with pytest.raises(ValueError):
        ci_overlap(ci_44, ci_44, algorithm="lowdin")

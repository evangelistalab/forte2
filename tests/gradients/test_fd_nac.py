import numpy as np
import pytest

from forte2 import (
    CI,
    CISolver,
    GHF,
    MCOptimizer,
    RelCISolver,
    RHF,
    ROHF,
    SelectedCISolver,
    State,
    System,
)
from forte2.gradients import FiniteDifference
from forte2.helpers import random_unitary
from forte2.helpers.comparisons import approx, approx_abs
from forte2.orbitals import rotate_ci_vectors

# produced with PySCF input at
# https://gist.github.com/brianz98/223c4fd30ca3b639663d0529fed05165
E_REF = np.array([-1.600377528293, -1.380067593847])
G_REF = np.array(
    [
        [
            [1.823772876822e-02, 0.0, 1.844445293503e-02],
            [-5.706708035675e-03, 0.0, 1.147125722011e-02],
            [-1.253102073255e-02, 0.0, -2.991571015514e-02],
        ],
        [
            [-2.982252948068e-02, 0.0, 9.062415973925e-02],
            [6.881781546153e-02, 0.0, -9.492804066086e-02],
            [-3.899528598085e-02, 0.0, 4.303880921606e-03],
        ],
    ]
)
D_REF = np.array(
    [
        [7.489054640155e-02, 0.0, -5.065553038667e-01],
        [3.396528976358e-01, 0.0, 6.254351022779e-01],
        [-4.652844755974e-01, 0.0, -2.167638626283e-01],
    ]
)


def _h3(two_component):
    """
    SA-CASSCF(3,3) of a scalene H3 over its two lowest doublets, or over the
    four roots of their Kramers pairs if two-component.
    """
    system = System(
        xyz="H 0 0 0; H 0 0 0.9; H 0.8 0 1.5",
        basis_set="cc-pvdz",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    if two_component:
        scf = GHF(charge=0)(system)
        ci_solver = RelCISolver(nel=3, active_orbitals=6, nroots=4)
    else:
        scf = ROHF(charge=0, ms=0.5)(system)
        ci_solver = CISolver(
            State(nel=3, multiplicity=2, ms=0.5), active_orbitals=3, nroots=2
        )
    return MCOptimizer(ci_solver, e_tol=1e-10, g_tol=1e-8)(scf)


def test_finite_difference_nac_and_gradients_match_pyscf():
    """
    One sweep of displacements gives the gradient of every root and the
    coupling between them.
    """
    mc = _h3(two_component=False)
    fd = FiniteDifference(compute_nac=True)(mc)
    d = fd.nonadiabatic_coupling(ket=1, bra=0)
    gradients = [fd.gradient(root=root) for root in range(2)]
    assert fd.n_evaluations == 4 * 9

    assert mc.E_ci == approx(E_REF)
    for g, g_ref in zip(gradients, G_REF):
        assert g == approx_abs(g_ref, 1e-7)
    # without a root, the gradient is that of the state-averaged energy
    assert fd.gradient() == approx(np.mean(gradients, axis=0))

    assert d.shape == (3, 3)
    assert not np.iscomplexobj(d)
    # the sign of the coupling follows the arbitrary signs of the CI vectors
    sign = np.sign(d[0, 2] / D_REF[0, 2])
    assert sign * d == approx_abs(D_REF, 1e-6)
    assert fd.nonadiabatic_coupling(ket=0, bra=1) == approx_abs(-d, 1e-6)
    h = fd.nonadiabatic_coupling(ket=1, bra=0, energy_gap_weighted=True)
    assert h == approx((mc.E_ci[1] - mc.E_ci[0]) * d)
    assert fd.anti_hermiticity_residual < 1e-6
    assert fd.min_overlap_singular_value > 0.999

    # components limits the displacements, and leaves the others NaN
    components = [(0, 2), (2, 0)]
    atoms, xyzs = map(list, zip(*components))
    skipped = np.ones((3, 3), dtype=bool)
    skipped[atoms, xyzs] = False
    sweep = 4 * len(components)

    # without compute_nac, couplings requested after a gradient take a second sweep
    fd_lazy = FiniteDifference(components=components)(_h3(two_component=False))
    g_lazy = fd_lazy.gradient(root=0)
    assert fd_lazy.n_evaluations == sweep
    assert np.isnan(g_lazy[skipped]).all()
    assert g_lazy[atoms, xyzs] == approx_abs(G_REF[0][atoms, xyzs], 1e-7)
    d_lazy = fd_lazy.nonadiabatic_coupling(ket=1, bra=0)
    assert fd_lazy.n_evaluations == 2 * sweep
    assert np.isnan(d_lazy[skipped]).all()
    assert np.abs(d_lazy[atoms, xyzs]) == approx_abs(np.abs(D_REF[atoms, xyzs]), 1e-6)

    # but the sweep for couplings also serves the gradients
    fd_nac_first = FiniteDifference(components=components)(_h3(two_component=False))
    fd_nac_first.nonadiabatic_coupling(ket=1, bra=0)
    g_1 = fd_nac_first.gradient(root=1)[atoms, xyzs]
    assert g_1 == approx_abs(G_REF[1][atoms, xyzs], 1e-7)
    assert fd_nac_first.n_evaluations == sweep


def test_finite_difference_nac_two_component_kramers_pairs():
    """
    Without spin-orbit coupling, each doublet becomes a Kramers pair, and the
    2c coupling block between the two pairs is the 1c coupling times a 2x2
    unitary that's the same for every Cartesian component. The reference is
    scrambled first, by a random rotation of its active spinors and random
    mixing within each Kramers pair, which the alignment of the displaced roots
    has to carry over.
    """
    components = [(0, 2), (2, 0), (2, 2)]
    atoms, xyzs = map(list, zip(*components))
    mc = _h3(two_component=True)
    mc.run()
    rng = np.random.default_rng(0)
    actv = mc.mo_space.active_indices
    U_actv = random_unitary(len(actv), cmplx=True, rng=rng)
    evecs = rotate_ci_vectors(mc.ci_solver, U_actv)[0]
    mc.mos.C[0][:, actv] = mc.mos.C[0][:, actv] @ U_actv
    for pair in ([0, 1], [2, 3]):
        evecs[:, pair] = evecs[:, pair] @ random_unitary(2, cmplx=True, rng=rng)
    mc.ci_solver.sub_solvers[0].evecs = evecs

    fd = FiniteDifference(compute_nac=True, components=components)(mc)
    block = np.array(
        [[fd.nonadiabatic_coupling(ket=k, bra=b) for k in (2, 3)] for b in (0, 1)]
    )

    assert mc.E_ci.real == approx(np.repeat(E_REF, 2))
    unitaries = block[:, :, atoms, xyzs] / D_REF[atoms, xyzs]
    # the mixing within the pairs leaves no element of the block small
    assert np.abs(unitaries).min() > 0.1
    for k in range(len(components)):
        U = unitaries[:, :, k]
        assert U.conj().T @ U == approx_abs(np.eye(2), 2e-6)
        assert U == approx_abs(unitaries[:, :, 0], 2e-6)
    # the two roots of a Kramers pair have no well-defined coupling
    with pytest.raises(ValueError):
        fd.nonadiabatic_coupling(ket=1, bra=0)
    # but they share the gradient of the 1c doublet
    for root in range(4):
        g = fd.gradient(root=root)[atoms, xyzs]
        assert g == approx_abs(G_REF[root // 2][atoms, xyzs], 1e-7)


def test_finite_difference_nac_rejects_unsupported_setups():
    system = System(
        xyz="H 0 0 0\nH 0 0 1.6",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0)(system)
    singlet = State(nel=2, multiplicity=1, ms=0.0)
    unsupported = [
        (lambda: rhf, TypeError),
        (
            lambda: MCOptimizer(
                SelectedCISolver(singlet, active_orbitals=[0, 1], nroots=2)
            )(rhf),
            TypeError,
        ),
        (
            lambda: MCOptimizer(CISolver(singlet, active_orbitals=[0, 1]))(rhf),
            ValueError,
        ),
    ]

    for parent, error in unsupported:
        # compute_nac checks the parent when it's attached
        with pytest.raises(error):
            FiniteDifference(compute_nac=True)(parent())
        # otherwise the couplings check it before any displacement
        fd = FiniteDifference()(parent())
        with pytest.raises(error):
            fd.nonadiabatic_coupling(ket=1, bra=0)
        assert fd.n_evaluations == 0

    for options in (
        {"step": 0.0},
        {"npoints": 3},
        {"degeneracy_tol": -1.0},
        {"overlap_algorithm": "lowdin"},
    ):
        with pytest.raises(ValueError):
            FiniteDifference(**options)

    FiniteDifference(compute_nac=True)(
        CI(CISolver(singlet, active_orbitals=[0, 1], nroots=2))(rhf)
    )
    fd = FiniteDifference(compute_nac=True)(
        MCOptimizer(CISolver(singlet, active_orbitals=[0, 1], nroots=2))(rhf)
    )
    for ket, bra in ((0, 0), (0, 2)):
        with pytest.raises(ValueError):
            fd.nonadiabatic_coupling(ket=ket, bra=bra)
    with pytest.raises(ValueError):
        fd.gradient(root=2)
    assert fd.n_evaluations == 0

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
from forte2.base_classes import X2CParams
from forte2.gradients import FiniteDifference
from forte2.helpers.comparisons import approx, approx_abs


def _lih():
    """SA-CASSCF(2,5) of LiH over S0, S1, and the Ms = 1 component of T1."""
    system = System(
        xyz="Li 0 0 0; H 0 0 4",
        basis_set="cc-pvdz",
        cholesky_tei=True,
        cholesky_tol=1e-10,
        x2c=X2CParams(x2c_type="sf"),
    )
    ci_solver = CISolver(
        states=[
            State(nel=4, multiplicity=1, ms=0.0),
            State(nel=4, multiplicity=3, ms=1.0),
        ],
        core_orbitals=1,
        active_orbitals=5,
        nroots=[2, 1],
    )
    return MCOptimizer(ci_solver, e_tol=1e-10, g_tol=1e-8)(RHF(charge=0)(system))


def _h3(two_component):
    """SA-CASSCF(3,3) of a scalene H3 over its two lowest doublets."""
    system = System(
        xyz="H 0 0 0; H 0 0 0.9; H 0.8 0 1.5",
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        x2c=X2CParams(x2c_type="so" if two_component else "sf"),
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
    coupling <S0|dS1/dz>. The references are analytic results from PySCF 2.14
    (pyscf.grad.sacasscf, and pyscf.nac.sacasscf with use_etfs=False) with
    forte2's cc-pVDZ and exact integrals, which the tight Cholesky
    decomposition reproduces. Overlaps are first order in the MCSCF
    convergence error, which varies between runs, so the couplings are only
    reproducible to a few 1e-6 and the gradients to a few 1e-8.
    """
    components = [(0, 2), (1, 2)]
    sweep = 4 * len(components)
    mc = _lih()
    fd = FiniteDifference(compute_nac=True, components=components)(mc)
    d = fd.nonadiabatic_coupling(ket=1, bra=0)
    gradients = [fd.gradient(root=root) for root in range(3)]
    assert fd.n_evaluations == sweep

    assert mc.E_ci == approx([-7.935773561, -7.883463204, -7.931146497])
    for g, g_H in zip(gradients, (3.686544246e-3, 8.910081672e-3, -1.095441977e-3)):
        assert np.isnan(g[:, :2]).all()
        assert g[:, 2] == approx_abs([-g_H, g_H], 2e-7)
    # without a root, the gradient is that of the state-averaged energy
    mean = np.mean([g[:, 2] for g in gradients], axis=0)
    assert fd.gradient()[:, 2] == approx(mean)

    assert d.shape == (2, 3)
    assert not np.iscomplexobj(d)
    assert np.isnan(d[:, :2]).all()
    # the sign of the coupling follows the arbitrary signs of the CI vectors
    reference = np.array([-0.270682936, 0.113610364])
    sign = np.sign(d[1, 2] / reference[1])
    assert sign * d[:, 2] == approx_abs(reference, 1e-5)

    assert fd.nonadiabatic_coupling(ket=0, bra=1)[:, 2] == approx_abs(-d[:, 2], 1e-6)
    h = fd.nonadiabatic_coupling(ket=1, bra=0, energy_gap_weighted=True)
    assert h[:, 2] == approx((mc.E_ci[1] - mc.E_ci[0]) * d[:, 2])
    # S0 and the Ms = 1 triplet don't couple without spin-orbit coupling
    assert fd.nonadiabatic_coupling(ket=2, bra=0)[:, 2] == approx_abs(np.zeros(2), 0)
    assert fd.anti_hermiticity_residual < 1e-6
    assert fd.min_overlap_singular_value > 0.999
    assert fd.n_evaluations == sweep

    # without compute_nac, couplings requested after a gradient take a second sweep
    fd_lazy = FiniteDifference(components=components)(_lih())
    assert fd_lazy.gradient(root=0)[:, 2] == approx_abs(gradients[0][:, 2], 2e-7)
    assert fd_lazy.n_evaluations == sweep
    d_lazy = fd_lazy.nonadiabatic_coupling(ket=1, bra=0)
    assert fd_lazy.n_evaluations == 2 * sweep
    assert np.abs(d_lazy[:, 2]) == approx_abs(np.abs(d[:, 2]), 1e-5)

    # but the sweep for couplings also serves the gradients
    fd_nac_first = FiniteDifference(components=components)(_lih())
    fd_nac_first.nonadiabatic_coupling(ket=1, bra=0)
    assert fd_nac_first.gradient(root=1)[:, 2] == approx_abs(gradients[1][:, 2], 2e-7)
    assert fd_nac_first.n_evaluations == sweep


def test_finite_difference_nac_two_component_kramers_pairs():
    """
    Spin-orbit coupling in H3 is so weak that each doublet becomes a Kramers
    pair, and the 2c coupling block between the two pairs is the 1c coupling
    times a 2x2 unitary. The elements of the block carry the arbitrary phases
    and mixing of the 2c CI vectors, but the unitary is the same for every
    Cartesian component up to first-order spin-orbit effects.
    """
    components = [(0, 2), (2, 0), (2, 2)]
    atoms, xyzs = map(list, zip(*components))
    fd_1c = FiniteDifference(compute_nac=True, npoints=2, components=components)(
        _h3(two_component=False)
    )
    d_1c = fd_1c.nonadiabatic_coupling(ket=1, bra=0)
    fd_2c = FiniteDifference(compute_nac=True, npoints=2, components=components)(
        _h3(two_component=True)
    )
    block = np.array(
        [[fd_2c.nonadiabatic_coupling(ket=k, bra=b) for k in (2, 3)] for b in (0, 1)]
    )

    assert np.iscomplexobj(block)
    unitaries = block[:, :, atoms, xyzs] / d_1c[atoms, xyzs]
    for k in range(len(components)):
        U = unitaries[:, :, k]
        assert U.conj().T @ U == approx_abs(np.eye(2), 1e-6)
        assert U == approx_abs(unitaries[:, :, 0], 1e-4)
    # the two roots of a Kramers pair have no well-defined coupling
    with pytest.raises(ValueError):
        fd_2c.nonadiabatic_coupling(ket=1, bra=0)
    # but they share the gradient of the 1c doublet
    g_1c = fd_1c.gradient(root=1)[atoms, xyzs]
    for root in (2, 3):
        assert fd_2c.gradient(root=root)[atoms, xyzs] == approx_abs(g_1c, 1e-7)


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

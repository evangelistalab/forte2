from forte2 import System, RHF, CISolver, MCOptimizer, AVAS, State
from forte2.helpers.comparisons import approx
import numpy as np
from forte2.mcopt.orbital_optimizer import OrbOptimizer


def test_penalty_gradient_fd():
    """Finite-difference validation of the penalty gradient."""

    eps = 1e-6
    lam = 0.7

    nmo = 6
    C0 = np.eye(nmo)

    core = slice(0, 2)
    actv = slice(2, 4)
    virt = slice(4, 6)
    extents = [core, actv, virt]

    nrr = np.zeros((nmo, nmo), dtype=bool)
    nrr[np.triu_indices(nmo, 1)] = True

    opt = OrbOptimizer(
        C=C0,
        extents=extents,
        fock_builder=None,
        hcore=None,
        e_nuc=0.0,
        nrr=nrr,
        lambda_penalty=lam,
    )

    #
    # Move away from the stationary point.
    # Rotate active orbital 2 into virtual orbital 4.
    #

    pairs = np.argwhere(nrr)

    idx = np.where((pairs[:, 0] == 2) & (pairs[:, 1] == 4))[0][0]

    R0 = np.zeros(len(pairs))
    R0[idx] = 0.2

    #
    # Evaluate analytic gradient at R0.
    #

    opt.R[:] = 0.0
    opt.U[:] = np.eye(nmo)
    opt._update_orbitals(R0)

    Gmat = opt._penalty_gradient_matrix()

    grad_analytic = (2 * (Gmat - Gmat.T.conj()))[nrr]

    #
    # Finite-difference gradient.
    #

    grad_fd = np.zeros_like(grad_analytic)

    for i in range(len(grad_fd)):

        dx = np.zeros_like(grad_fd)
        dx[i] = eps

        # E(R0 + dx)

        opt.R[:] = 0.0
        opt.U[:] = np.eye(nmo)
        opt._update_orbitals(R0 + dx)

        Ep = opt._penalty_energy()

        # E(R0 - dx)

        opt.R[:] = 0.0
        opt.U[:] = np.eye(nmo)
        opt._update_orbitals(R0 - dx)

        Em = opt._penalty_energy()

        grad_fd[i] = (Ep - Em) / (2 * eps)

    np.testing.assert_allclose(
        grad_fd,
        grad_analytic,
        rtol=1e-6,
        atol=1e-8,
    )


test_penalty_gradient_fd()


def test_casscf_cyclopropene_with_default_penalty():
    """Test CASSCF with cyclopropene (C3H4) molecule."""
    erhf = -114.40009162104958
    emcscf = -114.4408154316
    Dev = 7.3232704156e-03
    xyz = """
    H   0.912650   0.000000   1.457504
    H  -0.912650   0.000000   1.457504
    H   0.000000  -1.585659  -1.038624
    H   0.000000   1.585659  -1.038624
    C   0.000000   0.000000   0.859492
    C   0.000000  -0.651229  -0.499559
    C   0.000000   0.651229  -0.499559
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-6)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    ci_solver = CISolver(State(nel=rhf.nel, multiplicity=1, ms=0.0))
    mc = MCOptimizer(ci_solver, lambda_penalty=0.5)(avas)
    mc.run()

    assert rhf.E == approx(erhf)

    assert mc.E == approx(emcscf)
    assert mc.delta_act == approx(Dev)


def test_casscf_cyclopropene_with_no_penalty():
    """Test CASSCF with cyclopropene (C3H4) molecule."""
    erhf = -114.40009162104958
    emcscf = -114.4408319744
    xyz = """
    H   0.912650   0.000000   1.457504
    H  -0.912650   0.000000   1.457504
    H   0.000000  -1.585659  -1.038624
    H   0.000000   1.585659  -1.038624
    C   0.000000   0.000000   0.859492
    C   0.000000  -0.651229  -0.499559
    C   0.000000   0.651229  -0.499559
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-6)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    ci_solver = CISolver(State(nel=rhf.nel, multiplicity=1, ms=0.0))
    mc = MCOptimizer(ci_solver, lambda_penalty=0)(avas)
    mc.run()

    assert rhf.E == approx(erhf)

    assert mc.E == approx(emcscf)
    assert not hasattr(mc, "delta_act")


def test_casscf_cyclopropene_with_large_penalty():
    """Test CASSCF with cyclopropene (C3H4) molecule."""
    erhf = -114.40009162104958
    # emcscf = -114.440831983407
    xyz = """
    H   0.912650   0.000000   1.457504
    H  -0.912650   0.000000   1.457504
    H   0.000000  -1.585659  -1.038624
    H   0.000000   1.585659  -1.038624
    C   0.000000   0.000000   0.859492
    C   0.000000  -0.651229  -0.499559
    C   0.000000   0.651229  -0.499559
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-6)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    ci_solver = CISolver(State(nel=rhf.nel, multiplicity=1, ms=0.0))
    mc = MCOptimizer(ci_solver, lambda_penalty=1)(avas)
    mc.run()

    assert rhf.E == approx(erhf)


test_casscf_cyclopropene_with_large_penalty()

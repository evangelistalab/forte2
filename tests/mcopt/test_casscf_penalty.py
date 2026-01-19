from forte2 import System, RHF, MCOptimizer, AVAS, State
from forte2.helpers.comparisons import approx
import numpy as np
from forte2.mcopt.orbital_optimizer import OrbOptimizer


def test_penalty_gradient_fd():
    eps = 1e-6  # step size for finite difference
    lam = 0.7

    nmo = 6
    C0 = np.eye(nmo)

    core = slice(0, 2)
    actv = slice(2, 4)
    virt = slice(4, 6)
    extents = [core, actv, virt]

    nrr = np.zeros((nmo, nmo), dtype=bool)
    nrr[np.triu_indices(nmo, 1)] = True  # off-diagonal upper triangle

    opt = OrbOptimizer(
        C=C0,
        extents=extents,
        fock_builder=None,
        hcore=None,  # only compare the penalty functional
        e_nuc=0.0,
        nrr=nrr,
        lambda_penalty=lam,
    )

    # ensure reference state
    opt.R[:] = 0.0  # rotation parameters
    opt.U[:] = np.eye(nmo)  # unitary matrix

    # ----- analytic gradient -----
    Gmat = opt._penalty_gradient_matrix()  # get the gradient matrix from penalty
    grad_analytic = (2 * (Gmat - Gmat.T.conj()))[nrr]  # vectorize the gradient

    # ----- finite-difference gradient -----
    grad_fd = np.zeros_like(grad_analytic)

    for i in range(len(grad_fd)):
        dx = np.zeros_like(grad_fd)
        dx[i] = eps

        # +eps
        opt.R[:] = 0.0
        opt.U[:] = np.eye(nmo)
        opt._update_orbitals(dx)
        Ep = opt._penalty_energy()

        # -eps
        opt.R[:] = 0.0
        opt.U[:] = np.eye(nmo)
        opt._update_orbitals(-dx)
        Em = opt._penalty_energy()

        grad_fd[i] = (Ep - Em) / (2 * eps)

        print(
            "max |grad_fd - grad_analytic| =", np.max(np.abs(grad_fd - grad_analytic))
        )

    np.testing.assert_allclose(
        grad_fd,
        grad_analytic,
        rtol=1e-6,
        atol=1e-8,
    )


def test_casscf_cyclopropene():
    """Test CASSCF with cyclopropene (C3H4) molecule."""
    erhf = -114.400091611464
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

    rhf = RHF(charge=0, econv=1e-6)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    mc = MCOptimizer(State(nel=rhf.nel, multiplicity=1, ms=0.0), lambda_penalty=0.5)(
        avas
    )
    mc.run()

    assert rhf.E == approx(erhf)

    assert mc.E == approx(emcscf)
    assert mc.delta_act == approx(Dev)

import pytest
import numpy as np

from forte2 import System, RHF, MCOptimizer, AVAS, State, CISolver
from forte2.helpers.comparisons import approx


def test_sa_casscf_same_mult():
    erhf = -108.761639873604
    ecasscf = -108.8592663803

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_solver = CISolver(
        State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        nroots=2,
    )
    mc = MCOptimizer(
        ci_solver,
        gconv=1e-7,
    )(rhf)
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_sa_mcscf_diff_mult_with_avas():
    # This should be strictly identical to test_mcscf_sa_diff_mult given a sufficiently robust MCSCF solver.
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=3,
        subspace=["N(2p)"],
        diagonalize=True,
    )(rhf)
    singlet = State(nel=rhf.nel, multiplicity=1, ms=0.0)
    triplet = State(nel=rhf.nel, multiplicity=3, ms=0.0)
    ci_solver = CISolver(
        [singlet, triplet], weights=[[0.25], [0.75 * 0.85, 0.75 * 0.15]], nroots=[1, 2]
    )
    mc = MCOptimizer(ci_solver)(avas)
    mc.run()

    eref_singlet = -109.0664322107
    eref_triplet1 = -108.8450131892
    eref_triplet2 = -108.7888580871

    assert mc.E_ci[0] == approx(eref_singlet)
    assert mc.E_ci[1] == approx(eref_triplet1)
    assert mc.E_ci[2] == approx(eref_triplet2)
    assert mc.ci_solver.compute_average_energy() == approx(
        0.25 * eref_singlet + 0.75 * (0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    )

    nat_occ_ref = np.array(
        [
            [1.97531261, 1.9761829, 1.9797172, 1.97636293],
            [1.91200523, 1.4525388, 1.46608628, 1.5689295],
            [1.91200523, 1.45253877, 1.46608627, 1.56892948],
            [0.08779586, 0.54735532, 0.53380304, 0.43094082],
            [0.08779586, 0.54735529, 0.53380303, 0.4309408],
            [0.0250852, 0.02402894, 0.02050418, 0.02389647],
        ]
    )
    assert mc.ci_solver.nat_occs == pytest.approx(nat_occ_ref, abs=5e-7)


def test_sa_casscf_diff_mult():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    singlet = State(nel=rhf.nel, multiplicity=1, ms=0.0)
    triplet = State(nel=rhf.nel, multiplicity=3, ms=0.0)
    ci_solver = CISolver(
        [singlet, triplet],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        weights=[[0.25], [0.75 * 0.85, 0.75 * 0.15]],
        nroots=[1, 2],
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    eref_singlet = -109.0664322107
    eref_triplet1 = -108.8450131892
    eref_triplet2 = -108.7888580871

    assert mc.E_ci[0] == approx(eref_singlet)
    assert mc.E_ci[1] == approx(eref_triplet1)
    assert mc.E_ci[2] == approx(eref_triplet2)
    assert mc.E == approx(
        0.25 * eref_singlet + 0.75 * (eref_triplet1 * 0.85 + eref_triplet2 * 0.15)
    )

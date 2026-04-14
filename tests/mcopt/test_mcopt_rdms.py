import numpy as np
import pytest

from forte2 import System, RHF, MCOptimizer, State, CISolver
from forte2.base_classes import DavidsonLiuParams


def assert_tuple_allclose(got, ref, atol=1e-12):
    assert len(got) == len(ref)
    for x, y in zip(got, ref):
        np.testing.assert_allclose(x, y, rtol=0.0, atol=atol)


def test_mcoptimizer_rdm_accessors_single_solver():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.058354421806
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    rhf = RHF(charge=0, econv=1e-12)(system)

    ci_solver = CISolver(
        State(nel=2, multiplicity=1, ms=0.0),
        active_orbitals=[0, 1],
        nroots=2,
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-12, r_tol=1e-10),
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    solver = mc.ci_solver.sub_solvers[0]

    assert len(mc.ci_solver.sub_solvers) == 1

    assert_tuple_allclose(mc.make_sd_1rdm(0), solver.make_sd_1rdm(0))
    assert_tuple_allclose(mc.make_sd_2rdm(0), solver.make_sd_2rdm(0))
    assert_tuple_allclose(mc.make_sd_3rdm(0), solver.make_sd_3rdm(0))

    np.testing.assert_allclose(
        mc.make_sf_1rdm(1), solver.make_sf_1rdm(1), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        mc.make_sf_2rdm(1), solver.make_sf_2rdm(1), rtol=0.0, atol=1e-12
    )


def test_mcoptimizer_rdm_accessors_multi_solver():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, econv=1e-8)(system)

    singlet = State(nel=10, multiplicity=1, ms=0.0)
    triplet = State(nel=10, multiplicity=3, ms=1.0)
    ci_solver = CISolver(
        states=[singlet, triplet],
        nroots=[2, 1],
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6, 7],
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-8, r_tol=1e-4),
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    singlet_solver, triplet_solver = mc.ci_solver.sub_solvers

    with pytest.raises(ValueError, match="Cross-state RDMs are not supported"):
        mc.make_sd_1rdm(1, 2)

    with pytest.raises(ValueError, match="absolute_root must be between 0"):
        mc.make_sd_2rdm(1, 7)

    np.testing.assert_allclose(
        mc.make_sf_1rdm(1),
        singlet_solver.make_sf_1rdm(1),
        rtol=0.0,
        atol=1e-12,
    )
    assert_tuple_allclose(
        mc.make_sd_2rdm(2),
        triplet_solver.make_sd_2rdm(0),
    )
test_mcoptimizer_rdm_accessors_multi_solver()
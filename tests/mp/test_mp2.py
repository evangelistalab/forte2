import time

import pytest
import numpy as np

from forte2 import AVAS, System
from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals
from forte2.scf import RHF, ROHF, UHF
from forte2.helpers.comparisons import approx
from forte2.mp import RMP2, ROMP2, UMP2
from forte2.props import (
    RMP2MPQOnTheFly,
    UMP2MPQOnTheFly,
    MutualCorrelationAnalysis,
    rmp2_mpq_onthefly_no,
    ump2_mpq_onthefly_no,
)


def assert_uhf_rdm_invariants(mp2, na, nb):
    gamma1_a, gamma1_b = mp2.make_1rdm_sd()
    gamma1_sf = mp2.make_1rdm_sf()
    gamma2_aa, gamma2_ab, gamma2_bb = mp2.make_2rdm_sd((gamma1_a, gamma1_b))
    gamma2_sf = mp2.make_2rdm_sf((gamma1_a, gamma1_b))

    assert np.trace(gamma1_a) == approx(na)
    assert np.trace(gamma1_b) == approx(nb)
    assert np.trace(gamma1_sf) == approx(na + nb)

    assert np.max(np.abs(gamma1_a - gamma1_a.T)) == approx(0.0)
    assert np.max(np.abs(gamma1_b - gamma1_b.T)) == approx(0.0)
    assert np.max(np.abs(gamma1_sf - gamma1_sf.T)) == approx(0.0)

    assert np.max(np.abs(gamma2_aa + gamma2_aa.transpose(1, 0, 2, 3))) == approx(0.0)
    assert np.max(np.abs(gamma2_aa + gamma2_aa.transpose(0, 1, 3, 2))) == approx(0.0)
    assert np.max(np.abs(gamma2_bb + gamma2_bb.transpose(1, 0, 2, 3))) == approx(0.0)
    assert np.max(np.abs(gamma2_bb + gamma2_bb.transpose(0, 1, 3, 2))) == approx(0.0)
    assert np.max(np.abs(gamma2_aa - gamma2_aa.transpose(2, 3, 0, 1))) == approx(0.0)
    assert np.max(np.abs(gamma2_ab - gamma2_ab.transpose(2, 3, 0, 1))) == approx(0.0)
    assert np.max(np.abs(gamma2_bb - gamma2_bb.transpose(2, 3, 0, 1))) == approx(0.0)

    assert np.max(np.abs(gamma2_sf - gamma2_sf.transpose(1, 0, 3, 2))) == approx(0.0)
    assert np.max(np.abs(gamma2_sf - gamma2_sf.transpose(2, 3, 0, 1))) == approx(0.0)

    lambda2_sf = mp2.make_2cumulant(gamma1_sf, gamma2_sf)
    lambda2_aa, lambda2_ab, lambda2_bb = mp2.make_2cumulant_sd(
        (gamma1_a, gamma1_b), (gamma2_aa, gamma2_ab, gamma2_bb)
    )
    gamma1_sd, gamma2_sd, lambda2_sd = mp2.make_cumulants_sd()
    assert gamma1_sd[0] == approx(gamma1_a)
    assert gamma1_sd[1] == approx(gamma1_b)
    assert gamma2_sd[0] == approx(gamma2_aa)
    assert gamma2_sd[1] == approx(gamma2_ab)
    assert gamma2_sd[2] == approx(gamma2_bb)
    assert lambda2_sd[0] == approx(lambda2_aa)
    assert lambda2_sd[1] == approx(lambda2_ab)
    assert lambda2_sd[2] == approx(lambda2_bb)
    lambda2_sf_from_sd = (
        lambda2_aa + lambda2_bb + lambda2_ab + lambda2_ab.transpose(1, 0, 3, 2)
    )
    assert lambda2_sf == approx(lambda2_sf_from_sd)

    gamma1_ao_a, gamma1_ao_b = mp2.gamma1_mo_to_ao((gamma1_a, gamma1_b))
    assert np.max(np.abs(gamma1_ao_a - gamma1_ao_a.T)) == approx(0.0)
    assert np.max(np.abs(gamma1_ao_b - gamma1_ao_b.T)) == approx(0.0)
    S = mp2.system.ints_overlap()
    assert np.einsum("pq,qp->", S, gamma1_ao_a) == approx(na)
    assert np.einsum("pq,qp->", S, gamma1_ao_b) == approx(nb)


def assert_t2_not_stored(mp2):
    assert getattr(mp2, "t2", None) is None
    assert getattr(mp2, "t2_as", None) is None
    assert getattr(mp2, "t2_a", None) is None
    assert getattr(mp2, "t2_b", None) is None
    assert getattr(mp2, "t2_ab", None) is None


def test_mp2():
    # reference values from Psi4 using the cc-pVQZ basis set and the cc-pVQZ-JKFIT auxiliary basis set

    energy_scf = -76.0614664072629836
    energy_mp2 = -76.3710978841482984

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()

    print(f"RHF energy: {scf.E:.10f} [Eh]")

    assert scf.E == approx(energy_scf)

    jkbuilder = system.fock_builder
    nocc = scf.na
    nvir = scf.nbf - nocc
    Co = scf.C[0][:, :nocc]
    Cv = scf.C[0][:, nocc:]
    V = jkbuilder.two_electron_integrals_gen_block(Co, Co, Cv, Cv)
    epso = scf.eps[0][:nocc]
    epsv = scf.eps[0][nocc:]

    # Compute the MP2 energy
    start = time.monotonic()
    Emp2 = scf.E
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    den = 1.0 / (epso[i] + epso[j] - epsv[a] - epsv[b])
                    Emp2 += V[i, j, a, b] * (2 * V[i, j, a, b] - V[i, j, b, a]) * den
    end = time.monotonic()

    print(f"MP2 energy: {Emp2:.10f} [Eh]")
    print(f"Time taken: {end - start:.4f} seconds")

    assert Emp2 == approx(energy_mp2)


# Tests below use reference values from PYSCF using the cc-pVQZ basis set and the cc-pVQZ-JKFIT auxiliary basis set


def test_rhf_mp2():
    erhf = -76.0614664072629
    emp2 = -76.3710978833093
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = RHF(charge=0)(system)
    mp2 = RMP2(store_t2=True)(scf)
    mp2.run()

    g1 = mp2.make_1rdm()
    g2 = mp2.make_2rdm(g1)
    C_no, no_occs, U_no = mp2.make_natural_orbital_transform(g1)

    S = system.ints_overlap()
    assert C_no.T @ S @ C_no == approx(np.eye(mp2.nocc + mp2.nvir))
    assert U_no[: mp2.nocc, mp2.nocc :] == approx(0.0)
    assert U_no[mp2.nocc :, : mp2.nocc] == approx(0.0)
    assert U_no.T @ g1 @ U_no == approx(np.diag(no_occs))
    assert mp2.C_no == approx(C_no)
    assert mp2.no_occs == approx(no_occs)
    assert mp2.U_no == approx(U_no)

    moints = RestrictedMOIntegrals(system, scf.C[0], list(range(scf.nmo)))
    Ecore = moints.E
    H = moints.H
    V = moints.V

    mp2_rdm_E = mp2.energy_given_rdms(Ecore, H, V, g1, g2)

    linear_mpq = RMP2MPQOnTheFly(
        mp2, U=np.eye(mp2.nocc + mp2.nvir), include_quadratic=False
    )
    i, j, a, b = 0, 1, 0, 1
    assert linear_mpq.lambda2_ab_linear_elem(
        i, j, mp2.nocc + a, mp2.nocc + b
    ) == approx(mp2.t2[i, j, a, b])
    assert linear_mpq.lambda2_aa_linear_elem(
        i, j, mp2.nocc + a, mp2.nocc + b
    ) == approx(mp2.t2[i, j, a, b] - mp2.t2[i, j, b, a])

    avas = AVAS(
        subspace=["O(2p)"], selection_method="total", num_active=3
    )(scf)
    avas_mpq = rmp2_mpq_onthefly_no(mp2, avas=avas)
    assert avas.executed
    assert avas_mpq.rdm_info_selection == "avas"
    assert len(avas_mpq.rdm_info_indices) == avas.nactv
    weights = avas_mpq.rdm_info_selection_details[
        "avas_projection_weights"
    ]
    expected = tuple(
        sorted(np.argsort(-weights, kind="stable")[: avas.nactv].tolist())
    )
    assert avas_mpq.rdm_info_indices == expected
    expected_gamma1_no = avas_mpq.U.T @ g1 @ avas_mpq.U
    assert avas_mpq.Gamma1_mo == approx(g1)
    assert avas_mpq.Gamma1_no == approx(expected_gamma1_no)
    assert avas_mpq.Gamma1 == approx(expected_gamma1_no)
    assert avas_mpq.Γ1 == approx(expected_gamma1_no)
    assert avas_mpq.occs == approx(np.diag(expected_gamma1_no))
    assert avas_mpq.C_no == approx(mp2.C_no)
    assert avas_mpq.no_occs == approx(mp2.no_occs)
    assert avas_mpq.U == approx(mp2.U_no)

    assert scf.E == approx(erhf)
    assert mp2.E_total == approx(emp2)
    assert mp2_rdm_E == approx(emp2)


def test_rhf_mp2_1rdm_does_not_store_t2():
    erhf = -76.0614664072629
    emp2 = -76.3710978833093
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)(system)
    mp2 = RMP2(store_t2=False)(scf)
    mp2.run()

    g1 = mp2.make_1rdm()

    assert scf.E == approx(erhf)
    assert mp2.E_total == approx(emp2)
    assert np.trace(g1) == approx(scf.na + scf.nb)
    assert_t2_not_stored(mp2)


def test_h4_rhf_mp2():
    erhf = -1.998839903161
    emp2 = -2.0915387810627
    xyz = """
  H   -2.7270878    1.9884277    1.0000000
  H   -1.8074993    2.0159410    -1.0000000
  H   -1.8213175    1.0960448    0.0000000
  H   -2.7409060    1.0685315    0.0000000
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = RHF(charge=0)(system)
    mp2 = RMP2()(scf)
    mp2.run()

    assert scf.E == approx(erhf)
    assert mp2.E_total == approx(emp2)


def test_singlet_rohf_mp2():
    erohf = -76.061466407194
    emp2 = -76.37109788330923
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, ms=0)(system)
    mp2 = ROMP2()(scf)
    mp2.run()

    assert scf.E == approx(erohf)
    assert mp2.E_total == approx(emp2)


def test_sd_sf_cumulants():
    euhf = -76.061466407177
    emp2 = -76.3710978831473
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, ms=0)(system)
    mp2 = UMP2(store_t2=True)(scf)
    mp2.run()

    lambda2_sf = mp2._make_mp2_sf_2cumulants(mp2.make_1rdm_sf(), mp2.make_2rdm_sf())
    lambda2_aa, lambda2_ab, lambda2_bb = mp2.make_2cumulant_sd()
    lambda2_sf_from_sd = (
        lambda2_aa + lambda2_bb + lambda2_ab + lambda2_ab.transpose(1, 0, 3, 2)
    )

    assert scf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)
    assert np.allclose(lambda2_sf, lambda2_sf_from_sd, atol=1e-10)


def test_triplet_h2o_rohf_mp2():
    erohf = -75.805109024040
    emp2 = -76.0662395867740

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, ms=1)(system)
    mp2 = ROMP2()(scf)
    mp2.run()

    assert scf.E == approx(erohf)
    assert mp2.E_total == approx(emp2)


def test_triplet_h2o_uhf_mp2():
    euhf = -75.810772399321
    emp2 = -76.0662395867740
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, ms=1)(system)
    mp2 = UMP2()(scf)
    mp2.run()

    assert scf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)


def test_triplet_h2o_uhf_mp2_1rdm_does_not_store_t2():
    euhf = -75.810772399321
    emp2 = -76.0662395867740
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, ms=1)(system)
    mp2 = UMP2(store_t2=False)(scf)
    mp2.run()

    assert scf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)
    assert np.trace(mp2.make_1rdm_sf()) == approx(scf.na + scf.nb)
    assert_t2_not_stored(mp2)

def test_h2o_uhf_mp2():
    euhf = -76.061466407177
    emp2 = -76.3710978831473
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, ms=0)(system)
    mp2 = UMP2()(scf)
    mp2.run()

    assert scf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)


def test_ump2_mpq_linear_and_quadratic_terms():
    euhf = -76.0217659883263
    emp2 = -76.221819034
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    scf = UHF(charge=0, ms=0)(system)
    mp2 = UMP2(store_t2=True)(scf)
    mp2.run()

    identity = np.eye(mp2.nmo)
    linear = UMP2MPQOnTheFly(
        mp2, Ua=identity, Ub=identity, include_quadratic=False
    )
    full = UMP2MPQOnTheFly(
        mp2, Ua=identity, Ub=identity, include_quadratic=True
    )

    a = mp2.naocc
    b = mp2.nbocc
    assert linear.lambda2_aa_linear_elem(0, 1, a, a + 1) == approx(
        mp2.t2_a[0, 1, 0, 1]
    )
    assert linear.lambda2_bb_linear_elem(0, 1, b, b + 1) == approx(
        mp2.t2_b[0, 1, 0, 1]
    )
    assert linear.lambda2_ab_elem(0, 0, a, b) == approx(
        mp2.t2_ab[0, 0, 0, 0]
    )
    assert full.lambda2_ab_elem(0, b, 0, b) == approx(
        full.lambda2_ab_linear_elem(0, b, 0, b)
        + full.lambda2_ab_quadratic_elem(0, b, 0, b)
    )
    assert abs(full.lambda2_ab_quadratic_elem(0, b, 0, b)) > 0.0

    assert scf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)


def test_ump2_common_natural_orbitals_and_mpq_wrapper():
    # Reference energies kept for now, but not asserted in this construction test.
    # euhf = -1.131269839709
    # emp2 = -1.153966321003
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.4
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    scf = UHF(charge=0, ms=0)(system)
    mp2 = UMP2(store_t2=True)(scf)
    mp2.run()

    gamma1 = mp2.make_1rdm_sd()
    no_transform = mp2.make_natural_orbital_transform(gamma1)
    C_no, occupations, Ua, Ub = no_transform

    S = system.ints_overlap()
    gamma1_no_a, gamma1_no_b = mp2.make_1rdm_no_sd(gamma1, no_transform)
    gamma1_no = mp2.make_1rdm_no_sf(gamma1, no_transform)

    # assert scf.E == approx(euhf)
    # assert mp2.E_total == approx(emp2)
    assert C_no.T @ S @ C_no == approx(np.eye(mp2.nmo))
    assert gamma1_no == approx(np.diag(occupations))
    assert gamma1_no == approx(gamma1_no_a + gamma1_no_b)

    gamma1_no_bundle, gamma2_no_bundle, lambda2_no_bundle = mp2.make_cumulants_no_sd()
    gamma2_no_sf = mp2._make_mp2_sf_2rdm(*gamma2_no_bundle)
    lambda2_no_sf = mp2._make_mp2_sf_2cumulants(
        gamma1_no_bundle[0] + gamma1_no_bundle[1], gamma2_no_sf
    )
    lambda2_no_sf_from_sd = (
        lambda2_no_bundle[0]
        + lambda2_no_bundle[2]
        + lambda2_no_bundle[1]
        + lambda2_no_bundle[1].transpose(1, 0, 3, 2)
    )

    assert gamma1_no_bundle[0] + gamma1_no_bundle[1] == approx(gamma1_no)
    assert lambda2_no_sf == approx(lambda2_no_sf_from_sd)

    mpq = ump2_mpq_onthefly_no(mp2, mo_range=(0, 2))
    assert isinstance(mpq, UMP2MPQOnTheFly)
    assert mpq.C_no == approx(C_no)
    assert mpq.no_occs == approx(occupations)
    assert mpq.Ua == approx(Ua)
    assert mpq.Ub == approx(Ub)
    assert mpq.gamma1_mo_a == approx(gamma1[0])
    assert mpq.gamma1_mo_b == approx(gamma1[1])
    assert mpq.gamma1_a == approx(gamma1_no_a)
    assert mpq.gamma1_b == approx(gamma1_no_b)
    assert mpq.γa == approx(gamma1_no_a)
    assert mpq.γb == approx(gamma1_no_b)
    assert mpq.Gamma1_no == approx(gamma1_no)
    assert mpq.Gamma1 == approx(gamma1_no)
    assert mpq.Γ1 == approx(gamma1_no)
    assert mpq.occs == approx(occupations)
    assert mpq.rdm_info_indices == (0, 1)
    assert mpq.rdm_info_selection == "mo_range"

    M1 = mpq.make_M1()
    M2 = mpq.make_M2()
    assert M1.shape == (mp2.nmo,)
    assert M2.shape == (mp2.nmo, mp2.nmo)
    assert np.all(M1 >= -1e-12)
    assert np.count_nonzero(M1[2:]) == 0
    assert np.count_nonzero(M2[2:, :]) == 0
    assert np.count_nonzero(M2[:, 2:]) == 0

    occupation_mpq = ump2_mpq_onthefly_no(
        mp2, occupation_window=(0.02, 1.98)
    )
    expected = tuple(
        np.flatnonzero((occupations >= 0.02) & (occupations <= 1.98)).tolist()
    )
    assert occupation_mpq.rdm_info_indices == expected
    assert occupation_mpq.rdm_info_selection == "natural_occupation"

    avas = AVAS(subspace=["H(1s)"])(RHF(charge=0)(system))
    with pytest.raises(TypeError, match="requires restricted orbitals"):
        ump2_mpq_onthefly_no(mp2, avas=avas)

    with pytest.raises(ValueError, match="Choose only one"):
        ump2_mpq_onthefly_no(
            mp2, mo_range=(0, 2), occupation_window=(0.02, 1.98)
        )

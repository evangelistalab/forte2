import time

import pytest
import numpy as np

from forte2 import System
from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals
from forte2.scf import RHF, ROHF, UHF
from forte2.helpers.comparisons import approx
from forte2.mp import RMP2, ROMP2, UMP2
from forte2.props import RMP2MPQFast, UMP2MPQFast, MutualCorrelationAnalysis


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

    moints = RestrictedMOIntegrals(system, scf.C[0], list(range(scf.nmo)))
    Ecore = moints.E
    H = moints.H
    V = moints.V

    mp2_rdm_E = mp2.energy_given_rdms(Ecore, H, V, g1, g2)

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


@pytest.mark.skip(reason="ROMP2 canonicalization under construction")
def test_triplet_h2o_rohf_mp2():
    erohf = -75.805109024040
    emp2 = -76.0707816462552

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


def test_triplet_h2o_uhf_mp2_rdms():
    euhf = -75.810772399321
    emp2 = -76.0662395867740
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, ms=1)(system)
    mp2 = UMP2(store_t2=True)(scf)
    mp2.run()

    assert scf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)
    assert_uhf_rdm_invariants(mp2, scf.na, scf.nb)


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

    g1 = mp2.make_1rdm()

    assert scf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)
    assert np.trace(mp2.gamma1_a) == approx(scf.na)
    assert np.trace(mp2.gamma1_b) == approx(scf.nb)
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


def test_fast_mpq():
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

    fast = UMP2MPQFast(mp2)

    summary = fast.MPQ_matrix_summary()

    print(summary)

    assert scf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)


def test_fast_mpq_rmp2():
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    scf = RHF(charge=0)(system)
    mp2 = RMP2(store_t2=True)(scf)
    mp2.run()

    fast = RMP2MPQFast(mp2)
    t2_ss = mp2.t2 - mp2.t2.transpose(0, 1, 3, 2)

    for i in range(mp2.nocc):
        for j in range(mp2.nocc):
            for a in range(mp2.nvir):
                for b in range(mp2.nvir):
                    p = i
                    q = j
                    r = mp2.nocc + a
                    s = mp2.nocc + b

                    assert fast.lambda2_aa_elem(p, q, r, s) == approx(t2_ss[i, j, a, b])
                    assert fast.lambda2_bb_elem(p, q, r, s) == approx(t2_ss[i, j, a, b])
                    assert fast.lambda2_ab_elem(p, q, r, s) == approx(
                        mp2.t2[i, j, a, b]
                    )

                    assert fast.lambda2_aa_elem(r, s, p, q) == approx(t2_ss[i, j, a, b])
                    assert fast.lambda2_bb_elem(r, s, p, q) == approx(t2_ss[i, j, a, b])
                    assert fast.lambda2_ab_elem(r, s, p, q) == approx(
                        mp2.t2[i, j, a, b]
                    )

                    c_ref = 0.25 * (
                        abs(t2_ss[i, j, a, b]) ** 2
                        + abs(t2_ss[i, j, a, b]) ** 2
                        + abs(mp2.t2[i, j, a, b]) ** 2
                        + abs(mp2.t2[i, j, b, a]) ** 2
                        + abs(mp2.t2[j, i, a, b]) ** 2
                        + abs(mp2.t2[j, i, b, a]) ** 2
                    )
                    assert fast.C_elem(p, q, r, s) == approx(c_ref)

    assert fast.lambda2_aa_elem(0, 0, 0, 0) == approx(0.0)
    assert fast.lambda2_ab_elem(0, 0, 0, 0) == approx(0.0)

    M1, M2 = fast.make_measures()
    assert M1.shape == (scf.nmo,)
    assert M2.shape == (scf.nmo, scf.nmo)
    assert np.max(np.abs(M2 - M2.T)) == approx(0.0)
    assert np.max(np.abs(np.diag(M2))) == approx(0.0)
    assert fast.make_M1() is fast.M1
    assert fast.make_M2() is fast.M2
    assert fast.make_M1() == approx(M1)
    assert fast.make_M2() == approx(M2)
    assert "Mutual Correlation Matrix M2" in fast.MPQ_matrix_summary()


test_fast_mpq_rmp2()

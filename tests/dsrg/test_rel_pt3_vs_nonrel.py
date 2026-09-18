import numpy as np

from forte2 import CISolver, MCOptimizer, RHF, RelCISolver, RelState, State, System
from forte2.dsrg import DSRG_MRPT3, RelDSRG_MRPT3
from forte2.orbitals import SpinorUpcaster
from forte2.helpers.comparisons import approx, approx_abs

# H2O CAS(8e,6o) on a nonrelativistic Hamiltonian: no GAS anywhere, so the two
# formalisms describe the same wave function and must give the same energy.
#
# They once differed by 2.4e-6 Eh here, because `_RelDSRGHelper` was missing
# `H1_T2_C1_non_od` and so dropped the one-body part of [H0th, T2] wherever
# rel_dsrg_mrpt3.py commutes the block-diagonal Fock with the amplitudes. Every
# third-order term was affected; the second-order one was not. N2 CAS(6e,6o),
# used by test_dsrg_mrpt3.py::test_sf_mrpt3_matches_two_component, keeps that
# term under 1e-8 and so never caught it -- which is why this system is here.

XYZ = """
O
H  1 1.00
H  1 1.00 2 103.1
"""
CORE = [0]
ACTV = [1, 2, 3, 4, 5, 6]
TERMS = ("e_dsrg_mrpt2", "e_dsrg_mrpt3_1", "e_dsrg_mrpt3_2", "e_dsrg_mrpt3_3")


def _system():
    return System(xyz=XYZ, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")


def _spinor(orbitals):
    return [s for i in orbitals for s in (2 * i, 2 * i + 1)]


def _run(relativistic):
    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-10)(_system())
    rhf.run()
    if relativistic:
        parent = SpinorUpcaster(apply_random_phase=True, rng=1234)(rhf)
        parent.run()
        ci_solver = RelCISolver(
            states=RelState(nel=10),
            core_orbitals=_spinor(CORE),
            active_orbitals=_spinor(ACTV),
        )
        dsrg_cls = RelDSRG_MRPT3
    else:
        parent = rhf
        ci_solver = CISolver(
            states=State(nel=10, multiplicity=1, ms=0.0),
            core_orbitals=CORE,
            active_orbitals=ACTV,
        )
        dsrg_cls = DSRG_MRPT3
    mc = MCOptimizer(ci_solver, maxiter=500, e_tol=1e-12, g_tol=1e-10)(parent)
    mc.run()
    dsrg = dsrg_cls(flow_param=1.0)(mc)
    dsrg.run()
    terms = {t: float(np.real(getattr(dsrg, t))) for t in TERMS}
    return float(np.real(dsrg.ints["E"])), terms, float(np.real(dsrg.E_dsrg))


def test_nonrel_pt3_matches_forte_term_by_term():
    """The spin-free side is the correct one, term by term against forte.

    forte, C1/DF/cc-pVDZ, restricted_docc [1], active [6], dsrg_s 1.0,
    relax_ref none. Its spin-adapted and spin-integrated MRPT3 agree here to
    4e-13, so there is no convention ambiguity to worry about.
    """
    eref_forte = -76.078597606085
    terms_forte = {
        "e_dsrg_mrpt2": -0.142745303396,
        "e_dsrg_mrpt3_1": +0.000062189642,
        "e_dsrg_mrpt3_2": -0.011898758811,
        "e_dsrg_mrpt3_3": -0.001318334227,
    }
    total_forte = -76.234497812876

    eref, terms, total = _run(relativistic=False)

    assert eref == approx(eref_forte)
    for name, ref in terms_forte.items():
        assert terms[name] == approx(ref)
    assert total == approx(total_forte)


def test_rel_pt3_matches_nonrel_for_a_cas_reference():
    """With no GAS involved the two formalisms must agree, term by term.

    Compared per term rather than on the total, so that a regression names the
    stage at fault instead of showing a shifted energy.
    """
    eref_nr, terms_nr, total_nr = _run(relativistic=False)
    eref_rel, terms_rel, total_rel = _run(relativistic=True)

    # The references are identical, so nothing upstream of the DSRG differs.
    assert eref_rel == approx_abs(eref_nr, 1e-10)
    assert terms_rel["e_dsrg_mrpt2"] == approx_abs(terms_nr["e_dsrg_mrpt2"], 1e-8)

    for name in ("e_dsrg_mrpt3_1", "e_dsrg_mrpt3_2", "e_dsrg_mrpt3_3"):
        assert terms_rel[name] == approx_abs(terms_nr[name], 1e-8)
    assert total_rel == approx_abs(total_nr, 1e-8)

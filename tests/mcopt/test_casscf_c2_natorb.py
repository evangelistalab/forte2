from forte2 import CISolver, MCOptimizer, RHF, State, System, AVAS
from forte2.helpers.comparisons import approx


def test_casscf_c2_natorb():
    # without ci vector rotation this can converge to a different root
    xyz = """
    C 0.0 0.0 0.0
    C 0.0 0.0 2.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVTZ",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0)(system)

    avas = AVAS(
        selection_method="separate",
        num_active_docc=4,
        num_active_uocc=4,
        subspace=["C(2s)", "C(2p)"],
    )(rhf)

    ci = CISolver(State(system=system, multiplicity=1, ms=0.0))
    mc = MCOptimizer(ci, final_orbitals="natural")(avas)
    mc.run()
    assert mc.E == approx(-75.4241729220)

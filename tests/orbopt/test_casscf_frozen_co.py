from forte2 import System, AVAS, MCOptimizer, State, RHF
from forte2.helpers.comparisons import approx


def test_casscf_frozen_co():
    emcscf = -112.8641406910
    emcscf_frz = -112.8633865369

    xyz = """
    C 0.0 0.0 0.0
    O 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        subspace=["C(2p)", "O(2p)"],
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=3,
    )(rhf)

    mc = MCOptimizer(
        states=State(system=system, multiplicity=1, ms=0.0),
        frozen_core_orbitals=[0],
        core_orbitals=[1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        frozen_virtual_orbitals=[25, 26, 27],
        optimize_frozen_orbs=True,
    )(avas)
    mc.run()
    assert mc.E == approx(emcscf)

    mc = MCOptimizer(
        states=State(system=system, multiplicity=1, ms=0.0),
        frozen_core_orbitals=1,
        core_orbitals=3,
        active_orbitals=6,
        frozen_virtual_orbitals=3,
        optimize_frozen_orbs=False,
    )(avas)
    mc.run()
    assert mc.E == approx(emcscf_frz)

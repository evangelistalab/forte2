from forte2 import System, RHF
from forte2.orbitals.iao import IAO
from forte2.props.props import iao_partial_charge
from forte2.helpers.comparisons import approx


def test_iao_hcn():
    # geometry from cccbdb
    xyz = """
    C 0 0 0
    H 0 0 1.0640
    N 0 0 -1.1560
    """

    system = System(xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    C_occ = rhf.C[0][:, :7]
    iao = IAO(system, C_occ)
    g1_sf_iao = iao.make_sf_1rdm(2 * rhf.D[0])
    _, charges = iao_partial_charge(system, g1_sf_iao)

    assert charges == approx(
        [-0.007844869262523702, 0.21755558427361243, -0.2097107150110915]
    )


def test_iao_ch4():
    # geometry from cccbdb
    xyz = """
    C	0.0000	0.0000	0.0000
    H	0.6276	0.6276	0.6276
    H	0.6276	-0.6276	-0.6276
    H	-0.6276	0.6276	-0.6276
    H	-0.6276	-0.6276	0.6276
    """

    system = System(xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    C_occ = rhf.C[0][:, : rhf.ndocc]
    iao = IAO(system, C_occ)
    g1_sf_iao = iao.make_sf_1rdm(2 * rhf.D[0])
    _, charges = iao_partial_charge(system, g1_sf_iao)
    assert charges == approx(
        [
            -0.5280212871620078,
            0.13200532179051516,
            0.13200532179051516,
            0.13200532179051516,
            0.13200532179051516,
        ]
    )

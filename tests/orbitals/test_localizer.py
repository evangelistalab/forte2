import numpy as np
import pytest

from forte2 import AVAS, CI, CISolver, Localizer, RHF, ROHF, State, System
from forte2.helpers.comparisons import approx, approx_abs


def test_localizer_water_rhf():
    xyz = """
    O
    H 1 1.1
    H 1 1.1 2 104.5
    """
    system = System(xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    loc = Localizer()(rhf)
    loc.run()
    assert loc.blocks == [[0, 1, 2, 3, 4]]
    assert "mo_space" not in loc.provides
    for occ, ref in zip(loc.mos.occupations, rhf.mos.occupations):
        assert np.array_equal(occ, ref)

    # the localized orbitals span the same density, so they give the RHF energy
    C_occ = loc.mos.C[0][:, :5]
    E = np.einsum("pq,pq->", C_occ @ C_occ.T, rhf.F[0] + system.ints_hcore())
    assert E + system.nuclear_repulsion == approx(rhf.E)

    # the oxygen core and two lone pairs, and two O-H bonds
    pops = loc.populations[0]
    on_oxygen = pops[0] > 0.99
    assert on_oxygen.sum() == 3
    bonds = pops[:, ~on_oxygen]
    assert bonds[0] == approx_abs([0.663534, 0.663534], 1e-6)
    assert bonds[1:].max(axis=0) == approx_abs([0.336427, 0.336427], 1e-6)


@pytest.mark.parametrize("charges, exponent", [("iao", 4), ("mulliken", 2)])
def test_localizer_acrylic_acid(charges, exponent):
    # geometry from http://www.iboview.org/bin/ibo-ref.20141030.tar.bz2
    xyz = """
    C                 1.9750101       -0.0359056        0.0000000
    C                 0.8155445        0.6290773        0.0000000
    C                -0.4769136       -0.0990005        0.0000000
    O                -0.6246729       -1.3065879        0.0000000
    O                -1.5290139        0.7740207        0.0000000
    H                 1.9795702       -1.1269710        0.0000000
    H                 2.9318898        0.4848390        0.0000000
    H                 0.7696874        1.7186370        0.0000000
    H                -2.3334997        0.2190058        0.0000000
    """
    system = System(xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    loc = Localizer(charges=charges, exponent=exponent)(rhf)
    loc.run()

    C_occ = loc.mos.C[0][:, : rhf.ndocc]
    E = np.einsum("pq,pq->", C_occ @ C_occ.T, rhf.F[0] + system.ints_hcore())
    assert E + system.nuclear_repulsion == approx(rhf.E)

    # the atoms holding more than 0.15 of each IBO reproduce the Lewis structure
    centers = [tuple(np.flatnonzero(pop > 0.15)) for pop in loc.populations[0].T]
    cores_and_lone_pairs = [(0,), (1,), (2,), (3,), (3,), (3,), (4,), (4,), (4,)]
    heavy_atom_bonds = [(0, 1), (0, 1), (1, 2), (2, 3), (2, 3), (2, 4)]
    bonds_to_h = [(0, 5), (0, 6), (1, 7), (4, 8)]
    assert sorted(centers) == sorted(
        cores_and_lone_pairs + heavy_atom_bonds + bonds_to_h
    )


def test_localizer_h4_chain():
    xyz = "\n".join(f"H 0 0 {2.0 * i}" for i in range(4))
    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    rohf = ROHF(charge=0, ms=1.0)(system)
    avas = AVAS(subspace=["H(1s)"], cas=(4, 4))(rohf)

    # by default, the active space is localized, one orbital on each H
    loc = Localizer()(avas)
    loc.run()
    assert loc.blocks == [[0, 1, 2, 3]]
    assert sorted(loc.populations[0].argmax(axis=0)) == [0, 1, 2, 3]
    assert loc.populations[0].max(axis=0).min() > 0.99
    assert loc.mos.occupations is None
    assert np.allclose(avas.mos.C[0] @ loc.U, loc.mos.C[0])

    # rotations within the active space leave the CASCI energy unchanged
    energies = []
    for parent in (avas, loc):
        ci_solver = CISolver(State(nel=4, multiplicity=3, ms=0.0))
        ci = CI(ci_solver)(parent)
        ci.run()
        energies.append(ci.E)
    assert energies[1] == approx(energies[0])

    split = Localizer(split_by_occupation=True)(avas)
    split.run()
    assert split.blocks == [[0], [1, 2], [3]]
    for occ, ref in zip(split.mos.occupations, avas.mos.occupations):
        assert np.array_equal(occ, ref)

    after_ci = Localizer(split_by_occupation=True)(ci)
    with pytest.raises(ValueError, match="reference occupations"):
        after_ci.run()

    subset = Localizer(orbitals=[1, 2])(avas)
    subset.run()
    assert np.array_equal(subset.mos.C[0][:, [0, 3]], avas.mos.C[0][:, [0, 3]])

    virtual = Localizer(orbitals=[4, 5])(avas)
    with pytest.raises(ValueError, match="don't represent"):
        virtual.run()

    # Lowdin charges localize virtual orbitals, here one on each H
    lowdin = Localizer(charges="lowdin", orbitals=[4, 5, 6, 7])(avas)
    lowdin.run()
    assert sorted(lowdin.populations[0].argmax(axis=0)) == [0, 1, 2, 3]
    assert lowdin.populations[0].max(axis=0).min() > 0.9

    # without an MO space, the doubly and singly occupied orbitals are localized separately
    occupied = Localizer()(rohf)
    occupied.run()
    assert occupied.blocks == [[0], [1, 2]]
    for occ, ref in zip(occupied.mos.occupations, rohf.mos.occupations):
        assert np.array_equal(occ, ref)

    mixed = Localizer(orbitals=[0, 1, 2])(rohf)
    mixed.run()
    assert mixed.mos.occupations is None


def test_localizer_rejects_symmetry_and_bad_orbitals():
    system = System(
        xyz="N 0 0 0\nN 0 0 1.1",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )
    rhf = RHF(charge=0)(system)
    loc = Localizer()(rhf)
    with pytest.raises(ValueError, match="point-group symmetry"):
        loc.run()
    with pytest.raises(ValueError, match="more than once"):
        Localizer(orbitals=[[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="list of MO indices"):
        Localizer(orbitals=[0, [1, 2]])
    with pytest.raises(ValueError, match="charges"):
        Localizer(charges="boys")
    with pytest.raises(ValueError, match="exponent"):
        Localizer(exponent=3)

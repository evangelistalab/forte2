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
    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
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
    assert bonds[0] == approx_abs([0.654978, 0.654978], 1e-6)
    assert bonds[1:].max(axis=0) == approx_abs([0.344953, 0.344953], 1e-6)


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

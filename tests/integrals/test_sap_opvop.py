from pathlib import Path

import numpy as np
import pytest

from forte2 import System, integrals
from forte2.system.build_basis import build_sap_potential_basis


THIS_DIR = Path(__file__).parent


def test_sap_density_normalization_and_opvop_reference():
    """Compare the SAP small-component kernel with Libint's q_gau reference."""
    system = System(
        xyz="O 0 0 0\nO 0 0 2\nH 0 -1 -1\nH 0 1 3",
        basis_set="sto-3g",
        minao_basis_set=None,
        unit="bohr",
    )
    sap_basis = build_sap_potential_basis("sap_grasp_large", system.geom_helper)

    # Every contracted SAP density integrates to the corresponding atomic charge.
    integrated_charges = [
        np.sum(
            np.asarray(shell.coeff)
            * (np.pi / np.asarray(shell.exponents)) ** 1.5
        )
        for shell in sap_basis
    ]
    assert integrated_charges == pytest.approx([8.0, 8.0, 1.0, 1.0], abs=1.0e-12)

    W = integrals.coulomb_3c_opVop(system, sap_basis, system.basis, system.basis)
    assert np.allclose(W[0], W[0].T, atol=1.0e-12)
    for component in W[1:]:
        assert np.allclose(component, -component.T, atol=1.0e-12)

    # Libint's external q_gau test uses the Cartesian p ordering (px, py, pz),
    # while Forte2 uses (py, pz, px). Reorder Forte2 AOs to the reference order.
    reference_to_forte = np.array([0, 1, 4, 2, 3, 5, 6, 9, 7, 8, 10, 11])
    W_ref_order = [
        component[np.ix_(reference_to_forte, reference_to_forte)] for component in W
    ]

    # The scalar tolerance accounts for the slightly different STO-3G rounding in
    # Forte2's bundled BSE data. Vector components agree more closely.
    assert W_ref_order[0][0, 0] == pytest.approx(1017.1940316588756, abs=2.0e-4)
    assert W_ref_order[0][3, 3] == pytest.approx(59.23868921893008, abs=2.0e-6)
    assert W_ref_order[1][3, 0] == pytest.approx(-1.5576470863370833, abs=2.0e-7)
    assert W_ref_order[2][2, 0] == pytest.approx(1.5576470863370833, abs=2.0e-7)
    assert W_ref_order[3][2, 0] == pytest.approx(0.04151583448537164, abs=2.0e-7)
    assert W_ref_order[3][3, 2] == pytest.approx(-7.639347452209898, abs=2.0e-7)

    if integrals.LIBCINT_AVAILABLE:
        W_cint = integrals.cint_coulomb_3c_opVop(
            system, sap_basis, system.basis
        )
        for component_cint, component_libint in zip(W_cint, W):
            assert np.allclose(component_cint, component_libint, atol=2.0e-10)


@pytest.mark.skipif(not integrals.LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_sap_opvop_high_l():
    """The Libcint kernel supports angular momenta beyond Libint's derivative limit."""
    system = System(
        xyz="H 0 0 0\nH 0 0 1.0",
        basis_set="sap_helfem_large",
        auxiliary_basis_set=str(THIS_DIR / "high_l.json"),
        minao_basis_set=None,
    )

    W = integrals.cint_coulomb_3c_opVop(
        system, system.basis, system.auxiliary_basis
    )
    assert all(component.shape == (60, 60) for component in W)
    assert all(np.isfinite(component).all() for component in W)
    assert np.linalg.norm(W[0]) == pytest.approx(6398.564518450337, rel=1.0e-12)

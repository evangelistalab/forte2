import pytest
import forte2
import numpy as np
from forte2.helpers.comparisons import approx
from forte2.system.build_basis import BSE_AVAILABLE


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_1e_ovlp():
    xyz = """

    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    # Create system in Forte2
    system = forte2.System(xyz=xyz, basis="ano-rcc", auxiliary_basis=None)
    # Forte2 integrals
    S = forte2.ints.overlap(system.basis)
    assert np.linalg.norm(S) == approx(16.59004845412261)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_1e_hcore():
    xyz = """

    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    # Create system in Forte2
    system = forte2.System(xyz=xyz, basis="ano-pvtz", auxiliary_basis=None)
    # Forte2 integrals
    T = forte2.ints.kinetic(system.basis)
    V = forte2.ints.nuclear(system.basis, system.atoms)
    H = T + V
    assert np.linalg.norm(H) == approx(48.018559135658165)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_2e_eri():
    xyz = """

    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    # Create system in Forte2
    system = forte2.System(xyz=xyz, basis="DZ (Dunning-Hay)", auxiliary_basis=None)
    # Forte2 integrals
    eri = forte2.ints.coulomb_4c(system.basis)
    assert np.linalg.norm(eri.flatten()) == approx(25.67187172762279)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_custom_basis_assignment():
    # Test for custom basis assignment
    xyz = """
    H  1  1  1
    P  1  1 -1
    V  1 -1  1
    Mn  1 -1 -1
    Mn -1  1  1
    V -1  1 -1
    P -1 -1  1
    H -1 -1 -1 
    """
    system = forte2.System(
        xyz=xyz,
        basis={"H": "cc-pvdz", "P": "ano-pv5z", "default": "sap_helfem_large"},
        unit="bohr",
    )
    assert system.nbf() == 204

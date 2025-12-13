import pytest
import numpy as np
from pathlib import Path

THIS_DIR = Path(__file__).parent

from forte2 import System, ints
from forte2.helpers.comparisons import approx
from forte2.system.build_basis import BSE_AVAILABLE


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_1e_ovlp():
    xyz = """
    O 0.000000000000  0.000000000000 -0.061664597388
    H 0.000000000000 -0.711620616369  0.489330954643
    H 0.000000000000  0.711620616369  0.489330954643
    """

    # Create system in Forte2
    system = System(xyz=xyz, basis_set="ano-rcc", auxiliary_basis_set=None)
    # Forte2 integrals
    S = ints.overlap(system.basis)
    assert np.linalg.norm(S) == approx(16.59004845412261)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_1e_hcore():
    xyz = """
    O 0.000000000000  0.000000000000 -0.061664597388
    H 0.000000000000 -0.711620616369  0.489330954643
    H 0.000000000000  0.711620616369  0.489330954643
    """

    # Create system in Forte2
    system = System(xyz=xyz, basis_set="ano-pvtz", auxiliary_basis_set=None)
    # Forte2 integrals
    T = ints.kinetic(system.basis)
    V = ints.nuclear(system.basis, system.atoms)
    H = T + V
    assert np.linalg.norm(H) == approx(48.018559135658165)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_2e_eri():
    xyz = """
    O 0.000000000000  0.000000000000 -0.061664597388
    H 0.000000000000 -0.711620616369  0.489330954643
    H 0.000000000000  0.711620616369  0.489330954643
    """

    # Create system in Forte2
    system = System(xyz=xyz, basis_set="DZ (Dunning-Hay)", auxiliary_basis_set=None)
    # Forte2 integrals
    eri = ints.coulomb_4c(system.basis)
    assert np.linalg.norm(eri.flatten()) == approx(25.67187172762279)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_custom_basis_assignment():
    # Test for custom basis assignment
    # also test for combinations of line breaks with ; and \n, as well as extra spaces
    xyz = """
    H  1  1  1\nP  1  1 -1
    V  1 -1  1; Mn  1 -1 -1
Mn -1  1  1
            V -1  1 -1
    P     -1.00000 -1  1
    H -1 -1 -1 
    """
    system = System(
        xyz=xyz,
        basis_set={"H": "cc-pvdz", "P": "ano-pv5z", "default": "sap_helfem_large"},
        unit="bohr",
    )
    assert system.nbf == 204


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_custom_basis_with_decontract():
    xyz = """
    C 0 0 0
    O 0 0 1.2
    H 0 0 1.5
    N 0 0 1.7
    H 0 0 1.8
    C 0 0 2.0
    O 0 0 2.2
    H 0 0 2.5
    N 0 0 3.0
    """
    hbas = str(THIS_DIR / "cc-pvdz-trunc.json") + "::H"
    system = System(
        xyz=xyz,
        basis_set={
            "C1": "decon-cc-pvdz",
            "O": "sto-6g::N",
            "C2": "cc-pvtz",
            "H2-3": hbas,
            "N2": "decon-def2-svp",
            "default": "ano-r0",
        },
        auxiliary_basis_set={
            "C": "cc-pVQZ-JKFIT",
            "O1": "decon-def2-universal-JKFIT",
            "default": "def2-universal-JKFIT",
        },
    )
    assert len(system.basis) == 100
    assert len(system.auxiliary_basis) == 594


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_custom_basis_assignment_with_autoaux():
    # Test for custom basis assignment with autoaux
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
    system = System(
        xyz=xyz,
        basis_set={"H": "cc-pvdz", "P": "ano-pv5z", "default": "sap_helfem_large"},
        auxiliary_basis_set={"H": "cc-pvtz-jkfit", "default": "ano-rcc-vdz-autoaux"},
        unit="bohr",
    )
    assert system.nbf == 204
    assert system.naux == 1670

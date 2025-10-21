from scipy.linalg import eigh
import numpy as np

import forte2
from forte2.helpers.comparisons import approx


def test_one_electron_integrals():
    xyz = "H  0.0 0.0 0.0"

    system = forte2.System(xyz=xyz, basis_set="cc-pvdz")

    S = forte2.integrals.overlap(system)
    T = forte2.integrals.kinetic(system)
    V = forte2.integrals.nuclear(system)

    H = T + V

    # Solve the generalized eigenvalue problem H C = S C ε
    ε, _ = eigh(H, S)
    assert ε[0] == approx(-0.4992784)


def test_gaussian_charge_distribution_integrals_1():
    xyz = """Au 0.0 0.0 0.0
    Au 0.0 0.0 1.0
    """

    system = forte2.System(
        xyz=xyz,
        basis_set="ano-r0",
        use_gaussian_charges=True,
        minao_basis_set=None,
    )

    V = forte2.integrals.nuclear(system)
    print(np.linalg.eigvalsh(V)[0])
    assert np.linalg.eigvalsh(V)[0] == approx(-9218.63624025741)
    # From pyscf:
    # mol = pyscf.gto.M(atom = """Au 0 0 0; Au 0 0 1.0""", basis ='ano-r0', charge=0, spin=0, nucmod="g")
    # np.linalg.eigvalsh(mol.intor("int1e_nuc"))[0]


def test_gaussian_charge_distribution_integrals_2():
    import json
    from importlib import resources

    with resources.files("forte2.data").joinpath("otterbein_symmetry_db.json").open(
        "r"
    ) as f:
        bfile = json.load(f)

    xyz = bfile["30"]["xyz"]
    system = forte2.System(
        xyz=xyz, basis_set="ano-r0", minao_basis_set=None, use_gaussian_charges=True
    )
    V = forte2.integrals.nuclear(system)
    assert np.linalg.eigvalsh(V)[0] == approx(-6690.905512582018)

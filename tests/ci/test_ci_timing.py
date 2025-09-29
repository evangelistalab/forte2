import time
import pytest

from forte2 import System, ROHF, CI, State
from forte2.helpers.comparisons import approx


def molecule(n, r=1.0):
    """Helper function to make a molecular geometry string for a linear chain of hydrogen atoms
    with a specified bond length `r`.
    """
    for i in range(n):
        yield f"H 0.0 0.0 {i * r}"


def timing(n):
    xyz = "\n".join(molecule(n))

    system = System(
        xyz=xyz,
        basis_set="sto-6g",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )

    multiplicity = 1 + (n % 2)  # Singlet for even n, doublet for odd n
    ms = 0.5 * (n % 2)  # Unpaired electrons for odd n
    rohf = ROHF(charge=0, ms=0.5 * (n % 2), econv=1e-12)(system)
    rohf.run()
    ci = CI(
        State(nel=n, multiplicity=multiplicity, ms=ms),
        active_orbitals=list(range(n)),
        econv=1e-12,
    )(rohf)
    start = time.monotonic()
    ci.run()
    end = time.monotonic()

    return end - start, ci.E[0]


@pytest.mark.slow
def test_ci_timing():
    """Test the CI energy and timing for hydrogen chains of length 2 to 12."""

    ref_energies = [
        -1.108873664804,  # H2
        -2.180967812817,  # H4
        -3.257608942865,  # H6
        -4.336068592474,  # H8
        -5.415397091940,  # H10
        -6.495197015363,  # H12
        -7.575276862289,  # H14
    ]

    ci_timing = []
    energies = []
    for n in range(2, 14, 2):
        elapsed, energy = timing(n)
        ci_timing.append((n, elapsed, energy))
        energies.append(energy)

    for n, elapsed, energy in ci_timing:
        print(
            f"Timing for {n} hydrogens: {elapsed:.2f} seconds, CI energy: {energy:.6f}"
        )
    for i, (energy, ref_energy) in enumerate(zip(energies, ref_energies)):
        assert energy == approx(ref_energy), (
            f"CI energy mismatch for {2 * (i + 1)} hydrogens: "
            f"{energy} != {ref_energy}"
        )


@pytest.mark.slow
def test_ci_timing2():
    """Test the CI energy and timing for hydrogen chain of length 13."""

    ref_energies = [-7.013625049615]  # H13

    ci_timing = []
    energies = []

    elapsed, energy = timing(13)
    ci_timing.append((13, elapsed, energy))
    energies.append(energy)

    for n, elapsed, energy in ci_timing:
        print(
            f"Timing for {n} hydrogens: {elapsed:.2f} seconds, CI energy: {energy:.6f}"
        )
    for i, (energy, ref_energy) in enumerate(zip(energies, ref_energies)):
        assert energy == approx(ref_energy), (
            f"CI energy mismatch for {2 * (i + 1)} hydrogens: "
            f"{energy} != {ref_energy}"
        )

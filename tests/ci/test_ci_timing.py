import time
import pytest
from forte2 import *
from forte2.helpers.comparisons import approx


def molecule(n, r=1.0):
    for i in range(n):
        yield f"H 0.0 0.0 {i * r}"


def timing(n):
    xyz = "\n".join(molecule(n))

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    multiplicity = 1 + (n % 2)  # Singlet for even n, doublet for odd n
    ms = 0.5 * (n % 2)  # Unpaired electrons for odd n
    rohf = ROHF(charge=0, ms=0.5 * (n % 2), econv=1e-12)(system)
    rohf.run()
    ci_state = CIStates(
        active_orbitals=list(range(n)),
        states=State(nel=n, multiplicity=multiplicity, ms=ms),
        nroots=1,
    )
    ci = CI(ci_state, econv=1e-12)(rohf)
    start = time.monotonic()
    ci.run()
    end = time.monotonic()

    return end - start, ci.E[0]


@pytest.mark.slow
def test_ci_timing():

    ref_energies = [
        -1.108873664804,
        -2.180967812817,
        -3.257608942865,
        -4.336068592474,
        -5.415397091940,
        -6.495197015363,
        -7.575276862289,
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
        # assert elapsed < 10, f"CI timing exceeded 10 seconds for {n} hydrogens"

    for i, (energy, ref_energy) in enumerate(zip(energies, ref_energies)):
        assert energy == approx(ref_energy), (
            f"CI energy mismatch for {2 * (i + 1)} hydrogens: "
            f"{energy} != {ref_energy}"
        )

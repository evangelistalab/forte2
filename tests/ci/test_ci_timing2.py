import time
from numpy import isclose
import pytest
from forte2.helpers.comparisons import approx


from forte2 import *


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

    ci = CI(ci_state, econv=1e-9)(rohf)
    start = time.monotonic()
    ci.run()
    end = time.monotonic()

    # assert isclose(rhf.E, -1.05643120731551)
    # assert isclose(ci.E[0], -1.096071975854)
    return end - start, ci.E[0]


@pytest.mark.slow
def test_ci_timing2():

    ref_energies = [-7.013625049615]

    ci_timing = []
    energies = []

    elapsed, energy = timing(13)
    ci_timing.append((13, elapsed, energy))
    energies.append(energy)

    for i, (energy, ref_energy) in enumerate(zip(energies, ref_energies)):
        assert energy == approx(ref_energy), (
            f"CI energy mismatch for {2 * (i + 1)} hydrogens: "
            f"{energy} != {ref_energy}"
        )
    for n, elapsed, energy in ci_timing:
        print(
            f"Timing for {n} hydrogens: {elapsed:.2f} seconds, CI energy: {energy:.6f}"
        )

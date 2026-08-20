"""
Transition-dipole / oscillator-strength interfacing for the relativistic DMRG
solver (complex, spin-orbit).

Same rationale as test_dmrg_transition_dipoles.py, but through the SGF|CPX
(general-spin, complex) block2 path used by RelDMRGSolver, exercised via the
``RelDMRG``/``do_transition_dipole`` convenience class (mirroring ``RelCI``).
"""

import pytest

from forte2 import System, GHF, RelCI, X2CParams
from forte2.dmrg import RelDMRG
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2_complex

TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[400] * 4 + [800] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-14] * 8,
    n_sweeps=16,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


@requires_block2_complex
@pytest.mark.slow
def test_reldmrg_transition_dipoles_vs_relci(tmp_path):
    """RelDMRG transition dipoles/oscillator strengths == 2C-RelCI.

    HF/cc-pVDZ with spin-orbit X2C, CAS(10,8) (same system/active space as
    test_reldmrg_casscf_hf_small_cas). Among the 4 lowest 2C roots, 0-2 form a
    (near-)degenerate spin-orbit manifold and root 3 is well separated in
    energy, giving a genuinely bright, gauge-unambiguous (0, 3) transition
    (same non-degeneracy rationale as the (0, 6) pair in
    test_dmrg_transition_dipoles_vs_fci).
    """
    system = System(
        xyz="H 0.0 0.0 0.0\nF 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )

    scf = GHF(charge=0)(system)
    relci = RelCI(
        nel=10,
        core_orbitals=4,
        active_orbitals=8,
        nroots=4,
        do_transition_dipole=True,
    )(scf)
    relci.run()

    scf2 = GHF(charge=0)(system)
    reldmrg = RelDMRG(
        nel=10,
        core_orbitals=4,
        active_orbitals=8,
        nroots=4,
        dmrg_params=TIGHT(str(tmp_path)),
        do_transition_dipole=True,
    )(scf2)
    reldmrg.run()

    assert reldmrg.vertical_transition_energies[(0, 3)] == approx(
        relci.vertical_transition_energies[(0, 3)].real
    )
    assert reldmrg.oscillator_strengths[(0, 3)] == approx(
        relci.oscillator_strengths[(0, 3)].real
    )

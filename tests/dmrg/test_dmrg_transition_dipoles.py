"""
Transition-dipole / oscillator-strength interfacing for the DMRG solver.

DMRGSolver/RelDMRGSolver previously had no ``compute_transition_properties``
at all (unlike CISolver/RelCISolver): the transition-dipole machinery in
``CISolver.compute_transition_properties`` (forte2/ci/ci.py) is
representation-agnostic -- it only calls ``make_1rdm(left_root, right_root)``
for cross-root (transition) 1-RDMs, which ``DMRGSolver``/``RelDMRGSolver``
already provide via block2's ``get_npdm(..., bra=...)``. This test exercises
that binding end-to-end (via the ``DMRG``/``do_transition_dipole`` convenience
class, mirroring ``CI``/``RelCI``) and checks the resulting oscillator
strengths/vertical transition energies against the exact CI reference
(computed in-test), reusing the same system as test_ci_tdm in
tests/ci/test_ci_rhf.py.
"""

import pytest

from forte2 import System, RHF, State, CI
from forte2.dmrg import DMRG
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2

TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[200] * 4 + [400] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 8,
    n_sweeps=12,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


@requires_block2
@pytest.mark.slow
def test_dmrg_transition_dipoles_vs_fci(tmp_path):
    """DMRG transition dipoles/oscillator strengths == FCI, N2 CAS(6,6), 10 roots.

    Same system/active space as test_ci_tdm. Among the 10 lowest CAS(6,6)
    singlets, roots 0-5 are gerade (dipole-forbidden from the ground state by
    inversion symmetry) and root 6 is the first ungerade, dipole-allowed
    state, giving a genuinely bright (0, 6) transition. Root 6 is also
    non-degenerate (well separated in energy from its neighbors), so its
    transition dipole from the ground state is gauge-unambiguous -- unlike the
    degenerate pairs elsewhere in this manifold (e.g. roots 4/5, 8/9), whose
    individual transition-dipole components depend on the arbitrary basis
    chosen inside the degenerate subspace (see the RDM docstrings in
    forte2/dmrg/dmrg.py), even though the oscillator strength/VTE do not.
    """
    xyz = """
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """
    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    state = State(nel=14, multiplicity=1, ms=0.0)

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        states=state,
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=10,
        do_transition_dipole=True,
    )(rhf)
    ci.run()

    rhf2 = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=state,
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=10,
        dmrg_params=TIGHT(str(tmp_path)),
        do_transition_dipole=True,
    )(rhf2)
    dmrg.run()

    assert dmrg.vertical_transition_energies[(0, 6)] == approx(
        ci.vertical_transition_energies[(0, 6)]
    )
    assert dmrg.oscillator_strengths[(0, 6)] == pytest.approx(
        ci.oscillator_strengths[(0, 6)], abs=1e-4
    )

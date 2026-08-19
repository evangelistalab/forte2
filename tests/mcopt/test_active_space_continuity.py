"""
Regression coverage for MCOptimizerBase._check_active_space_continuity.

Investigation context: a finite-difference nuclear gradient of a CASSCF(2,2)
energy over water/STO-3G, seeding each displaced geometry's SCF guess by
projecting the reference geometry's converged orbitals (as
GeometryOptimizer/FiniteDifferenceGradient do), showed one displacement
0.002 Bohr from the reference converging to an energy ~0.02 Eh lower than the
smooth trend set by its neighbors. This is NOT a multi-threading race: the
same wrong energy reproduces bit-for-bit at 1, 2, 4, 8, 16, and 32 threads
(FORTE_NUM_THREADS_OVERRIDE), and is insensitive to max_rotation (0.2 down to
0.01 all converge to the same value, just in more macroiterations). Tracing
the macroiteration table shows the CASSCF energy tracking the correct branch
smoothly for several macroiterations, then dropping discontinuously by
~0.0185 Eh in a single macroiteration (right after several consecutive
"L-BFGS Warning: Skip this vector due to negative rho" messages) and
reconverging from there -- i.e. gradient-based orbital optimization, seeded
close to a separatrix between two competing CASSCF solutions, genuinely
walked into the other one.

_check_active_space_continuity does not prevent this (a real fix would need
a trust-region-style step rejection or a maximum-overlap/orbital-following
scheme, and the wrong-seed geometry itself is arguably a legitimate CASSCF
solution, just not the one that continues the reference smoothly). It
instead detects it: the active-space subspace overlap
|det(C_old[:,active]^T S C_new[:,active])| between consecutive macroiterations
drops from ~1 for a normal rotation step to 0.139 for this jump, which is
what the tests below check for -- both that it fires on the reproducer and
that it does not false-positive on ordinary CASSCF convergence.
"""

import numpy as np
import pytest
import scipy as sp

import forte2.integrals as integrals
from forte2 import CISolver, MCOptimizer, RHF, State, System
from forte2.helpers.comparisons import approx


def _water(coords):
    symbols = ["O", "H", "H"]
    lines = [f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for s, c in zip(symbols, coords)]
    return System(
        xyz="\n".join(lines),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )


def _project(old_system, new_system, C_old, nocc):
    """Project occupied orbitals into a displaced AO basis (mirrors
    forte2.base_classes.rebuild.project_occupied_orbitals, inlined here since
    this branch predates that module)."""
    X_new = new_system.get_Xorth()
    S_cross = integrals.overlap(new_system, new_system.basis, old_system.basis)
    Q_occ_raw = X_new.T.conj() @ S_cross @ C_old[:, :nocc]
    Q_occ, _ = np.linalg.qr(Q_occ_raw, mode="reduced")
    Q_occ = Q_occ[:, :nocc]
    nvirt = X_new.shape[1] - nocc
    Q_virt = sp.linalg.null_space(Q_occ.T.conj())
    Q = np.hstack((Q_occ, Q_virt[:, :nvirt]))
    return X_new @ Q


_BASE_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.9], [1.6, 0.3, 0.0]])
_STEP = 1.0e-3


def _casscf_seeded_from_reference(offset):
    refsys = _water(_BASE_COORDS)
    ref_rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(refsys)
    ref_rhf.run()

    coords = _BASE_COORDS.copy()
    coords[1, 1] += offset * _STEP
    system = _water(coords)
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    rhf.C = [_project(refsys, system, ref_rhf.C[0], ref_rhf.na)]
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5],
        core_orbitals=[0, 1, 2, 3],
    )
    mc = MCOptimizer(
        ci_solver, e_tol=1.0e-12, g_tol=1.0e-10, final_orbitals="original"
    )(rhf)
    mc.run()
    return mc


def test_active_space_continuity_flags_known_discontinuous_geometry():
    mc = _casscf_seeded_from_reference(offset=-2)

    assert mc.discontinuous_macro_iterations == [4]
    assert mc.E == pytest.approx(-74.95866747061628, abs=1.0e-6)


@pytest.mark.parametrize("offset", [-1, 1])
def test_active_space_continuity_silent_on_neighboring_geometries(offset):
    # The immediate neighbors of the discontinuous displacement above (half
    # its distance from the reference) converge smoothly with the same
    # seeding procedure: this is not a blanket property of orbital
    # projection, only of this specific, larger displacement.
    mc = _casscf_seeded_from_reference(offset=offset)

    assert mc.discontinuous_macro_iterations == []


def test_active_space_continuity_silent_without_orbital_seeding():
    # No projected seed at all (the default minao/SAP guess): CASSCF
    # converges smoothly at the same geometry that is discontinuous when
    # seeded from the reference.
    coords = _BASE_COORDS.copy()
    coords[1, 1] += -2 * _STEP
    system = _water(coords)
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5],
        core_orbitals=[0, 1, 2, 3],
    )
    mc = MCOptimizer(
        ci_solver, e_tol=1.0e-12, g_tol=1.0e-10, final_orbitals="original"
    )(rhf)
    mc.run()

    assert mc.discontinuous_macro_iterations == []


def test_active_space_continuity_silent_for_ordinary_casscf_hf():
    # Baseline regression check: a plain, unseeded CASSCF run far from any
    # known-bad case should never trip the diagnostic.
    xyz = "H 0.0 0.0 0.0\nF 0.0 0.0 2.0"
    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=6,
        core_orbitals=1,
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    assert mc.discontinuous_macro_iterations == []
    assert mc.E == approx(-100.0435018956)

import numpy as np

from forte2 import System, RHF, MCOptimizer, State, CISolver
from forte2.helpers.comparisons import approx


def test_mcscf_noncontiguous_spaces():
    # The results of this test should be strictly identical to test_casscf_n2

    erhf = -108.761639873604
    ecasscf = -108.9800484156

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    rhf.run()
    assert rhf.E == approx(erhf)

    # swap orbitals to make them non-contiguous
    core = [0, 1, 3, 6]
    actv = [2, 4, 5, 7, 8, 11]
    virt = sorted(set(range(system.nbf)) - set(core + actv))
    rhf.C[0][:, core + actv + virt] = rhf.C[0]

    ci_solver = CISolver(
        State(nel=14, multiplicity=1, ms=0.0), active_orbitals=actv, core_orbitals=core
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()
    assert mc.E == approx(ecasscf)


def test_mcscf_noncontiguous_spaces_with_symmetry():
    # Regression: the irrep symmetry mask on nonredundant orbital rotations was
    # built from irrep_indices in the *original* MO ordering while the rotation
    # matrix nrr is built in *contiguous* ordering. For a non-contiguous active
    # space under non-C1 symmetry this froze/enabled the wrong (i, j) rotations
    # and gave a wrong CASSCF energy. Selecting the same physical orbitals as a
    # contiguous vs non-contiguous space must yield the identical energy.
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    # Reference: contiguous active space with symmetry enabled.
    system_c = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )
    rhf_c = RHF(charge=0, e_tol=1e-12)(system_c)
    rhf_c.run()
    core_c = [0, 1, 2, 3]
    actv_c = [4, 5, 6, 7, 8, 9]
    e_contig = (
        MCOptimizer(
            CISolver(
                State(nel=14, multiplicity=1, ms=0.0),
                active_orbitals=actv_c,
                core_orbitals=core_c,
            )
        )(rhf_c)
        .run()
        .E
    )

    # Same physical orbitals, relabeled into a non-contiguous space.
    system_nc = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )
    rhf_nc = RHF(charge=0, e_tol=1e-12)(system_nc)
    rhf_nc.run()
    core_nc = [0, 1, 3, 6]
    actv_nc = [2, 4, 5, 7, 8, 11]
    allmo = list(range(system_nc.nbf))
    virt_nc = sorted(set(allmo) - set(core_nc + actv_nc))
    old_order = core_c + actv_c + sorted(set(allmo) - set(core_c + actv_c))
    new_order = core_nc + actv_nc + virt_nc

    C_new = rhf_nc.C[0].copy()
    C_new[:, new_order] = rhf_nc.C[0][:, old_order]
    rhf_nc.C[0] = C_new
    irr = np.array(rhf_nc.irrep_indices[0])
    irr_new = irr.copy()
    irr_new[new_order] = irr[old_order]
    rhf_nc.irrep_indices[0] = list(irr_new)

    e_noncontig = (
        MCOptimizer(
            CISolver(
                State(nel=14, multiplicity=1, ms=0.0),
                active_orbitals=actv_nc,
                core_orbitals=core_nc,
            )
        )(rhf_nc)
        .run()
        .E
    )

    assert e_noncontig == approx(e_contig)

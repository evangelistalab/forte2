import pytest
from forte2 import System, RHF, MCOptimizer, State, DSRG_MRPT2
from forte2.scf.scf_utils import repair_symmetry
from forte2.helpers.comparisons import approx


def test_dsrg_mrpt2_with_sym_1():
    xyz = """
    C 0.0 0.0 0.0
    C 0.0 0.0 2.1
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )

    rhf = RHF(charge=0)(system)
    rhf = repair_symmetry(rhf)
    mc = MCOptimizer(
        states=State(nel=12, multiplicity=1, ms=0.0, symmetry=0),
        nroots=3,
        core_orbitals=2,
        active_orbitals=8,
    )(rhf)
    mc.run()
    pt = DSRG_MRPT2(relax_reference="once")(mc)
    pt.run()
    assert pt.relax_eigvals == approx([-75.5589037326, -75.5546573565, -75.5331537682])


def test_dsrg_mrpt2_with_sym_2():
    # Validated against the following forte1 input:
    # import forte

    # molecule c2{
    #   0 1
    #   C
    #   C 1 1.3
    # }

    # set globals{
    #   basis                   cc-pvdz
    #   df_basis_mp2            cc-pvtz-jkfit
    #   df_basis_scf            cc-pvtz-jkfit
    #   reference               rhf
    #   scf_type                df
    # }

    # set forte{
    #   restricted_docc         [1,0,0,0,0,1,0,0]
    #   active                  [2,0,1,1,0,2,1,1]
    #   int_type                df
    #   avg_state               [[0,1,3]]
    #   correlation_solver      dsrg-mrpt2
    #   active_space_solver     genci
    #   mcscf_reference         true
    #   calc_type               sa
    #   RELAX_REF               twice
    # }

    # energy('forte')

    xyz = """
    C 0.0 0.0 0.0
    C 0.0 0.0 1.3
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )

    rhf = RHF(charge=0)(system)
    rhf = repair_symmetry(rhf)
    mc = MCOptimizer(
        states=State(nel=12, multiplicity=1, ms=0.0, symmetry=0),
        nroots=3,
        core_orbitals=2,
        active_orbitals=8,
    )(rhf)
    mc.run()
    pt = DSRG_MRPT2(relax_reference="twice")(mc)
    pt.run()
    assert pt.relax_eigvals == approx([-75.7202896794, -75.6479318059, -75.6375093503])

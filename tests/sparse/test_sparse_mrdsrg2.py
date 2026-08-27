import math

import numpy as np
import pytest

import forte2
from forte2.dsrg.sparse_mrdsrg2 import _gno_commutator
from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def det(s):
    return Determinant(s)


def correlated_singlet_vacuum(weight0=0.7):
    return sparse_ops.SparseState(
        {
            det("20"): math.sqrt(weight0),
            det("02"): math.sqrt(1.0 - weight0),
        }
    )


def one_alpha_active_vacuum():
    return sparse_ops.SparseState(
        {
            det("a00"): 1.0 / math.sqrt(2.0),
            det("0a0"): 1.0 / math.sqrt(2.0),
        }
    )


def expectation(op, state):
    return sparse_ops.overlap(state, op.apply_to_state(state)).real


def test_sparse_mrdsrg2_returns_reference_energy_without_external_excitations():
    vacuum = correlated_singlet_vacuum(weight0=0.7)
    ham = sparse_ops.sparse_operator(
        [
            ("[]", 1.0),
            ("[0a+ 0a-]", 2.0),
            ("[0b+ 0b-]", 3.0),
            ("[0a+ 0b+ 0b- 0a-]", 0.4),
        ]
    )

    result = forte2.solve_sparse_mrdsrg2(ham, vacuum, 2, [])

    assert result.converged
    assert result.iterations == 0
    assert result.energy == pytest.approx(expectation(ham, vacuum))
    assert result.energy == pytest.approx(4.78)


def test_sparse_mrdsrg_diagonalizes_model_space_effective_hamiltonian():
    vacuum = correlated_singlet_vacuum(weight0=0.7)
    ham = sparse_ops.sparse_operator(
        [
            ("[0a+ 0a-]", -0.2),
            ("[0b+ 0b-]", -0.2),
            ("[1a+ 1a-]", 0.3),
            ("[1b+ 1b-]", 0.3),
            ("[0a+ 0b+ 1b- 1a-]", 0.05),
            ("[1a+ 1b+ 0b- 0a-]", 0.05),
        ]
    )
    model_space = [det("20"), det("02")]

    result = forte2.solve_sparse_mrdsrg2(
        ham,
        vacuum,
        2,
        [],
        model_space=model_space,
    )

    hmat = np.array(
        [
            [
                sparse_ops.overlap(
                    sparse_ops.SparseState({bra: 1.0}),
                    ham.apply_to_state(sparse_ops.SparseState({ket: 1.0})),
                )
                for ket in model_space
            ]
            for bra in model_space
        ],
        dtype=complex,
    )
    expected = min(np.linalg.eigvalsh(hmat).real)

    assert result.converged
    assert result.scalar_energy == pytest.approx(expectation(ham, vacuum).real)
    assert result.energy == pytest.approx(expected)
    assert result.model_space_energies[0] == pytest.approx(expected)


def test_enumerate_mrdsrg_excitations_excludes_pure_active_operators():
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[0],
        active_orbitals=[1],
        virtual_orbitals=[2],
        orbital_energies=[-1.0, 0.0, 1.0],
        max_rank=1,
    )
    labels = {exc.label for exc in excitations}

    assert "[1a+ 0a-]" in labels
    assert "[2a+ 0a-]" in labels
    assert "[2a+ 1a-]" in labels
    assert "[1a+ 1a-]" not in labels
    assert "[1b+ 1b-]" not in labels


def test_sparse_mrdsrg2_uses_normal_ordered_hamiltonian_denominators_by_default():
    vacuum = one_alpha_active_vacuum()
    ham = sparse_ops.sparse_operator(
        [
            ("[0a+ 0a-]", 0.2),
            ("[0b+ 0b-]", 0.2),
            ("[1a+ 1a-]", 0.4),
            ("[1b+ 1b-]", 0.4),
            ("[2a+ 2a-]", 1.0),
            ("[2b+ 2b-]", 1.0),
        ]
    )
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[],
        active_orbitals=[0, 1],
        virtual_orbitals=[2],
        orbital_energies=[0.0, 0.0, 0.0],
        max_rank=1,
    )

    result = forte2.solve_sparse_mrdsrg2(ham, vacuum, 3, excitations, maxiter=1)
    denominators = {
        excitation.label: excitation.denominator for excitation in result.excitations
    }

    assert denominators["[2a+ 0a-]"] == pytest.approx(-0.8)
    assert denominators["[2a+ 1a-]"] == pytest.approx(-0.6)


def test_sparse_mrdsrg2_iterates_tiny_multireference_model():
    vacuum = one_alpha_active_vacuum()
    ham = sparse_ops.sparse_operator(
        [
            ("[1a+ 1a-]", 0.2),
            ("[2a+ 2a-]", 1.0),
            ("[2a+ 0a-]", 0.001),
            ("[0a+ 2a-]", 0.001),
            ("[2a+ 1a-]", 0.001),
            ("[1a+ 2a-]", 0.001),
        ]
    )
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[],
        active_orbitals=[0, 1],
        virtual_orbitals=[2],
        orbital_energies=[0.0, 0.2, 1.0],
        max_rank=1,
    )

    result = forte2.solve_sparse_mrdsrg2(
        ham,
        vacuum,
        3,
        excitations,
        flow_param=0.1,
        maxiter=80,
        e_tol=1.0e-10,
        r_tol=1.0e-7,
        max_commutators=8,
        do_diis=False,
    )

    assert result.converged
    assert result.iterations > 1
    assert result.energy < expectation(ham, vacuum)
    assert result.history[-1].rms_update < 1.0e-7


def test_sparse_mrdsrg2_cumulant_backend_matches_and_validates_sparse_engine():
    vacuum = one_alpha_active_vacuum()
    ham = sparse_ops.sparse_operator(
        [
            ("[1a+ 1a-]", 0.2),
            ("[2a+ 2a-]", 1.0),
            ("[2a+ 0a-]", 0.001),
            ("[0a+ 2a-]", 0.001),
            ("[2a+ 1a-]", 0.001),
            ("[1a+ 2a-]", 0.001),
        ]
    )
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[],
        active_orbitals=[0, 1],
        virtual_orbitals=[2],
        orbital_energies=[0.0, 0.2, 1.0],
        max_rank=1,
    )
    options = dict(
        flow_param=0.1,
        maxiter=3,
        max_commutators=4,
        do_diis=False,
    )

    sparse = forte2.solve_sparse_mrdsrg2(
        ham, vacuum, 3, excitations, gno_backend="sparse", **options
    )
    cumulant = forte2.solve_sparse_mrdsrg2(
        ham, vacuum, 3, excitations, gno_backend="cumulant", **options
    )
    validated = forte2.solve_sparse_mrdsrg2(
        ham, vacuum, 3, excitations, gno_backend="validate", **options
    )

    assert sparse.gno_backend == "sparse"
    assert cumulant.gno_backend == "cumulant"
    assert validated.gno_backend == "validate"
    assert cumulant.energy == pytest.approx(sparse.energy, abs=1.0e-11)
    assert validated.energy == pytest.approx(sparse.energy, abs=1.0e-11)
    assert [entry.energy for entry in cumulant.history] == pytest.approx(
        [entry.energy for entry in sparse.history], abs=1.0e-11
    )


def test_sparse_mrdsrg_rejects_unsupported_cumulant_backend_options():
    vacuum = correlated_singlet_vacuum()
    ham = sparse_ops.sparse_operator("[]", 1.0)

    with pytest.raises(ValueError, match="gno_backend"):
        forte2.SparseMRDSRG(ham, vacuum, 2, [], gno_backend="unknown")
    with pytest.raises(ValueError, match="max_rank"):
        forte2.SparseMRDSRG(ham, vacuum, 2, [], max_rank=5, gno_backend="cumulant")
    with pytest.raises(ValueError, match="max_cumulant"):
        forte2.SparseMRDSRG(
            ham,
            vacuum,
            2,
            [],
            max_cumulant=5,
            gno_backend="cumulant",
        )
    with pytest.raises(ValueError, match="include_four_body_cumulant"):
        forte2.SparseMRDSRG(
            ham,
            vacuum,
            2,
            [],
            max_cumulant=2,
            include_four_body_cumulant=True,
        )


def test_sparse_mrdsrg_validation_supports_rank_three_cumulant_truncation():
    vacuum = correlated_singlet_vacuum()
    lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        3,
        sparse_ops.sqop("[0a+ 1a+ 0b+ 1b- 0b- 0a-]")[0],
        1.0,
    )
    rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        3,
        sparse_ops.sqop("[1a+ 0b+ 0b- 0a-]")[0],
        1.0,
    )
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)
    sparse_engine = sparse_ops.GeneralizedNormalOrderedProductComputer(
        reference, 3, 1.0e-14
    )
    cumulant_engine = sparse_ops.CumulantWickEngine(reference, 3, 1.0e-14)

    validated = _gno_commutator(
        lhs,
        rhs,
        vacuum,
        2,
        3,
        3,
        1.0e-14,
        gno_backend="validate",
        sparse_product_engine=sparse_engine,
        cumulant_engine=cumulant_engine,
    )
    direct = cumulant_engine.commutator(lhs, rhs)
    assert {term.str(): value for term, value in validated} == pytest.approx(
        {term.str(): value for term, value in direct}, abs=1.0e-12
    )


def test_sparse_mrdsrg_four_body_cumulants_are_opt_in():
    vacuum = sparse_ops.SparseState({det("2200"): 1.0})
    ham = sparse_ops.sparse_operator(
        [
            ("[0a+ 0a-]", 0.0),
            ("[0b+ 0b-]", 0.0),
            ("[1a+ 1a-]", 0.1),
            ("[1b+ 1b-]", 0.1),
            ("[2a+ 2a-]", 1.0),
            ("[2b+ 2b-]", 1.0),
            ("[3a+ 3a-]", 1.2),
            ("[3b+ 3b-]", 1.2),
        ]
    )
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[0, 1],
        active_orbitals=[],
        virtual_orbitals=[2, 3],
        orbital_energies=[0.0, 0.1, 1.0, 1.2],
        max_rank=4,
    )

    assert any(excitation.rank == 3 for excitation in excitations)
    assert any(excitation.rank == 4 for excitation in excitations)

    result3 = forte2.solve_sparse_mrdsrg3(
        ham,
        vacuum,
        4,
        excitations,
        maxiter=3,
        max_commutators=3,
    )
    result4_default = forte2.solve_sparse_mrdsrg4(
        ham,
        vacuum,
        4,
        excitations,
        maxiter=3,
        max_commutators=3,
    )
    result4 = forte2.solve_sparse_mrdsrg4(
        ham,
        vacuum,
        4,
        excitations,
        include_four_body_cumulant=True,
        gno_backend="cumulant",
        maxiter=3,
        max_commutators=3,
    )

    assert result3.converged
    assert result3.max_rank == 3
    assert result3.max_cumulant == 3
    assert not result3.include_four_body_cumulant
    assert all(excitation.rank <= 3 for excitation in result3.excitations)
    assert result3.hbar.max_cumulant() == 3
    assert result3.energy == pytest.approx(expectation(ham, vacuum))

    assert result4_default.max_rank == 4
    assert result4_default.max_cumulant == 3
    assert not result4_default.include_four_body_cumulant
    assert result4_default.hbar.max_cumulant() == 3

    assert result4.converged
    assert result4.max_rank == 4
    assert result4.max_cumulant == 4
    assert result4.include_four_body_cumulant
    assert result4.gno_backend == "cumulant"
    assert result4.hbar.max_cumulant() == 4
    assert result4.energy == pytest.approx(expectation(ham, vacuum))


def test_sparse_mrdsrg3_backends_produce_the_same_iterative_energy():
    vacuum = sparse_ops.SparseState(
        {
            det("2000"): math.sqrt(0.7),
            det("0200"): math.sqrt(0.3),
        }
    )
    ham = sparse_ops.sparse_operator(
        [
            ("[]", 0.2),
            ("[0a+ 0a-]", -0.4),
            ("[0b+ 0b-]", -0.4),
            ("[1a+ 1a-]", -0.1),
            ("[1b+ 1b-]", -0.1),
            ("[2a+ 2a-]", 0.8),
            ("[2b+ 2b-]", 0.8),
            ("[3a+ 3a-]", 1.1),
            ("[3b+ 3b-]", 1.1),
            ("[2a+ 0a-]", 0.03),
            ("[0a+ 2a-]", 0.03),
            ("[2b+ 0b-]", 0.03),
            ("[0b+ 2b-]", 0.03),
            ("[2a+ 2b+ 0b- 0a-]", 0.05),
            ("[0a+ 0b+ 2b- 2a-]", 0.05),
            ("[3a+ 2b+ 1b- 0a-]", 0.02),
            ("[0a+ 1b+ 2b- 3a-]", 0.02),
        ]
    )
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[],
        active_orbitals=[0, 1],
        virtual_orbitals=[2, 3],
        orbital_energies=[-0.4, -0.1, 0.8, 1.1],
        max_rank=3,
    )
    options = dict(
        flow_param=0.5,
        max_cumulant=3,
        max_commutators=3,
        maxiter=3,
        e_tol=0.0,
        r_tol=0.0,
        screen_thresh=1.0e-14,
        commutator_threshold=0.0,
        do_diis=False,
    )

    sparse = forte2.solve_sparse_mrdsrg3(
        ham, vacuum, 4, excitations, gno_backend="sparse", **options
    )
    cumulant = forte2.solve_sparse_mrdsrg3(
        ham, vacuum, 4, excitations, gno_backend="cumulant", **options
    )

    assert [item.energy for item in sparse.history] == pytest.approx(
        [item.energy for item in cumulant.history], abs=2.0e-13
    )
    assert sparse.energy == pytest.approx(cumulant.energy, abs=2.0e-13)
    assert (
        max(
            abs(amplitude)
            for amplitude, excitation in zip(sparse.amplitudes, sparse.excitations)
            if excitation.rank == 3
        )
        > 1.0e-7
    )

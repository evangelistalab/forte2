import itertools
import math

import numpy as np
import pytest

from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def alpha_bits(orbitals):
    result = Determinant.zero()
    for orbital in orbitals:
        result.set_na(orbital, True)
    return result


def permutation_phase(permutation):
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def dense_antisymmetric_tensor(values, rank, dimension):
    tensor = np.zeros((dimension,) * (2 * rank))
    permutations = tuple(itertools.permutations(range(rank)))
    for (upper, lower), value in values.items():
        for lower_perm in permutations:
            for upper_perm in permutations:
                indices = tuple(lower[i] for i in lower_perm) + tuple(
                    upper[i] for i in upper_perm
                )
                tensor[indices] = (
                    permutation_phase(lower_perm)
                    * permutation_phase(upper_perm)
                    * value
                )
    return tensor


def antisymmetrize_output(tensor, rank):
    if rank == 0:
        return tensor
    result = np.zeros_like(tensor)
    for lower_perm in itertools.permutations(range(rank)):
        for upper_perm in itertools.permutations(range(rank)):
            axes = tuple(lower_perm) + tuple(rank + i for i in upper_perm)
            result += (
                permutation_phase(lower_perm)
                * permutation_phase(upper_perm)
                * tensor.transpose(axes)
            )
    return result


def gno_tensor(operator, rank, dimension):
    values = {}
    for term, coefficient in operator:
        if term.cre().count() != rank:
            continue
        upper = tuple(i for i in range(dimension) if term.cre().na(i))
        lower = tuple(i for i in range(dimension) if term.ann().na(i))
        values[(upper, lower)] = coefficient.real
    if rank == 0:
        return np.asarray(values.get(((), ()), 0.0))
    return dense_antisymmetric_tensor(values, rank, dimension)


def test_rank_three_commutator_matches_wickd_cumulant_truncation():
    wickd = pytest.importorskip("wickd")
    numerical = pytest.importorskip("wickd.spinadapt.numerical")

    dimension = 6
    vacuum = sparse_ops.SparseState(
        {
            Determinant("aaa000"): math.sqrt(0.5),
            Determinant("000aaa"): math.sqrt(0.3),
            Determinant("a0a0a0"): math.sqrt(0.2),
        }
    )
    reference = sparse_ops.CumulantReference(
        vacuum, dimension, max_cumulant=3, screen_thresh=0.0
    )

    combinations = tuple(itertools.combinations(range(dimension), 3))
    index_pairs = tuple(itertools.product(combinations, repeat=2))
    rng = np.random.default_rng(8)

    def random_operator():
        values = {
            index_pairs[index]: rng.normal()
            for index in rng.choice(len(index_pairs), 12, replace=False)
        }
        operator = sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, dimension, 3
        )
        for (upper, lower), value in values.items():
            operator.add(
                sparse_ops.SQOperatorString(alpha_bits(upper), alpha_bits(lower)),
                value,
            )
        return operator, dense_antisymmetric_tensor(values, 3, dimension)

    lhs, lhs_tensor = random_operator()
    rhs, rhs_tensor = random_operator()
    gamma = np.array(
        [
            [reference.gamma(p, True, q, True).real for q in range(dimension)]
            for p in range(dimension)
        ]
    )

    def cumulant_tensor(rank):
        combinations = itertools.combinations(range(dimension), rank)
        values = {
            (upper, lower): reference.cumulant(
                alpha_bits(upper), alpha_bits(lower)
            ).real
            for upper in combinations
            for lower in itertools.combinations(range(dimension), rank)
        }
        return dense_antisymmetric_tensor(values, rank, dimension)

    wickd.reset_space()
    wickd.add_space("a", "fermion", "general", list("pqrstuvwxyzabcdefghijklmno"))
    lhs_symbol = wickd.utils.gen_op("O", 3, "a", "a")
    rhs_symbol = wickd.utils.gen_op("T", 3, "a", "a")
    theorem = wickd.WickTheorem()
    theorem.set_max_cumulant(3)
    theorem.set_single_threaded(True)
    equations = (
        theorem.contract(
            wickd.rational(1), wickd.commutator(lhs_symbol, rhs_symbol), 0, 6
        )
        .canonicalize()
        .to_manybody_equation("C")
    )
    arrays = {
        "O": lhs_tensor,
        "T": rhs_tensor,
        "gamma1": gamma,
        "eta1": np.eye(dimension) - gamma,
        "lambda2": cumulant_tensor(2),
        "lambda3": cumulant_tensor(3),
    }

    direct = sparse_ops.CumulantWickEngine(reference, 3, 0.0).commutator(lhs, rhs)
    alternate = sparse_ops.GeneralizedNormalOrderedProductComputer(
        reference, 3, 0.0
    ).commutator(lhs, rhs)
    for block in equations:
        rank = len(block.split("|")[0])
        expected = antisymmetrize_output(
            numerical.evaluate_equations(equations[block], arrays), rank
        )
        np.testing.assert_allclose(
            gno_tensor(direct, rank, dimension), expected, atol=2.0e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            gno_tensor(alternate, rank, dimension), expected, atol=2.0e-12, rtol=0.0
        )

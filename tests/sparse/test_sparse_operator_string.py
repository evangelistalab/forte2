import random

import forte2


def _det(alpha=(), beta=()):
    d = forte2.Determinant.zero()
    for i in alpha:
        d.set_na(i, True)
    for i in beta:
        d.set_nb(i, True)
    return d


def _reference_sign_mask(acre=(), bcre=(), aann=(), bann=()):
    norb = forte2.Determinant.maxnorb
    ops = [*acre, *aann, *(norb + i for i in bcre), *(norb + i for i in bann)]
    alpha = [i for i in range(norb) if sum(op > i for op in ops) % 2 == 1]
    beta = [i for i in range(norb) if sum(op > norb + i for op in ops) % 2 == 1]
    return _det(alpha, beta)


def _assert_sign_mask_fast_consistent(acre=(), bcre=(), aann=(), bann=()):
    cre = _det(acre, bcre)
    ann = _det(aann, bann)
    expected = _reference_sign_mask(acre, bcre, aann, bann)
    sqop = forte2.SQOperatorString(cre, ann)
    assert forte2.compute_sign_mask_fast(cre, ann) == expected
    assert sqop.sign_mask() == expected


def test_compute_sign_mask_fast_matches_reference():
    cases = [
        ((), (), (), ()),
        ((0,), (), (), ()),
        ((63,), (), (), ()),
        ((), (0,), (), ()),
        ((), (63,), (), ()),
        ((0, 2, 63), (), (1, 2, 62), ()),
        ((), (0, 2, 63), (), (1, 2, 62)),
        ((0, 63), (0, 63), (0, 63), (0, 63)),
        ((0, 5, 63), (1, 9, 63), (5, 8), (0, 9)),
    ]
    for acre, bcre, aann, bann in cases:
        _assert_sign_mask_fast_consistent(acre, bcre, aann, bann)

    rng = random.Random(7)
    norb = forte2.Determinant.maxnorb
    for _ in range(100):
        acre = rng.sample(range(norb), rng.randrange(9))
        bcre = rng.sample(range(norb), rng.randrange(9))
        aann = rng.sample(range(norb), rng.randrange(9))
        bann = rng.sample(range(norb), rng.randrange(9))
        _assert_sign_mask_fast_consistent(acre, bcre, aann, bann)


def test_sparse_operator_string_count():
    sop, _ = forte2.sqop("[1a+ 3a+ 3a- 2a-]")
    assert sop.count() == 4


def test_sparse_operator_string_is_number():
    sop, _ = forte2.sqop("[1a+ 3a+ 3a- 2a-]")
    assert sop.is_identity() is False

    sop, _ = forte2.sqop("[1a+]")
    assert sop.is_identity() is False

    sop, _ = forte2.sqop("[1a+ 1a-]")
    assert sop.is_identity() is False

    sop, _ = forte2.sqop("[]")
    assert sop.is_identity() is True


def test_sparse_operator_string_is_nilpotent():
    sop, _ = forte2.sqop("[1a+ 3a+ 3a- 2a-]")
    assert sop.is_nilpotent() is True

    sop, _ = forte2.sqop("[1a+]")
    assert sop.is_nilpotent() is True

    # number operators and the identity operator are not nilpotent
    sop, _ = forte2.sqop("[1a+ 1a-]")
    assert sop.is_nilpotent() is False

    sop, _ = forte2.sqop("[]")
    assert sop.is_nilpotent() is False


def test_sparse_operator_string_commutator_type():
    # test commuting terms
    sop1, _ = forte2.sqop("[1a+ 0a-]")
    sop2, _ = forte2.sqop("[3a+ 2a-]")
    assert forte2.commutator_type(sop1, sop2) == forte2.CommutatorType.commute

    sop1, _ = forte2.sqop("[1a+]")
    sop2, _ = forte2.sqop("[3a+]")
    assert forte2.commutator_type(sop1, sop2) == forte2.CommutatorType.anticommute

    sop1, _ = forte2.sqop("[1a+]")
    sop2, _ = forte2.sqop("[3a-]")
    assert forte2.commutator_type(sop1, sop2) == forte2.CommutatorType.anticommute

    sop1, _ = forte2.sqop("[1a+ 3a-]")
    sop2, _ = forte2.sqop("[3a-]")
    assert forte2.commutator_type(sop1, sop2) == forte2.CommutatorType.may_not_commute


def test_sparse_operator_string_components():
    # test the number and non-number components functions
    sop, _ = forte2.sqop("[1a+ 3a+ 3a- 2a-]")
    sop_n = sop.number_component()
    assert sop_n == forte2.sqop("[3a+ 3a-]")[0]
    sop_nn = sop.non_number_component()
    assert sop_nn == forte2.sqop("[1a+ 2a-]")[0]


def test_sparse_operator_string_spin_flip():
    # test the spin flip function
    sop, _ = forte2.sqop("[1a+ 3a+ 3a- 2a-]")
    sop_flip = sop.spin_flip()
    assert sop_flip == forte2.sqop("[1b+ 3b+ 3b- 2b-]")[0]

    sop, _ = forte2.sqop("[1a+ 1b+ 2b- 1a-]")
    sop_flip = sop.spin_flip()
    assert sop_flip == forte2.sqop("[1a+ 1b+ 1b- 2a-]")[0]

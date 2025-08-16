import forte2


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

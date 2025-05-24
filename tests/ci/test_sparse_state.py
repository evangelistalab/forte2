#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte2


def det(s: str) -> forte2.Determinant:
    return forte2.Determinant(s)


def test_sparse_vector():
    import pytest
    import math

    ### Overlap tests ###
    ref = forte2.SparseState(
        {det(""): 1.0, det("+"): 1.0, det("-"): 1.0, det("2"): 1.0, det("02"): 1.0}
    )
    ref2 = forte2.SparseState({det("02"): 0.3})
    ref3 = forte2.SparseState({det("002"): 0.5})
    assert forte2.overlap(ref, ref) == pytest.approx(5.0, abs=1e-9)
    assert forte2.overlap(ref, ref2) == pytest.approx(0.3, abs=1e-9)
    assert forte2.overlap(ref2, ref) == pytest.approx(0.3, abs=1e-9)
    assert forte2.overlap(ref, ref3) == pytest.approx(0.0, abs=1e-9)

    ref_str = ref.str(2)

    ### Number projection tests ###
    proj1 = forte2.SparseState({det("2"): 1.0, det("02"): 1.0})
    test_proj1 = forte2.apply_number_projector(1, 1, ref)
    assert proj1 == test_proj1

    proj2 = forte2.SparseState({det(""): 1.0})
    test_proj2 = forte2.apply_number_projector(0, 0, ref)
    assert proj2 == test_proj2

    proj3 = forte2.SparseState({det("+"): 1.0})
    test_proj3 = forte2.apply_number_projector(1, 0, ref)
    assert proj3 == test_proj3

    proj4 = forte2.SparseState({det("-"): 1.0})
    test_proj4 = forte2.apply_number_projector(0, 1, ref)
    assert proj4 == test_proj4

    ref4 = forte2.SparseState({det("+"): 1, det("-"): 1})
    ref4 = forte2.normalize(ref4)
    assert ref4.norm() == pytest.approx(1.0, abs=1e-9)
    assert forte2.spin2(ref4, ref4) == pytest.approx(0.75, abs=1e-9)

    ref5 = forte2.SparseState({det("2"): 1})
    assert forte2.spin2(ref5, ref5) == pytest.approx(0, abs=1e-9)


if __name__ == "__main__":
    test_sparse_vector()

import pytest

import forte2


def det(s: str) -> forte2.Determinant:
    return forte2.Determinant(s)


def test_sparse_vector_overlap():
    ref = forte2.SparseState(
        {det(""): 1.0, det("a"): 1.0, det("b"): 1.0, det("2"): 1.0, det("02"): 1.0}
    )
    ref2 = forte2.SparseState({det("02"): 0.3})
    ref3 = forte2.SparseState({det("002"): 0.5})
    assert forte2.overlap(ref, ref) == pytest.approx(5.0, abs=1e-9)
    assert forte2.overlap(ref, ref2) == pytest.approx(0.3, abs=1e-9)
    assert forte2.overlap(ref2, ref) == pytest.approx(0.3, abs=1e-9)
    assert forte2.overlap(ref, ref3) == pytest.approx(0.0, abs=1e-9)


def test_sparse_vector_number_projector():
    ref = forte2.SparseState(
        {det(""): 1.0, det("a"): 1.0, det("b"): 1.0, det("2"): 1.0, det("02"): 1.0}
    )

    proj1 = forte2.SparseState({det("2"): 1.0, det("02"): 1.0})
    test_proj1 = forte2.apply_number_projector(1, 1, ref)
    assert proj1 == test_proj1

    proj2 = forte2.SparseState({det(""): 1.0})
    test_proj2 = forte2.apply_number_projector(0, 0, ref)
    assert proj2 == test_proj2

    proj3 = forte2.SparseState({det("a"): 1.0})
    test_proj3 = forte2.apply_number_projector(1, 0, ref)
    assert proj3 == test_proj3

    proj4 = forte2.SparseState({det("b"): 1.0})
    test_proj4 = forte2.apply_number_projector(0, 1, ref)
    assert proj4 == test_proj4

    ref4 = forte2.SparseState({det("a"): 1, det("b"): 1})
    ref4 = forte2.normalize(ref4)
    assert ref4.norm() == pytest.approx(1.0, abs=1e-9)
    assert forte2.spin2(ref4, ref4) == pytest.approx(0.75, abs=1e-9)

    ref5 = forte2.SparseState({det("2"): 1})
    assert forte2.spin2(ref5, ref5) == pytest.approx(0, abs=1e-9)


def test_sparse_vector_complex():
    psi1 = forte2.SparseState({det("2"): 2.0 + 1j})
    psi2 = forte2.SparseState({det("2"): 1.0 - 1j})
    assert forte2.overlap(psi1, psi2) == pytest.approx(1.0 - 3.0j, abs=1e-9)
    assert forte2.overlap(psi2, psi1) == pytest.approx(1.0 + 3.0j, abs=1e-9)

    # different lengths
    psi3 = forte2.SparseState({det("2"): 2.0 + 1j, det("ab"): 1.0 - 1j})
    psi4 = forte2.SparseState({det("2"): 1.0 - 1j})
    assert forte2.overlap(psi3, psi4) == pytest.approx(1.0 - 3.0j, abs=1e-9)
    assert forte2.overlap(psi4, psi3) == pytest.approx(1.0 + 3.0j, abs=1e-9)


def test_sparse_vector_addition():
    psi1 = forte2.SparseState({det("2"): 2.0 + 1j})
    psi2 = forte2.SparseState({det("2"): 1.0 - 0.5j, det("ab"): 0.5 + 0.5j})
    psi3 = psi1 + psi2
    assert psi3[det("2")] == pytest.approx(3.0 + 0.5j, abs=1e-9)
    assert psi3[det("ab")] == pytest.approx(0.5 + 0.5j, abs=1e-9)

    psi1 += psi2
    assert psi1[det("2")] == pytest.approx(3.0 + 0.5j, abs=1e-9)
    assert psi1[det("ab")] == pytest.approx(0.5 + 0.5j, abs=1e-9)


def test_sparse_vector_subtraction():
    psi1 = forte2.SparseState({det("2"): 2.0 + 1j})
    psi2 = forte2.SparseState({det("2"): 1.0 - 0.5j, det("ab"): 0.5 + 0.5j})
    psi3 = psi1 - psi2
    assert psi3[det("2")] == pytest.approx(1.0 + 1.5j, abs=1e-9)
    assert psi3[det("ab")] == pytest.approx(-0.5 - 0.5j, abs=1e-9)

    psi1 -= psi2
    assert psi1[det("2")] == pytest.approx(1.0 + 1.5j, abs=1e-9)
    assert psi1[det("ab")] == pytest.approx(-0.5 - 0.5j, abs=1e-9)


def test_sparse_vector_scalar_multiplication():
    psi1 = forte2.SparseState({det("2"): 2.0 + 1j, det("ab"): 1.0 + 2.0j})
    scalar = 2.0 + 3.0j

    # this calls the __rmul__ of SparseState
    psi2 = scalar * psi1
    assert psi2[det("2")] == pytest.approx((1.0 + 8.0j), abs=1e-9)
    assert psi2[det("ab")] == pytest.approx((-4.0 + 7.0j), abs=1e-9)

    # this calls the __mul__ of SparseState
    psi2 = psi1 * scalar
    assert psi2[det("2")] == pytest.approx((1.0 + 8.0j), abs=1e-9)
    assert psi2[det("ab")] == pytest.approx((-4.0 + 7.0j), abs=1e-9)

    psi1 *= scalar
    assert psi1[det("2")] == pytest.approx((1.0 + 8.0j), abs=1e-9)
    assert psi1[det("ab")] == pytest.approx((-4.0 + 7.0j), abs=1e-9)


def test_sparse_vector_norm():
    import math

    psi = forte2.SparseState({det("2"): 2.0 + 1j, det("ab"): 1.0 - 3.0j})
    # default norm is 2-norm
    assert psi.norm() == pytest.approx(math.sqrt(15.0), abs=1e-9)
    # -1 is infinity norm
    assert psi.norm(p=-1) == pytest.approx(math.sqrt(10.0), abs=1e-9)
    # 1-norm
    assert psi.norm(p=1) == pytest.approx(math.sqrt(5.0) + math.sqrt(10.0), abs=1e-9)

import forte2
import time
from forte2 import Determinant
import pytest
import numpy as np


def test_linear_apply_op():
    op = forte2.SparseOperator()
    ref = forte2.SparseState({Determinant("22"): 1.0})
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.2)
    op.add("[2a+ 0a-]", 0.2)
    op.add("[2b+ 0b-]", 0.1)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)
    op.add("[3a+ 3b+ 1b- 1a-]", -0.21)
    op.add("[1a+ 1b+ 3b- 3a-]", 0.13 * 0.17)

    wfn = forte2.apply_op(op, ref)
    assert Determinant("2200") not in wfn
    assert wfn[Determinant("+2-0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[Determinant("-2+0")] == pytest.approx(-0.3, abs=1e-9)
    assert wfn[Determinant("0220")] == pytest.approx(0.15, abs=1e-9)
    assert wfn[Determinant("2002")] == pytest.approx(-0.21, abs=1e-9)


def test_exp_apply_op():
    op = forte2.SparseOperator()
    ref = forte2.SparseState({Determinant("22"): 1.0})
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.1)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)
    op.add("[3a+ 3b+ 1b- 1a-]", -0.077)

    exp = forte2.SparseExp()
    wfn = exp.apply_op(op, ref)
    assert wfn[Determinant("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn[Determinant("0220")] == pytest.approx(0.16, abs=1e-9)
    assert wfn[Determinant("+2-0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[Determinant("-2+0")] == pytest.approx(-0.1, abs=1e-9)
    assert wfn[Determinant("2002")] == pytest.approx(-0.077, abs=1e-9)
    assert wfn[Determinant("+0-2")] == pytest.approx(-0.0077, abs=1e-9)
    assert wfn[Determinant("-0+2")] == pytest.approx(-0.0077, abs=1e-9)


def test_exp_apply_antiherm():
    op = forte2.SparseOperator()
    ref = forte2.SparseState({Determinant("22"): 1.0})
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.1)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)

    exp = forte2.SparseExp()
    wfn = exp.apply_antiherm(op, ref)
    assert wfn[Determinant("-2+0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[Determinant("+2-0")] == pytest.approx(-0.091500564912, abs=1e-9)
    assert wfn[Determinant("0220")] == pytest.approx(+0.158390400605, abs=1e-9)
    assert wfn[Determinant("2200")] == pytest.approx(+0.978860446763, abs=1e-9)


def test_exp_apply_antiherm_scale():
    op = forte2.SparseOperator()
    ref = forte2.SparseState({Determinant("22"): 1.0})
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.1)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)

    exp = forte2.SparseExp()
    wfn = exp.apply_antiherm(op, ref)
    wfn2 = exp.apply_antiherm(op, wfn, scaling_factor=-1.0)
    assert wfn2[Determinant("2200")] == pytest.approx(1.0, abs=1e-9)
    assert wfn2[Determinant("0220")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[Determinant("+2-0")] == pytest.approx(0.0, abs=1e-9)
    assert wfn2[Determinant("-2+0")] == pytest.approx(0.0, abs=1e-9)


def test_fact_exp_apply_antiherm_1():
    op = forte2.SparseOperatorList()
    op.add("[2a+ 0a-]", 0.1)
    op.add("[2b+ 0b-]", 0.2)
    op.add("[2a+ 2b+ 0b- 0a-]", 0.15)
    ref = forte2.SparseState({Determinant("22"): 1.0})

    factexp = forte2.SparseFactExp()
    wfn = factexp.apply_antiherm(op, ref)

    assert wfn[Determinant("+2-0")] == pytest.approx(-0.197676811654, abs=1e-9)
    assert wfn[Determinant("-2+0")] == pytest.approx(-0.097843395007, abs=1e-9)
    assert wfn[Determinant("0220")] == pytest.approx(+0.165338757995, abs=1e-9)
    assert wfn[Determinant("2200")] == pytest.approx(+0.961256283877, abs=1e-9)

    wfn2 = factexp.apply_antiherm(op, wfn, inverse=True)

    assert wfn2[Determinant("2200")] == pytest.approx(1.0, abs=1e-9)


def test_fact_exp_apply_antiherm_2():
    op = forte2.SparseOperatorList()
    op.add("[1a+ 0a-]", 0.1)
    op.add("[1a+ 1b+ 0b- 0a-]", -0.3)
    op.add("[1b+ 0b-]", 0.05)
    op.add("[2a+ 2b+ 1b- 1a-]", -0.07)

    ref = forte2.SparseState({Determinant("20"): 0.5, Determinant("02"): 0.8660254038})
    factexp = forte2.SparseFactExp()
    wfn = factexp.apply_antiherm(op, ref)

    assert wfn[Determinant("200")] == pytest.approx(0.733340213919, abs=1e-9)
    assert wfn[Determinant("+-0")] == pytest.approx(-0.049868863373, abs=1e-9)
    assert wfn[Determinant("002")] == pytest.approx(-0.047410073759, abs=1e-9)
    assert wfn[Determinant("020")] == pytest.approx(0.676180171388, abs=1e-9)
    assert wfn[Determinant("-+0")] == pytest.approx(0.016058887563, abs=1e-9)


def test_fact_exp_reverse():
    # this is the manually reversed operator from the previous test
    op = forte2.SparseOperatorList()
    op.add("[2a+ 2b+ 1b- 1a-]", -0.07)
    op.add("[1b+ 0b-]", 0.05)
    op.add("[1a+ 1b+ 0b- 0a-]", -0.3)
    op.add("[1a+ 0a-]", 0.1)

    ref = forte2.SparseState({Determinant("20"): 0.5, Determinant("02"): 0.8660254038})
    factexp = forte2.SparseFactExp()
    wfn = factexp.apply_antiherm(op, ref, reverse=True)

    assert wfn[Determinant("200")] == pytest.approx(0.733340213919, abs=1e-9)
    assert wfn[Determinant("+-0")] == pytest.approx(-0.049868863373, abs=1e-9)
    assert wfn[Determinant("002")] == pytest.approx(-0.047410073759, abs=1e-9)
    assert wfn[Determinant("020")] == pytest.approx(0.676180171388, abs=1e-9)
    assert wfn[Determinant("-+0")] == pytest.approx(0.016058887563, abs=1e-9)


def set_up_operator(norb, nocc, amp=0.1):
    # Create a random operator
    oplist = forte2.SparseOperatorList()

    for i in range(nocc):
        for a in range(nocc, norb):
            oplist.add(f"[{a}a+ {i}a-]", amp / (1 + (a - i) ** 2))
            oplist.add(f"[{a}b+ {i}b-]", amp / (1 + (a - i) ** 2))

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, norb):
                for b in range(nocc, norb):
                    if i < j and a < b:
                        oplist.add(
                            f"[{a}a+ {b}a+ {j}a- {i}a-]",
                            amp / (1 + (a + b - i - j) ** 2),
                        )
                        oplist.add(
                            f"[{a}b+ {b}b+ {j}b- {i}b-]",
                            amp / (1 + (a + b - i - j) ** 2),
                        )
                    oplist.add(
                        f"[{a}a+ {b}b+ {j}b- {i}a-]", amp / (1 + (a + b - i - j) ** 2)
                    )
    print(f"Number of terms in the operator list: {len(oplist)}")
    return oplist


def test_equivalence_exp_and_factexp():
    # Compare the performance of the two methods to apply an operator to a state
    # when the operator all commute with each other

    norb = 10
    nocc = 5
    amp = 0.1
    oplist = set_up_operator(norb=norb, nocc=nocc, amp=amp)
    op = oplist.to_operator()

    # Apply the operator to the reference state timing it
    ref = forte2.SparseState({Determinant("2" * nocc): 1.0})
    start = time.time()
    exp = forte2.SparseExp()
    A = exp.apply_op(op, ref)
    end = time.time()
    print(f"Time to apply operator: {end - start:.8f} (SparseExp)")

    # Apply the operator to the reference state timing it
    ref = forte2.SparseState({Determinant("2" * nocc): 1.0})
    start = time.time()
    exp = forte2.SparseFactExp()
    B = exp.apply_op(oplist, ref)
    end = time.time()
    print(f"Time to apply operator: {end - start:.8f} (SparseFactExp)")

    # Check that the two methods give the same result
    AmB = forte2.SparseState(A)
    AmB -= B
    print(f"|A| = {A.norm()}")
    print(f"|B| = {B.norm()}")
    print(f"size(A) = {len(A)}")
    print(f"size(B) = {len(B)}")
    print(f"|A - B| = {AmB.norm()}")
    assert abs(AmB.norm()) < 1e-9


def test_factexp_unitarity():
    norb = 10
    nocc = 5
    amp = 0.1
    oplist = set_up_operator(norb=norb, nocc=nocc, amp=amp)

    # Apply the operator to the reference state timing it
    ref = forte2.SparseState({Determinant("2" * nocc): 1.0})
    start = time.time()
    exp = forte2.SparseFactExp(screen_thresh=1.0e-14)
    C = exp.apply_antiherm(oplist, ref)
    print(f"Size of C = {len(C)}")
    end = time.time()
    print(f"Time to apply operator: {end - start:.8f} (SparsFactExp::antiherm)")
    print(f"|C| = {C.norm()}")
    assert C.norm() == pytest.approx(1.0, abs=1e-8)


def test_factexp_timing():
    norb = 30
    nocc = 2
    amp = 0.1
    oplist = set_up_operator(norb=norb, nocc=nocc, amp=amp)

    # Apply the operator to the reference state timing it
    ref = forte2.SparseState({Determinant("2" * nocc): 1.0})
    factexp = forte2.SparseFactExp(screen_thresh=1.0e-14)
    exp = forte2.SparseExp(maxk=100, screen_thresh=1.0e-14)

    start = time.time()
    C = factexp.apply_antiherm(oplist, ref)
    print(f"Size of C = {len(C)}")
    end = time.time()
    print(f"Time to apply operator (async): {end - start:.8f}")
    print(f"|C| = {C.norm()}")
    assert C.norm() == pytest.approx(1.0, abs=1e-8)

    C = factexp.apply_antiherm_serial(oplist, ref)
    print(f"Size of C = {len(C)}")
    end = time.time()
    print(f"Time to apply operator (serial): {end - start:.8f}")
    print(f"|C| = {C.norm()}")
    assert C.norm() == pytest.approx(1.0, abs=1e-8)

    C = exp.apply_antiherm(oplist, ref)
    print(f"Size of C = {len(C)}")
    end = time.time()
    print(f"Time to apply operator (serial SparseExp): {end - start:.8f}")
    print(f"|C| = {C.norm()}")
    assert C.norm() == pytest.approx(1.0, abs=1e-8)


def test_idempotent_complex():
    op = forte2.SparseOperatorList()
    op.add("[0a+ 0a-]", np.pi * 0.25j)
    exp = forte2.SparseExp(maxk=100, screen_thresh=1e-15)
    factexp = forte2.SparseFactExp()
    ref = forte2.SparseState({Determinant("20"): 1.0})
    s1 = exp.apply_op(op, ref)
    s2 = factexp.apply_op(op, ref)
    assert s1[Determinant("20")] == pytest.approx(s2[Determinant("20")], abs=1e-9)
    assert s2[Determinant("20")] == pytest.approx(
        np.sqrt(2) * (1.0 + 1.0j) / 2, abs=1e-9
    )
    s1 = exp.apply_antiherm(op, ref)
    s2 = factexp.apply_antiherm(op, ref)
    assert s1[Determinant("20")] == pytest.approx(s2[Determinant("20")], abs=1e-9)
    assert s2[Determinant("20")] == pytest.approx(1.0j, abs=1e-9)
    op = forte2.SparseOperatorList()
    op.add("[1a+ 1a-]", np.pi * 0.25j)
    s1 = exp.apply_antiherm(op, ref)
    s2 = factexp.apply_antiherm(op, ref)
    assert s1[Determinant("20")] == pytest.approx(s2[Determinant("20")], abs=1e-9)
    assert s2[Determinant("20")] == pytest.approx(1.0, abs=1e-9)


def test_exp_apply_complex():
    # Test the factorized exponential operator with an antihermitian operator with complex coefficients
    op = forte2.SparseOperatorList()
    op.add("[1a+ 0a-]", 0.1 + 0.2j)

    op_inv = forte2.SparseOperatorList()
    op_inv.add("[0a+ 1a-]", 0.1 - 0.2j)

    exp = forte2.SparseExp(maxk=100, screen_thresh=1e-15)
    factexp = forte2.SparseFactExp()
    ref = forte2.SparseState({Determinant("20"): 0.5, Determinant("02"): 0.8660254038})

    s1 = exp.apply_antiherm(op, ref)
    s2 = factexp.apply_antiherm(op, ref)
    assert s1[Determinant("20")] == pytest.approx(s2[Determinant("20")], abs=1e-9)
    assert s1[Determinant("02")] == pytest.approx(s2[Determinant("02")], abs=1e-9)
    assert s1[Determinant("+-")] == pytest.approx(s2[Determinant("+-")], abs=1e-9)
    assert s1[Determinant("-+")] == pytest.approx(s2[Determinant("-+")], abs=1e-9)

    s1 = exp.apply_antiherm(op, ref)
    s2 = exp.apply_antiherm(op_inv, s1)
    assert s2[Determinant("20")] == pytest.approx(0.5, abs=1e-9)
    assert s2[Determinant("02")] == pytest.approx(0.8660254038, abs=1e-9)

    s1 = factexp.apply_antiherm(op, ref, inverse=True)
    s2 = factexp.apply_antiherm(op_inv, ref, inverse=False)
    assert s1 == s2


def test_fact_exp_deriv_complex():
    factexp = forte2.SparseFactExp()
    exp = forte2.SparseExp(maxk=100, screen_thresh=1e-15)

    # create an operator with a general complex coefficient
    theta = 0.271 + 0.829j
    t = forte2.SparseOperatorList()
    t.add("[1a+ 0a-]", theta)

    # create a state with two determinants
    psi = forte2.SparseState({Determinant("2"): 0.866, Determinant("-+"): 0.5})

    # SparseExp will compute the action of the exponential operator numerically using Taylor expansion
    res = exp.apply_antiherm(t, psi)
    # finite difference derivatives wrt the real part
    dt = 1e-6
    tdt = forte2.SparseOperatorList()
    tdt.add("[1a+ 0a-]", theta + dt)
    res2 = exp.apply_antiherm(tdt, psi)

    # analytical derivatives from SparseFactExp
    deriv = factexp.apply_antiherm_deriv(*t(0), psi)

    # assert the analytical derivatives match the finite difference derivatives
    dx1 = (res2[Determinant("2")] - res[Determinant("2")]) / dt
    assert deriv[0][Determinant("2")] == pytest.approx(dx1, abs=1e-6)
    dx2 = (res2[Determinant("-+")] - res[Determinant("-+")]) / dt
    assert deriv[0][Determinant("-+")] == pytest.approx(dx2, abs=1e-6)

    # finite difference derivatives wrt the imaginary part
    dt = 1e-6 * 1j
    tdt = forte2.SparseOperatorList()
    tdt.add("[1a+ 0a-]", theta + dt)
    res2 = exp.apply_antiherm(tdt, psi)

    # assert the analytical derivatives match the finite difference derivatives
    dy1 = (res2[Determinant("2")] - res[Determinant("2")]) / dt.imag
    assert deriv[1][Determinant("2")] == pytest.approx(dy1, abs=1e-6)
    dy2 = (res2[Determinant("-+")] - res[Determinant("-+")]) / dt.imag
    assert deriv[1][Determinant("-+")] == pytest.approx(dy2, abs=1e-6)


def test_fact_exp_deriv_real_antiherm():
    # test that the derivatives of a real antihermitian operator are as expected
    factexp = forte2.SparseFactExp()
    theta = 0.176
    t = forte2.SparseOperatorList()
    t.add("[1a+ 0a-]", theta)
    psi = forte2.SparseState({Determinant("2"): 1.0})

    deriv = factexp.apply_antiherm_deriv(*t(0), psi)
    assert deriv[0][Determinant("2")] == pytest.approx(-np.sin(theta), abs=1e-6)
    assert deriv[0][Determinant("-+")] == pytest.approx(np.cos(theta), abs=1e-6)
    assert deriv[1][Determinant("2")] == pytest.approx(0.0, abs=1e-6)
    assert deriv[1][Determinant("-+")] == pytest.approx(
        1j * np.sin(theta) / theta, abs=1e-6
    )


def test_fact_exp_deriv_imagherm():
    # Test a simple imagherm case
    factexp = forte2.SparseFactExp()
    theta = 0.127
    t = forte2.SparseOperatorList()
    t.add("[1a+ 0a-]", theta * 1j)
    psi = forte2.SparseState({Determinant("2"): 1.0})

    deriv = factexp.apply_antiherm_deriv(*t(0), psi)
    assert deriv[1][Determinant("2")] == pytest.approx(-np.sin(theta), abs=1e-6)
    assert deriv[1][Determinant("-+")] == pytest.approx(1j * np.cos(theta), abs=1e-6)


def test_fact_exp_deriv_zero_division():
    # test the analytical derivatives at theta = 0
    # to make sure there's no division by zero errors
    factexp = forte2.SparseFactExp()
    exp = forte2.SparseExp(maxk=100, screen_thresh=1e-15)
    theta = 0
    t = forte2.SparseOperatorList()
    t.add("[1a+ 0a-]", theta)
    psi = forte2.SparseState({Determinant("2"): 0.866, Determinant("-+"): 0.5})

    res = exp.apply_antiherm(t, psi)
    deriv = factexp.apply_antiherm_deriv(*t(0), psi)

    dt = 1e-6
    tdt = forte2.SparseOperatorList()
    tdt.add("[1a+ 0a-]", theta + dt)
    res2 = exp.apply_antiherm(tdt, psi)

    dx1 = (res2[Determinant("2")] - res[Determinant("2")]) / dt
    assert deriv[0][Determinant("2")] == pytest.approx(dx1, abs=1e-6)
    dx2 = (res2[Determinant("-+")] - res[Determinant("-+")]) / dt
    assert deriv[0][Determinant("-+")] == pytest.approx(dx2, abs=1e-6)

import forte2
import numpy as np
import pytest


def det(s):
    return forte2.Determinant(s)


def test_sparse_operator_creation():
    sop = forte2.SparseOperator()
    sop.add([1], [], [1], [])
    assert sop["[1a+ 1a-]"] == 1.0

    # verify that order is ignored when adding terms (1,3 and 3,1 are the same)
    sop = forte2.SparseOperator()
    sop.add([3, 1], [], [], [])
    assert sop["[1a+ 3a+]"] == 1.0

    sop = forte2.SparseOperator()
    sop.add([1, 3], [], [], [])
    assert sop["[1a+ 3a+]"] == 1.0


def test_sparse_operator():
    # test get/set
    sop = forte2.SparseOperator()
    sop.add("[]", 2.3)
    sop.add("[0a+ 0b+ 0b- 0a-]")
    to_str = sorted(sop.str())
    assert to_str == sorted(["(2.3 + 0i) * []", "(1 + 0i) * [0a+ 0b+ 0b- 0a-]"])
    to_latex = sop.latex()
    assert (
        to_latex
        == r"(2.300000 + 0.000000 i)\; + \;\hat{a}_{0 \alpha}^\dagger\hat{a}_{0 \beta}^\dagger\hat{a}_{0 \beta}\hat{a}_{0 \alpha}"
    )

    sop = forte2.SparseOperator()
    sop.add("[1a+ 1b+ 0b- 0a-]", 1.0)
    to_str = sop.str()
    assert to_str == ["(1 + 0i) * [1a+ 1b+ 0b- 0a-]"]
    to_latex = sop.latex()
    assert (
        to_latex
        == r"\;\hat{a}_{1 \alpha}^\dagger\hat{a}_{1 \beta}^\dagger\hat{a}_{0 \beta}\hat{a}_{0 \alpha}"
    )

    sop = forte2.SparseOperator()
    sop.add("[]", 1.0)
    sop.add("[0a+ 0b+ 0b- 0a-]", 1.0)

    assert sop.coefficient("[]") == 1.0
    assert sop.coefficient("[0a+ 0b+ 0b- 0a-]") == 1.0
    sop.set_coefficient("[]", 0.5)
    sop.set_coefficient("[0a+ 0b+ 0b- 0a-]", 0.3)
    assert sop.coefficient("[]") == 0.5
    assert sop.coefficient("[0a+ 0b+ 0b- 0a-]") == 0.3
    # remove one term from sop
    sop.remove("[0a+ 0b+ 0b- 0a-]")
    # check the size
    assert len(sop) == 1
    # check the element
    assert sop.coefficient("[]") == 0.5
    assert sop.coefficient("[0a+ 0b+ 0b- 0a-]") == 0.0
    assert len(sop) == 1
    sop.remove("[]")
    assert len(sop) == 0

    # copy a term into a new operator
    sop = forte2.SparseOperator()
    sop.add("[]", 1.0)
    sop.add("[0a+ 0b+ 0b- 0a-]", 1.0)

    # test apply operator against safe implementation
    sop = forte2.SparseOperator()
    sop.add("[2a+ 0a-]", 0.0)
    sop.add("[1a+ 0a-]", 0.1)
    sop.add("[1b+ 0b-]", 0.3)
    sop.add("[1a+ 1b+ 0b- 0a-]", 0.5)
    dtest = forte2.Determinant("20")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = sop.apply_to_state(ref)
    assert wfn[det("20")] == pytest.approx(0.0, abs=1e-9)
    assert wfn[det("-+")] == pytest.approx(0.1, abs=1e-9)
    assert wfn[det("+-")] == pytest.approx(0.3, abs=1e-9)
    assert wfn[det("02")] == pytest.approx(0.5, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[2a+ 0a-]", 0.0)
    sop.add("[1a+ 0a-]", 0.1)
    sop.add("[1b+ 0b-]", 0.3)
    sop.add("[1a+ 1b+ 0b- 0a-]", 0.7)
    ref = forte2.SparseState({det("20"): 1.0, det("02"): 0.5})
    wfn = forte2.apply_antiherm(sop, ref)
    assert wfn[det("20")] == pytest.approx(-0.35, abs=1e-9)
    assert wfn[det("02")] == pytest.approx(0.7, abs=1e-9)
    assert wfn[det("+-")] == pytest.approx(0.25, abs=1e-9)
    assert wfn[det("-+")] == pytest.approx(-0.05, abs=1e-9)

    ### Operator ordering tests ###

    # test repeated operator exception
    sop = forte2.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[0a+ 0a+]", 1.0)

    sop = forte2.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[1b+ 1b+]", 1.0)

    sop = forte2.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[5a- 5a-]", 1.0)

    sop = forte2.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[3b- 3b-]", 1.0)

    # test ordering exception
    sop = forte2.SparseOperator()
    with pytest.raises(RuntimeError):
        sop.add("[0a+ 0b+ 0a- 0b-]", 1.0)

    sop = forte2.SparseOperator()
    sop.add("[]", 1.0)
    assert len(sop) == 1

    # test ordering: 0a+ 0b+ 0b- 0a- |2> = +|2>
    sop = forte2.SparseOperator()
    sop.add("[0a+ 0b+ 0b- 0a-]", 1.0)
    dtest = det("20")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 0a+ 0b+ 0a- 0b- |2> = -|2>
    sop = forte2.SparseOperator()
    sop.add("[0a+ 0b+ 0a- 0b-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0b- 0a- |2> = -|2>
    sop = forte2.SparseOperator()
    sop.add("[0b+ 0a+ 0b- 0a-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(-1.0, abs=1e-9)

    # test ordering: 0b+ 0a+ 0a- 0b- |2> = +|2>
    sop = forte2.SparseOperator()
    sop.add("[0b+ 0a+ 0a- 0b-]", 1.0, allow_reordering=True)
    dtest = det("20")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test ordering: 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    # 3a+ 4a+ 2b+ 1b- 0a- |22000> = + 3a+ 4a+ 2b+ 1b- |-2000>
    #                             = + 3a+ 4a+ 2b+ |-+000>
    #                             = + 3a+ 4a+ |-+-00>
    #                             = - 3a+ |-+-0+>
    #                             = + 3a+ |-+-++>
    sop = forte2.SparseOperator()
    sop.add("[3a+ 4a+ 2b+ 1b- 0a-]", 1.0)
    ref = forte2.SparseState({det("22"): 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[det("-+-++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[0a+]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det(""): 1.0}))
    assert wfn[det("+")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[0a+ 1a+]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det(""): 1.0}))
    assert wfn[det("++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[0a+ 1a+ 2a+]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det(""): 1.0}))
    assert wfn[det("+++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[0a+ 1a+ 2a+ 3a+]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det(""): 1.0}))
    assert wfn[det("++++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[0a+ 1a+ 2a+ 3a+ 4a+]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det(""): 1.0}))
    assert wfn[det("+++++")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[2a+ 0a+ 3a+ 1a+ 4a+]", 1.0, allow_reordering=True)
    wfn = forte2.apply_op(sop, forte2.SparseState({det(""): 1.0}))
    assert wfn[det("+++++")] == pytest.approx(-1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[0a-]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det("22222"): 1.0}))
    assert wfn[det("-2222")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[1a- 0a-]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det("22222"): 1.0}))
    assert wfn[det("--222")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[2a- 1a- 0a-]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det("22222"): 1.0}))
    assert wfn[det("---22")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[3a- 2a- 1a- 0a-]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det("22222"): 1.0}))
    assert wfn[det("----2")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[4a- 3a- 2a- 1a- 0a-]", 1.0)
    wfn = forte2.apply_op(sop, forte2.SparseState({det("22222"): 1.0}))
    assert wfn[det("-----")] == pytest.approx(1.0, abs=1e-9)

    sop = forte2.SparseOperator()
    sop.add("[1a- 3a- 2a- 4a- 0a-]", 1.0, allow_reordering=True)
    wfn = forte2.apply_op(sop, forte2.SparseState({det("22222"): 1.0}))
    assert wfn[det("-----")] == pytest.approx(-1.0, abs=1e-9)

    ### Test for cases that are supposed to return zero ###
    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    sop = forte2.SparseOperator()
    sop.add("[0a+ 0b+ 0b- 0a-]", 1.0)
    dtest = det("00")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    # test destroying empty orbitals: (0a+ 0b+ 0b- 0a-) |0> = 0
    sop = forte2.SparseOperator()
    sop.add("[0a+ 0b+ 0b-]", 1.0)
    dtest = det("00")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    sop = forte2.SparseOperator()
    sop.add("[1a+ 0a+ 0a-]", 1.0, allow_reordering=True)
    dtest = det("+")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    # test creating in filled orbitals: (0a+ 1a+ 0a-) |22> = 0
    sop = forte2.SparseOperator()
    sop.add("[1b+ 0a+ 0a-]", 1.0, allow_reordering=True)
    dtest = det("+")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    ### Number operator tests ###
    # test number operator: (0a+ 0a-) |0> = 0
    sop = forte2.SparseOperator()
    sop.add("[0a+ 0a-]", 1.0)
    dtest = det("0")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |+> = |+>
    sop = forte2.SparseOperator()
    sop.add("[0a+ 0a-]", 1.0)
    dtest = det("+")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0a+ 0a-) |-> = 0
    sop = forte2.SparseOperator()
    sop.add("[0a+ 0a-]", 1.0)
    dtest = det("-")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    # test number operator: (0a+ 0a-) |2> = |2>
    sop = forte2.SparseOperator()
    sop.add("[0a+ 0a-]", 1.0)
    dtest = det("2")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |0> = 0
    sop = forte2.SparseOperator()
    sop.add("[0b+ 0b-]", 1.0)
    dtest = det("0")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |+> = 0
    sop = forte2.SparseOperator()
    sop.add("[0b+ 0b-]", 1.0)
    dtest = det("+")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    # test number operator: (0b+ 0b-) |-> = |->
    sop = forte2.SparseOperator()
    sop.add("[0b+ 0b-]", 1.0)
    dtest = det("-")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[dtest] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (0b+ 0b-) |2> = |2>
    sop = forte2.SparseOperator()
    sop.add("[0b+ 0b-]", 1.0)
    ref = forte2.SparseState({det("2"): 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[det("2")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |222> = |222>
    sop = forte2.SparseOperator()
    sop.add("[2a+ 2a-]", 1.0)
    sop.add("[2b+ 2b-]", 1.0)
    ref = forte2.SparseState({det("222"): 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[det("222")] == pytest.approx(2.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |22+> = |22+>
    sop = forte2.SparseOperator()
    sop.add("[2a+ 2a-]", 1.0)
    sop.add("[2b+ 2b-]", 1.0)
    ref = forte2.SparseState({det("22+"): 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[det("22+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |22-> = |22->
    sop = forte2.SparseOperator()
    sop.add("[2a+ 2a-]", 1.0)
    sop.add("[2b+ 2b-]", 1.0)
    ref = forte2.SparseState({det("22-"): 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[det("22-")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (2a+ 2a- + 2b+ 2b-) |220> = |220>
    sop = forte2.SparseOperator()
    sop.add("[2a+ 2a-]", 1.0)
    sop.add("[2b+ 2b-]", 1.0)
    ref = forte2.SparseState({det("220"): 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert dtest not in wfn

    ### Excitation operator tests ###
    # test excitation operator: (3a+ 0a-) |2200> = 3a+ |-200> = -|-20+>
    sop = forte2.SparseOperator()
    sop.add("[3a+ 0a-]", 1.0)
    dtest = det("22")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[det("-20+")] == pytest.approx(-1.0, abs=1e-9)

    # test excitation operator: (0a- 3a+) |22> = 0a- |220+> = |-20+>
    sop = forte2.SparseOperator()
    sop.add("[0a- 3a+]", 1.0, allow_reordering=True)
    dtest = det("22")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[det("-20+")] == pytest.approx(1.0, abs=1e-9)

    # test number operator: (3b+ 0b-) |2200> = 3b+ |+200> = |+20->
    sop = forte2.SparseOperator()
    sop.add("[3b+ 0b-]", 1.0)
    dtest = det("22")
    ref = forte2.SparseState({dtest: 1.0})
    wfn = forte2.apply_op(sop, ref)
    assert wfn[det("+20-")] == pytest.approx(-1.0, abs=1e-9)

    ### Adjoint tests
    # test adjoint of SqOperator is idempotent
    sqop = forte2.sparse_operator("[0a+ 1a+ 0b+ 2b- 2a-]", 1.0)
    sqopd = sqop.adjoint()
    sqopdd = sqopd.adjoint()
    assert sqop.str() == sqopdd.str()
    # test adjoint of SqOperator
    ref = forte2.sparse_operator("[2a+ 2b+ 0b- 1a- 0a-]", 1.0)
    assert sqopd.str() == ref.str()
    # test adjoint of SparseOperator is idempotent
    sop = forte2.SparseOperator()
    sop.add("[3a+ 0a-]", 1.0)
    sop.add("[2b+ 1b-]", 1.0)
    sopd = sop.adjoint()
    sopdd = sopd.adjoint()
    assert sop.str() == sopdd.str()
    # test adjoint of SparseOperator
    ref = forte2.SparseOperator()
    ref.add("[0a+ 3a-]", 1.0)
    ref.add("[1b+ 2b-]", 1.0)
    assert sopd.str() == ref.str()


def test_sparse_operator_product():
    sop1 = forte2.SparseOperator()
    sop1.add("[1a+ 1a-]", 1.0)
    sop1.add("[0a+ 0a-]", 1.0)
    sop2 = forte2.SparseOperator()
    sop2.add("[3a+ 3a-]", 1.0)
    sop2.add("[2a+ 2a-]", 1.0)
    sop3 = sop1 @ sop2
    sop3_test = forte2.SparseOperator()
    sop3_test.add("[0a+ 2a+ 2a- 0a-]", 1.0)
    sop3_test.add("[0a+ 3a+ 3a- 0a-]", 1.0)
    sop3_test.add("[1a+ 2a+ 2a- 1a-]", 1.0)
    sop3_test.add("[1a+ 3a+ 3a- 1a-]", 1.0)
    assert sop3.str() == sop3_test.str()


def test_sparse_operator_list_reverse():
    sopl = forte2.SparseOperatorList()
    sopl.add("[1a+ 1a-]", 1.0)
    sopl.add("[0a+ 0a-]", 2.0)
    reversed_sopl = sopl.reverse()
    assert len(sopl) == 2
    assert reversed_sopl[0] == 2.0
    assert reversed_sopl[1] == 1.0
    # assert reversed_sopl(0)[0].str() == "[0a+ 0a-]"
    # assert reversed_sopl(1)[0].str() == "[1a+ 1a-]"


def test_sparse_operator_list_remove():
    sopl = forte2.SparseOperatorList()
    sopl.add("[1a+ 1a-]", 1.0)
    sopl.add("[0a+ 0a-]", 1.0)
    sopl.remove("[1a+ 1a-]")
    assert len(sopl) == 1


def test_sparse_operator_list_add():
    sop1 = forte2.SparseOperatorList()
    sop1.add("[1a+ 1a-]", 1.0)
    sop2 = forte2.SparseOperatorList()
    sop2.add("[0a+ 0a-]", 1.0)
    sop3 = sop1 + sop2
    assert len(sop3) == 2
    # assert sop3(0)[0].str() == "[1a+ 1a-]"
    # assert sop3(1)[0].str() == "[0a+ 0a-]"


def test_sparse_operator_hamiltonian():
    from forte2.jkbuilder.jkbuilder import DFFockBuilder

    erhf = -76.021765988335
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT")

    scf = forte2.scf.RHF(charge=0)(system)
    scf.econv = 1e-10
    scf.dconv = 1e-8
    scf.run()
    assert np.isclose(
        scf.E, erhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {erhf}"

    hf_reference = forte2.Determinant("22222")
    print(f"\n  Hartree-Fock determinant: {hf_reference}")

    # Compute the energy using the sparse operator
    hf_state = forte2.SparseState({hf_reference: 1.0})
    scalar = forte2.ints.nuclear_repulsion(system.atoms)
    oei = forte2.ints.kinetic(system.basis) + forte2.ints.nuclear(
        system.basis, system.atoms
    )
    C = scf.C[0]
    oei_mo = np.einsum("pq,pi,qj->ij", oei, C, C, optimize=True)
    jkbuilder = DFFockBuilder(system)
    eri_ab = jkbuilder.two_electron_integrals_block(C)
    eri_aa = eri_ab.copy()
    eri_aa -= eri_aa.swapaxes(2, 3)

    ham = forte2.sparse_operator_hamiltonian(
        scalar, oei_mo, oei_mo, eri_aa, eri_ab, eri_aa
    )
    Href = forte2.apply_op(ham, hf_state)
    energy = forte2.overlap(hf_state, Href).real
    print(f"Energy from SparseOperator Hamiltonian: {energy}")

    assert energy == pytest.approx(
        scf.E, abs=1e-8
    ), f"Hamiltonian energy {energy} is not close to SCF energy {scf.E}"


if __name__ == "__main__":
    test_sparse_operator_creation()
    test_sparse_operator()
    test_sparse_operator_product()
    test_sparse_operator_list_reverse()
    test_sparse_operator_list_remove()
    test_sparse_operator_list_add()
    test_sparse_operator_hamiltonian()

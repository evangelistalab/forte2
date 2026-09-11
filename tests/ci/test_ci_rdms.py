import numpy as np
import pytest

from forte2 import System, RHF, CI, State
from forte2.base_classes import CIParams
from forte2.helpers.comparisons import approx
from forte2.lib.ci_helpers import CISigmaBuilder, CIStrings
from forte2.lib import rdms
from rdm_debug_utils import sparse_state_from_ci_vector
from forte2.ci.ci_utils import (
    make_2cumulant_sf,
    make_3cumulant_sf,
)


def _pair_index_gt(norb):
    """Canonical (p, q) pairs with p > q."""
    return np.tril_indices(norb, -1)


def _triplet_index_gt(norb):
    """Canonical (p, q, r) triplets with p > q > r."""
    if norb < 3:
        empty = np.array([], dtype=int)
        return empty, empty, empty
    p_list, q_list, r_list = [], [], []
    for p in range(2, norb):
        q_idx, r_idx = np.tril_indices(p, -1)
        p_list.append(np.full(q_idx.shape, p))
        q_list.append(q_idx)
        r_list.append(r_idx)
    return np.concatenate(p_list), np.concatenate(q_list), np.concatenate(r_list)


def _pack_ss_2rdm(full, norb):
    """Extract the canonical p>q,r>s packed slice, matching the native aa/bb_2rdm layout."""
    p_idx, q_idx = _pair_index_gt(norb)
    return full[p_idx[:, None], q_idx[:, None], p_idx[None, :], q_idx[None, :]]


def _pack_sss_3rdm(full, norb):
    """Extract the canonical p>q>r,s>t>u packed slice, matching the native aaa/bbb_3rdm layout."""
    p_idx, q_idx, r_idx = _triplet_index_gt(norb)
    return full[
        p_idx[:, None],
        q_idx[:, None],
        r_idx[:, None],
        p_idx[None, :],
        q_idx[None, :],
        r_idx[None, :],
    ]


def _pack_aab_3rdm(full, norb):
    """Extract the canonical p>q,s>t slice, matching the native aab_3rdm layout
    (npair, norb, npair, norb) with r, u the unrestricted beta indices."""
    p_idx, q_idx = _pair_index_gt(norb)
    free = np.arange(norb)
    return full[
        p_idx[:, None, None, None],
        q_idx[:, None, None, None],
        free[None, :, None, None],
        p_idx[None, None, :, None],
        q_idx[None, None, :, None],
        free[None, None, None, :],
    ]


def _pack_abb_3rdm(full, norb):
    """Extract the canonical q>r,t>u slice, matching the native abb_3rdm layout
    (norb, npair, norb, npair) with p, s the unrestricted alpha indices."""
    q_idx, r_idx = _pair_index_gt(norb)
    free = np.arange(norb)
    return full[
        free[:, None, None, None],
        q_idx[None, :, None, None],
        r_idx[None, :, None, None],
        free[None, None, :, None],
        q_idx[None, None, None, :],
        r_idx[None, None, None, :],
    ]


def compare_rdms(ci):
    rdm_threshold = 1e-12

    ci_solver = ci.sub_solvers[0]
    norb = ci_solver.norb
    dets = ci_solver.dets

    ci_0_det = np.zeros((ci_solver.ndet))
    ci_1_det = np.zeros((ci_solver.ndet))
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, 0], ci_0_det)
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, 1], ci_1_det)

    def check(label, actual, ref):
        err = np.linalg.norm(actual - ref)
        assert (
            err < rdm_threshold
        ), f"Norm of the difference for {label} is too large: {err:.12f}."

    # Every (left, right) root pair covers both same-state RDMs and transition RDMs.
    for (l, cl_det), (r, cr_det) in [
        ((0, ci_0_det), (0, ci_0_det)),
        ((1, ci_1_det), (1, ci_1_det)),
        ((0, ci_0_det), (1, ci_1_det)),
        ((1, ci_1_det), (0, ci_0_det)),
    ]:
        left = sparse_state_from_ci_vector(dets, cl_det)
        right = sparse_state_from_ci_vector(dets, cr_det)

        # 1-RDMs: fast (block-addressed) kernels vs. the generic SparseState reference.
        a1, b1 = ci_solver.make_rdm(l, r, order=1, spin_type="sd")
        a1_ref = rdms.compute_a_1rdm(left, right, norb)
        b1_ref = rdms.compute_b_1rdm(left, right, norb)
        check(f"a_1rdm({l},{r})", a1, a1_ref)
        check(f"b_1rdm({l},{r})", b1, b1_ref)

        sf1 = ci_solver.make_rdm(l, r, order=1, spin_type="sf")
        check(f"sf_1rdm({l},{r})", sf1, a1_ref + b1_ref)

        # 2-RDMs
        aa2, ab2, bb2 = ci_solver.make_rdm(l, r, order=2, spin_type="sd")
        aa2_ref = rdms.compute_aa_2rdm(left, right, norb)
        ab2_ref = rdms.compute_ab_2rdm(left, right, norb)
        bb2_ref = rdms.compute_bb_2rdm(left, right, norb)
        check(f"aa_2rdm({l},{r})", aa2, _pack_ss_2rdm(aa2_ref, norb))
        check(f"ab_2rdm({l},{r})", ab2, ab2_ref)
        check(f"bb_2rdm({l},{r})", bb2, _pack_ss_2rdm(bb2_ref, norb))

        # sf_2rdm, checked independently against the (already full) sparse references directly
        # rather than through spin_free_2rdm, which expects the native packed aa/bb format.
        sf2 = ci_solver.make_rdm(l, r, order=2, spin_type="sf")
        sf2_ref = ab2_ref + ab2_ref.transpose(1, 0, 3, 2) + aa2_ref + bb2_ref
        check(f"sf_2rdm({l},{r})", sf2, sf2_ref)

        # 3-RDMs
        aaa3, aab3, abb3, bbb3 = ci_solver.make_rdm(l, r, order=3, spin_type="sd")
        aaa3_ref = rdms.compute_aaa_3rdm(left, right, norb)
        aab3_ref = rdms.compute_aab_3rdm(left, right, norb)
        abb3_ref = rdms.compute_abb_3rdm(left, right, norb)
        bbb3_ref = rdms.compute_bbb_3rdm(left, right, norb)
        check(f"aaa_3rdm({l},{r})", aaa3, _pack_sss_3rdm(aaa3_ref, norb))
        check(f"aab_3rdm({l},{r})", aab3, _pack_aab_3rdm(aab3_ref, norb))
        check(f"abb_3rdm({l},{r})", abb3, _pack_abb_3rdm(abb3_ref, norb))
        check(f"bbb_3rdm({l},{r})", bbb3, _pack_sss_3rdm(bbb3_ref, norb))

    # Cumulants, checked at the same-state pairs only (no cross-state 2-/3-RDM support).
    for root in (0, 1):
        sf1 = ci_solver.make_rdm(root, order=1, spin_type="sf")
        sf2 = ci_solver.make_rdm(root, order=2, spin_type="sf")
        sf3 = ci_solver.make_rdm(root, order=3, spin_type="sf")

        sf_2cumulant = ci_solver.make_cumulant(root, order=2, spin_type="sf")
        check(f"sf_2cumulant({root})", sf_2cumulant, make_2cumulant_sf(sf1, sf2))

        sf_3cumulant = ci_solver.make_cumulant(root, order=3, spin_type="sf")
        check(f"sf_3cumulant({root})", sf_3cumulant, make_3cumulant_sf(sf1, sf2, sf3))


def test_ci_rdms_1():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        do_test_rdms=True,
        nroots=2,
    )(rhf)
    ci.run()
    compare_rdms(ci)

    assert ci.E[0] == approx(-100.019788438077)


def test_ci_rdms_sa():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        states=[
            State(nel=10, multiplicity=1, ms=0.0),
            State(nel=10, multiplicity=3, ms=1.0),
        ],
        nroots=[2, 1],
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        do_test_rdms=True,
    )(rhf)
    ci.run()
    compare_rdms(ci)

    assert ci.E[0] == approx(-100.01978843799819)
    assert ci.E[1] == approx(-99.68758394141096)
    assert ci.E[2] == approx(-99.7052645828813)

    g1 = ci.make_average_rdm(1)
    l2 = ci.make_average_cumulant(2)
    e_avg = ci.compute_average_energy()
    ci_ints = ci.sub_solvers[0].ints

    e_from_cumulants = ci_ints.E
    e_from_cumulants += np.einsum("pq,pq->", ci_ints.H, g1)
    e_from_cumulants += 0.5 * np.einsum("pqrs,pqrs->", ci_ints.V, l2)
    e_from_cumulants += 0.5 * np.einsum("pqrs,pr,qs->", ci_ints.V, g1, g1)
    e_from_cumulants -= 0.25 * np.einsum("pqrs,ps,qr->", ci_ints.V, g1, g1)

    assert e_avg == approx(e_from_cumulants)


def test_ci_rdm_validation():
    """make_rdm/make_cumulant argument handling: spin-type aliases, and rejection of a bad
    root, order, or spin type, and of an unsupported cross-state order, on both the
    single-state solver and the state-averaged solver."""
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.058354421806
    """
    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        states=[
            State(nel=2, multiplicity=1, ms=0.0),
            State(nel=2, multiplicity=3, ms=1.0),
        ],
        nroots=[1, 1],
        active_orbitals=[0, 1],
    )(rhf)
    ci.run()

    solver = ci.sub_solvers[0]

    # The spelled-out spin types are aliases of the canonical two-letter forms.
    for alias in ("spin_free", "spin-free"):
        assert np.allclose(
            ci.make_rdm(0, order=1, spin_type=alias),
            ci.make_rdm(0, order=1, spin_type="sf"),
        )
        assert np.allclose(
            ci.make_cumulant(0, order=2, spin_type=alias),
            ci.make_cumulant(0, order=2, spin_type="sf"),
        )
    for alias in ("spin_dependent", "spin-dependent"):
        for aliased, canonical in zip(
            solver.make_rdm(0, order=2, spin_type=alias),
            solver.make_rdm(0, order=2, spin_type="sd"),
        ):
            assert np.allclose(aliased, canonical)

    with pytest.raises(ValueError, match="left_root must be between 0"):
        solver.make_rdm(5, order=1, spin_type="sd")

    with pytest.raises(ValueError, match="order must be one of"):
        solver.make_rdm(0, order=4, spin_type="sd")

    with pytest.raises(ValueError, match="spin_type must be one of"):
        solver.make_rdm(0, order=1, spin_type="so")

    with pytest.raises(ValueError, match="absolute_root must be between 0"):
        ci.make_rdm(5, order=1, spin_type="sd")

    with pytest.raises(ValueError, match="order must be one of"):
        ci.make_rdm(0, order=4, spin_type="sd")

    with pytest.raises(ValueError, match="spin_type must be one of"):
        ci.make_rdm(0, order=1, spin_type="so")

    with pytest.raises(
        ValueError, match="Cross-state requests are not supported for order 2"
    ):
        ci.make_rdm(0, 1, order=2, spin_type="sd")

    # A solver that never declared a capability is reported by name, rather than surfacing
    # as an AttributeError or a TypeError from the membership test. Shadow the class
    # variable on the instance to stand in for a subclass that forgot to declare it.
    ci._rdm_spin_types = None
    with pytest.raises(NotImplementedError, match=r"CI does not declare spin_types"):
        ci.make_rdm(0, order=1, spin_type="sd")
    del ci._rdm_spin_types

    ci._rdm_cross_state_orders = None
    with pytest.raises(
        NotImplementedError, match=r"CI does not declare cross_state_orders"
    ):
        ci.make_rdm(0, order=1, spin_type="sd")
    del ci._rdm_cross_state_orders

    solver._cumulant_orders = None
    with pytest.raises(
        NotImplementedError, match=r"_CISingleStateSolver does not declare orders"
    ):
        solver.make_cumulant(0, order=2, spin_type="sf")
    del solver._cumulant_orders


def test_ci_rdms_respect_small_builder_memory_across_composite_hole_chunks():
    """Low-memory composite-hole chunks must agree with an unchunked contraction."""
    norb = 12
    lists = CIStrings(2, 5, 0, [[0] * norb], [], [])
    H = np.zeros((norb, norb))
    V = np.zeros((norb, norb, norb, norb))
    low_memory_builder = CISigmaBuilder(lists, 0.0, H, V, 0)
    reference_builder = CISigmaBuilder(lists, 0.0, H, V, 0)

    rng = np.random.default_rng(204)
    C_left = rng.normal(size=lists.ndet)
    C_right = rng.normal(size=lists.ndet)
    C_left /= np.linalg.norm(C_left)
    C_right /= np.linalg.norm(C_right)

    with pytest.raises(ValueError, match="memory must be non-negative"):
        low_memory_builder.set_memory(-1)

    # A zero-byte limit cannot hold even one column, and must fail without corrupting the builder.
    low_memory_builder.set_memory(0)
    with pytest.raises(RuntimeError, match="too small for one K-block column"):
        low_memory_builder.ab_2rdm(C_left, C_right)

    # One MiB forces chunks that split the flattened (Ka, Kb) index, including within Kb.
    low_memory_builder.set_memory(1)
    reference_builder.set_memory(64)
    for method_name in ("ab_2rdm", "aab_3rdm", "abb_3rdm"):
        low_memory_rdm = getattr(low_memory_builder, method_name)(C_left, C_right)
        reference_rdm = getattr(reference_builder, method_name)(C_left, C_right)
        np.testing.assert_allclose(
            low_memory_rdm, reference_rdm, rtol=1e-12, atol=1e-12
        )


def test_ci_builder_memory_reconfiguration_is_retry_safe():
    """A failed cached-buffer request must not corrupt a later sigma build."""
    norb = 6
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    H = np.diag(np.linspace(-1.0, 1.0, norb))
    V = np.zeros((norb, norb, norb, norb))
    builder = CISigmaBuilder(lists, 0.25, H, V, 0, "hz")

    rng = np.random.default_rng(205)
    basis = rng.normal(size=lists.ndet)
    basis /= np.linalg.norm(basis)

    builder.set_memory(1)
    expected = np.zeros(lists.ndet)
    builder.Hamiltonian(basis, expected)

    builder.set_memory(0)
    with pytest.raises(RuntimeError, match="too small for one K-block column"):
        builder.Hamiltonian(basis, np.zeros(lists.ndet))

    builder.set_memory(1)
    actual = np.zeros(lists.ndet)
    builder.Hamiltonian(basis, actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_sf_cumulants_are_ms_invariant():
    """
    Spin-free cumulants need to be identical for every Ms component of a multiplet.
    """
    # An evenly spaced chain introduces spatial degeneracies in the CI spectrum,
    # which can cause arbitrariness between CI vectors if different ms states.
    # The uneven spacings keep every root non-degenerate.
    z_coords = [0.0, 1.9, 3.6, 5.6, 7.4, 9.5]
    xyz = "\n".join(f"H 0.0 0.0 {z}" for z in z_coords)
    system = System(
        xyz=xyz,
        basis_set="sto-6g",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    ms_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ci = CI(
        states=[State(nel=6, multiplicity=5, ms=ms) for ms in ms_values],
        nroots=[1] * len(ms_values),
        active_orbitals=[0, 1, 2, 3, 4, 5],
        # Exact diagonalization, so the comparison isn't limited by Davidson convergence.
        ci_params=CIParams(ci_algorithm="exact"),
    )(rhf)
    ci.run()

    # differnt ms states are degenerate
    for ms, energy in zip(ms_values, ci.E):
        assert energy == approx(ci.E[0]), f"ms={ms} is not the same state as ms={-2.0}"

    for order in (2, 3):
        cumulants = [
            ci.make_cumulant(i, order=order, spin_type="sf") for i in range(len(ci.E))
        ]
        for ms, cumulant in zip(ms_values, cumulants):
            np.testing.assert_allclose(
                cumulant,
                cumulants[0],
                rtol=0.0,
                atol=1e-10,
                err_msg=f"sf {order}-cumulant differs between ms={ms} and ms={ms_values[0]}",
            )

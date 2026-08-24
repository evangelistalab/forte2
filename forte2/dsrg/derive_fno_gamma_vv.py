"""
Wick&d derivation of the virtual-virtual block of the unrelaxed second-order
1-RDM used to build FNOs for (Rel)DSRG-MRPT2, following the definition in

  C. Li, S. Mao, R. Huang, F. A. Evangelista, J. Chem. Theory Comput. 2024, 20, 4170-4181,
  eq 8-9 and Appendix A.

Physics
-------
Gamma^p_q(s) = <Psi(s)|E^p_q|Psi(s)>, expanded to second order in the DSRG
first-order cluster operator Ahat = Ahat1 + Ahat2 (antihermitian, Ahat_n = Tn - Tn^dagger):

    Gamma^(2)_pq = (1/2) <Phi0| [[E^p_q, Ahat], Ahat] |Phi0>

For p, q both virtual, the zeroth- and first-order terms vanish (no reference
occupation in the virtual space), so this double commutator is the whole story.
Lambda3 (the reference's 3-body density cumulant) is dropped, matching the paper's
approximation (their eq 8/A9 discussion) -- see the "truncated" block below.

Wick&d mechanics
-----------------
E^p_q is represented by a generic probe operator D = op("Gamma", ["v+ v"]) with an
as-yet-unspecified coefficient tensor "Gamma". Because D appears exactly once
(linearly) in the double commutator, every surviving term in the fully-contracted
(rank 0) result carries exactly one "Gamma" tensor factor -- whose own two indices
are, by construction, never summed against anything else. Reading those indices
off as the *output* indices (instead of treating "Gamma" as just another summed
tensor) directly gives the working equations for Gamma^e_f. This was verified
against the well-known single-reference MP2 result Gamma_ef = 1/2 sum_ijc t_ijec t_ijfc
before being applied to the full core/active/virtual multireference case below.

Run this file directly to print the generated einsum code.
"""

import wickd as w


def setup_spaces():
    w.reset_space()
    w.add_space("c", "fermion", "occupied", ["i", "j", "k", "l", "m", "n"])
    w.add_space("a", "fermion", "general", ["u", "v", "w", "x", "y", "z"])
    w.add_space("v", "fermion", "unoccupied", ["a", "b", "c", "d", "e", "f"])


def build_operators():
    # hole = core + active, particle = active + virtual, matching the T1/T2
    # block structure already used throughout rel_dsrg_common.py
    T1 = w.utils.gen_op("t", 1, "av", "ca", diagonal=False)
    T2 = w.utils.gen_op("t", 2, "av", "ca", diagonal=False)
    A1 = T1 - T1.adjoint()
    A2 = T2 - T2.adjoint()
    A = A1 + A2

    # probe operator standing in for the bare excitation E^p_q, restricted to the
    # virtual-virtual block since that's the only block FNO needs (paper, Sec. 2)
    D = w.op("Gamma", ["v+ v"])
    return A, D


def compile_probe_terms(expr, probe_label="Gamma", overall_scale=1.0):
    """
    Like wickd.utils.compile_einsum, but treats the tensor named `probe_label`
    as the *output* tensor (its own indices become the einsum output spec)
    instead of folding it into the summed RHS tensor list.

    expr must be a rank-0 (fully contracted scalar) Expression in which
    `probe_label` appears exactly once (linearly) per term -- true for our
    double-commutator construction, since D=op(probe_label,["v+ v"]) enters
    the commutator only once.
    """
    osi = w.osi()
    lines = []
    # rank-0 contraction -> single '|' (scalar) key
    for eq in expr.to_manybody_equations(probe_label)["|"]:
        unused_indices = osi.to_dict()
        index_dict = {}

        def get_indices(tensor):
            s = ""
            for i in list(tensor.upper()) + list(tensor.lower()):
                i = str(i)
                if i in index_dict:
                    s += index_dict[i]
                else:
                    idx = unused_indices[i[0]].pop(0)
                    index_dict[i] = idx
                    s += idx
            return s

        def block_of(tensor):
            return "".join(osi.label(i.space()) for i in tensor.upper()) + "".join(
                osi.label(i.space()) for i in tensor.lower()
            )

        tensors = eq.rhs().tensors()
        probe_tensor = next(t for t in tensors if t.label() == probe_label)
        other_tensors = [t for t in tensors if t.label() != probe_label]

        out_indices = get_indices(probe_tensor)
        out_block = block_of(probe_tensor)

        rhs_index_str = ",".join(get_indices(t) for t in other_tensors)
        rhs_label_str = "".join(
            f'{t.label()}["{block_of(t)}"], ' for t in other_tensors
        )

        coeff = overall_scale * float(eq.rhs_factor())
        lines.append(
            f'{probe_label}["{out_block}"] += {coeff:+.6f} * np.einsum('
            f'"{rhs_index_str}->{out_indices}", {rhs_label_str}optimize=True)'
        )
    return lines


T1_BLOCKS = {"ca", "cv", "av"}
T2_BLOCKS = {"caaa", "aaav", "ccaa", "caav", "aavv", "cavv", "ccvv", "ccav"}
CUMULANT_LABELS = {"gamma1", "eta1", "lambda2", "lambda3"}


def compile_probe_terms_forte2(expr, probe_label="Gamma", overall_scale=1.0):
    """
    Like compile_probe_terms, but resolves every "t"-labeled tensor against the
    T1/T2 blocks actually persisted by RelDSRG_MRPT2/_Slow (T1_BLOCKS, T2_BLOCKS).
    A tensor whose own block isn't stored must have its *reversed* (upper<->lower
    group swapped) block stored instead -- that's the adjoint (T1 dagger / T2
    dagger) contribution, which numerically is the stored array's complex
    conjugate with the upper/lower index groups swapped. This mirrors exactly how
    the historical wicked-derived H_T_C* code threads T1/T2 adjoints through by
    hand, except done mechanically here to avoid hand-transcription mistakes.
    """
    osi = w.osi()
    lines = []
    for eq in expr.to_manybody_equations(probe_label)["|"]:
        unused_indices = osi.to_dict()
        index_dict = {}

        def get_group_indices(indices):
            s = []
            for i in indices:
                i = str(i)
                if i in index_dict:
                    s.append(index_dict[i])
                else:
                    idx = unused_indices[i[0]].pop(0)
                    index_dict[i] = idx
                    s.append(idx)
            return s

        def space_str(indices):
            return "".join(osi.label(i.space()) for i in indices)

        tensors = eq.rhs().tensors()
        probe_tensor = next(t for t in tensors if t.label() == probe_label)
        other_tensors = [t for t in tensors if t.label() != probe_label]

        probe_idx = get_group_indices(
            list(probe_tensor.upper()) + list(probe_tensor.lower())
        )
        out_indices = "".join(probe_idx)
        out_block = space_str(probe_tensor.upper()) + space_str(probe_tensor.lower())

        rhs_index_parts = []
        rhs_label_parts = []
        for t in other_tensors:
            upper = list(t.upper())
            lower = list(t.lower())
            upper_idx = get_group_indices(upper)
            lower_idx = get_group_indices(lower)
            block = space_str(upper) + space_str(lower)

            if t.label() in CUMULANT_LABELS:
                rhs_label_parts.append(f'cumulants["{t.label()}"]')
                rhs_index_parts.append("".join(upper_idx + lower_idx))
                continue

            assert t.label() == "t", f"unexpected tensor label {t.label()}"
            rank = len(upper)
            dict_name = "T1" if rank == 1 else "T2"

            if block in (T1_BLOCKS if rank == 1 else T2_BLOCKS):
                rhs_label_parts.append(f'{dict_name}["{block}"]')
                rhs_index_parts.append("".join(upper_idx + lower_idx))
            else:
                rev_block = block[rank:] + block[:rank]
                assert rev_block in (
                    T1_BLOCKS if rank == 1 else T2_BLOCKS
                ), f"neither {block} nor its reverse {rev_block} is a stored block"
                rhs_label_parts.append(f'{dict_name}["{rev_block}"].conj()')
                # adjoint: stored array's own (hole,particle) axis order is
                # (this term's lower group, this term's upper group)
                rhs_index_parts.append("".join(lower_idx + upper_idx))

        rhs_index_str = ",".join(rhs_index_parts)
        rhs_label_str = ", ".join(rhs_label_parts)

        coeff = overall_scale * float(eq.rhs_factor())
        lines.append(
            f'{probe_label}["{out_block}"] += {coeff:+.6f} * np.einsum('
            f'"{rhs_index_str}->{out_indices}", {rhs_label_str}, optimize=True)'
        )
    return lines


def main():
    setup_spaces()
    A, D = build_operators()

    comm2 = w.commutator(w.commutator(D, A), A)

    # (1/2) from the BCH-like Taylor expansion <Phi0|e^{-A} E^p_q e^{A}|Phi0>
    #     = ... + (1/2)[[E^p_q,A],A] + ...
    overall_scale = 0.5

    print("=" * 88)
    print("Full result (includes the reference's 3-body density cumulant, lambda3)")
    print("=" * 88)
    wt_full = w.WickTheorem()
    expr_full = wt_full.contract(comm2, 0, 0).canonicalize()
    lines_full = compile_probe_terms(expr_full, overall_scale=overall_scale)
    print(f"# {len(lines_full)} terms\n")
    for line in lines_full:
        print(line)

    print()
    print("=" * 88)
    print("Truncated result (lambda3 dropped, matching the paper's FNO approximation)")
    print("=" * 88)
    wt_trunc = w.WickTheorem()
    wt_trunc.set_max_cumulant(2)
    expr_trunc = wt_trunc.contract(comm2, 0, 0).canonicalize()
    lines_trunc = compile_probe_terms(expr_trunc, overall_scale=overall_scale)
    print(f"# {len(lines_trunc)} terms\n")
    for line in lines_trunc:
        print(line)

    print()
    print("=" * 88)
    print("Truncated result, in RelDSRG_MRPT2/_Slow's actual T1/T2 block convention")
    print("=" * 88)
    lines_forte2 = compile_probe_terms_forte2(expr_trunc, overall_scale=overall_scale)
    print(f"# {len(lines_forte2)} terms\n")
    for line in lines_forte2:
        print(line)


if __name__ == "__main__":
    main()

#include <algorithm>

#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "rel_ci_sigma_builder.h"
#include "ci_sigma_builder.h"

namespace forte2 {

np_matrix_complex RelCISigmaBuilder::compute_so_1rdm(np_vector_complex C_left,
                                                     np_vector_complex C_right) const {
    const auto na = lists_.na();
    const auto norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({norb, norb});

    // skip building the RDM if there are no electrons or there are zero orbitals
    if (na < 1 || norb < 1)
        return rdm;

    const auto& alpha_address = lists_.alpha_address();
    const auto& beta_address = lists_.beta_address();
    auto rdm_view = rdm.view();
    auto Cl_span = vector::as_span<std::complex<double>>(C_left);
    auto Cr_span = vector::as_span<std::complex<double>>(C_right);

    // placeholder spin
    Spin spin = Spin::Alpha;

    // loop over blocks of the right state
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.block_size(nI) == 0)
            continue;

        auto tr = gather_block(Cr_span, TR, spin, lists_, class_Ia, class_Ib);

        // loop over blocks of the left state
        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                (is_beta(spin) and (class_Ia != class_Ja)))
                continue;
            if (lists_.block_size(nJ) == 0)
                continue;

            auto tl = gather_block(Cl_span, TL, spin, lists_, class_Ja, class_Jb);

            const size_t maxL =
                is_alpha(spin) ? beta_address->strpcls(class_Ib) : alpha_address->strpcls(class_Ia);

            const auto& pq_vo_list = is_alpha(spin) ? lists_.get_alpha_vo_list(class_Ia, class_Ja)
                                                    : lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                std::complex<double> rdm_element = 0.0;
                for (const auto& [sign, I, J] : vo_list) {
                    // Compute the RDM element contribution
                    for (size_t idx{0}; idx != maxL; ++idx) {
                        rdm_element += sign * std::conj(tl[J * maxL + idx]) * tr[I * maxL + idx];
                    }
                }
                rdm_view(p, q) += rdm_element;
            }
        }
    }
    return rdm;
}

template <typename GetHoleList, typename OrbitalIndex>
void RelCISigmaBuilder::contract_spinor_rdm(np_vector_complex C_left, np_vector_complex C_right,
                                            std::complex<double>* rdm_data,
                                            const CIOrbitalIrrepBlocks& orbitals,
                                            const ci_flat::StringAddress& holes,
                                            GetHoleList get_hole_list,
                                            OrbitalIndex orbital_index) const {
    if (lists_.ndet() == 0)
        return;

    const auto Cl_span = vector::as_span<std::complex<double>>(C_left);
    const auto Cr_span = vector::as_span<std::complex<double>>(C_right);
    CIIrrepRDM rdm_blocks(orbitals, rdm_data);

    for (int class_K = 0; class_K < holes.nclasses(); ++class_K) {
        const size_t maxK = holes.strpcls(class_K);
        // The beta string is the vacuum, so each column is just one alpha hole string.
        const auto& block = orbitals.for_intermediate(lists_, class_K, 0);
        const size_t nrows = block.size();
        if (maxK == 0 || nrows == 0)
            continue;

        auto [Kblock_left, Kblock_right, K_chunk_size] = get_Kblock_spans(nrows, maxK);
        for (size_t K_start = 0; K_start < maxK;) {
            const size_t K_size = std::min(K_chunk_size, maxK - K_start);
            std::fill_n(Kblock_left.begin(), nrows * K_size, 0.0);
            std::fill_n(Kblock_right.begin(), nrows * K_size, 0.0);

            // Gather all determinant classes into the same K-blocks to include transitions
            // between GAS occupation patterns in one GEMM per chunk.
            for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                if (lists_.block_size(nI) == 0)
                    continue;
                const size_t offset = lists_.block_offset(nI);
                for (size_t K = 0; K < K_size; ++K) {
                    for (const auto& sub :
                         (lists_.*get_hole_list)(class_K, K_start + K, class_Ia)) {
                        const size_t row = orbitals.local_row(orbital_index(sub));
                        const double sign = static_cast<double>(sub.sign);
                        const size_t det_index = offset + sub.J;
                        Kblock_left[row * K_size + K] += sign * std::conj(Cl_span[det_index]);
                        Kblock_right[K * nrows + row] += sign * Cr_span[det_index];
                    }
                }
            }

            // The left K-block stores conjugated bra rows by hole columns.
            // The right K-block stores hole rows by ket columns.
            matrix_product('N', 'N', nrows, nrows, K_size, 1.0, Kblock_left.data(), K_size,
                           Kblock_right.data(), nrows, 1.0, rdm_blocks.data(block), nrows);
            K_start += K_size;
        }
    }
    rdm_blocks.unpack();
}

np_tensor4_complex RelCISigmaBuilder::compute_so_2rdm(np_vector_complex C_left,
                                                      np_vector_complex C_right) const {
    const size_t norb = lists_.norb();
    if (norb < 2 || lists_.na() < 2)
        return make_zeros<nb::numpy, std::complex<double>, 4>({norb, norb, norb, norb});

    const size_t npairs = norb * (norb - 1) / 2;
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({npairs, npairs});
    contract_spinor_rdm(
        C_left, C_right, rdm.data(), spinor_pairs_, *lists_.alpha_address_2h(),
        &ci_flat::CIStrings::get_alpha_2h_list,
        [](const ci_flat::H2StringSubstitution& sub) { return pair_index_gt(sub.p, sub.q); });
    return matrix::packed_tensor4_to_tensor4<std::complex<double>>(rdm);
}

np_tensor6_complex RelCISigmaBuilder::compute_so_3rdm(np_vector_complex C_left,
                                                      np_vector_complex C_right) const {
    const size_t norb = lists_.norb();
    if (norb < 3 || lists_.na() < 3)
        return make_zeros<nb::numpy, std::complex<double>, 6>({norb, norb, norb, norb, norb, norb});

    const size_t ntriplets = norb * (norb - 1) * (norb - 2) / 6;
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({ntriplets, ntriplets});
    const auto triplets = CIOrbitalIrrepBlocks::spinor_triplets(lists_);
    contract_spinor_rdm(C_left, C_right, rdm.data(), triplets, *lists_.alpha_address_3h(),
                        &ci_flat::CIStrings::get_alpha_3h_list,
                        [](const ci_flat::H3StringSubstitution& sub) {
                            return triplet_index_gt(sub.p, sub.q, sub.r);
                        });

    auto rdm_full =
        make_zeros<nb::numpy, std::complex<double>, 6>({norb, norb, norb, norb, norb, norb});

    // Expand the unique packed 3-RDM gamma[p>q>r, s>t>u] into the full antisymmetric tensor.
    // For a fixed bra triplet the entire (s,t,u) sub-block is identical up to the bra-permutation
    // sign, so we assemble that norb^3 "ket block" once in a small buffer and then
    // write each of the 6 bra permutations as a signed copy.
    const auto* rdm_data_packed = rdm.data();
    auto* full_data = rdm_full.data();
    const size_t N3 = norb * norb * norb;

    // Zero-initialized once: positions with non-distinct (s,t,u) are never written and stay zero
    // across all bra iterations; the distinct positions are fully overwritten each iteration.
    std::vector<std::complex<double>> ketrow(N3, 0.0);

    auto write_block = [&](bool positive, size_t a, size_t b, size_t c) {
        auto* dst = full_data + ((a * norb + b) * norb + c) * N3;
        if (positive) {
            for (size_t i = 0; i < N3; ++i)
                dst[i] = ketrow[i];
        } else {
            for (size_t i = 0; i < N3; ++i)
                dst[i] = -ketrow[i];
        }
    };

    for (size_t p{2}, pqr{0}; p < norb; ++p) {
        for (size_t q{1}; q < p; ++q) {
            for (size_t r{0}; r < q; ++r, ++pqr) {
                const auto* row = rdm_data_packed + pqr * ntriplets;
                // Scatter the ket triplets into the (d,e,f) block with antisymmetric signs.
                for (size_t s{2}, stu{0}; s < norb; ++s) {
                    for (size_t t{1}; t < s; ++t) {
                        for (size_t u{0}; u < t; ++u, ++stu) {
                            const auto el = row[stu];
                            ketrow[(s * norb + t) * norb + u] = +el;
                            ketrow[(s * norb + u) * norb + t] = -el;
                            ketrow[(u * norb + s) * norb + t] = +el;
                            ketrow[(u * norb + t) * norb + s] = -el;
                            ketrow[(t * norb + u) * norb + s] = +el;
                            ketrow[(t * norb + s) * norb + u] = -el;
                        }
                    }
                }
                // Write the 6 bra permutations as contiguous streaming signed copies.
                write_block(true, p, q, r);
                write_block(false, p, r, q);
                write_block(true, r, p, q);
                write_block(false, r, q, p);
                write_block(true, q, r, p);
                write_block(false, q, p, r);
            }
        }
    }

    return rdm_full;
}

} // namespace forte2

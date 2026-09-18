#include <iostream>

#include "helpers/timer.hpp"
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

np_tensor4_complex RelCISigmaBuilder::compute_so_2rdm(np_vector_complex C_left,
                                                      np_vector_complex C_right) const {
    Spin spin = Spin::Alpha; // placeholder spin

    const auto na = lists_.na();
    const size_t norb = lists_.norb();

    // if there are less than two orbitals or two electrons, return an empty matrix
    if (norb < 2 || na < 2) {
        return make_zeros<nb::numpy, std::complex<double>, 4>({norb, norb, norb, norb});
    }

    // calculate the number of pairs of orbitals p > q
    const size_t npairs = (norb * (norb - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({npairs, npairs});

    auto Cl_span = vector::as_span<std::complex<double>>(C_left);
    auto Cr_span = vector::as_span<std::complex<double>>(C_right);

    auto rdm_data = rdm.data();

    const auto& alpha_address = lists_.alpha_address();
    const auto& beta_address = lists_.beta_address();

    int num_2h_classes = is_alpha(spin) ? lists_.alpha_address_2h()->nclasses()
                                        : lists_.beta_address_2h()->nclasses();

    // The contraction gamma[pq,rs] = sum_K sum_idx sign_K sign_L conj(C_L[I,idx]) C_R[J,idx]
    // can be reformulated into a matrix product over the composite index c = (K, idx)
    // We gather the (conjugated/signed) left coefficients into B_L[pq, c] and the
    // (signed) right coefficients into the transposed B_R[c, rs], then accumulate
    // gamma += B_L * B_R via a single zgemm per K-chunk.
    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        const size_t maxK = is_alpha(spin) ? lists_.alpha_address_2h()->strpcls(class_K)
                                           : lists_.beta_address_2h()->strpcls(class_K);
        if (maxK == 0)
            continue;

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {

            if (lists_.block_size(nI) == 0)
                continue;

            const size_t maxL =
                is_alpha(spin) ? beta_address->strpcls(class_Ib) : alpha_address->strpcls(class_Ia);
            if (maxL == 0)
                continue;

            auto tl = gather_block(Cl_span, TL, spin, lists_, class_Ia, class_Ib);

            // B_L is npairs x dimKL, B_R^T is dimKL x npairs; both fit in a Kblock buffer.
            auto [Kblock1, Kblock2, K_chunk_size] = get_Kblock_spans(npairs * maxL, maxK);

            for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                    (is_beta(spin) and (class_Ia != class_Ja)))
                    continue;
                if (lists_.block_size(nJ) == 0)
                    continue;

                auto tr = gather_block(Cr_span, TR, spin, lists_, class_Ja, class_Jb);

                for (size_t K_start = 0; K_start < maxK; K_start += K_chunk_size) {
                    const size_t K_end = std::min(K_start + K_chunk_size, maxK);
                    const size_t K_size = K_end - K_start;
                    const size_t dimKL = K_size * maxL;

                    // B_L[pq, c] with c = K*maxL + idx, leading dim dimKL
                    std::fill_n(Kblock1.begin(), npairs * dimKL, 0.0);
                    // B_R^T[c, rs], leading dim npairs
                    std::fill_n(Kblock2.begin(), dimKL * npairs, 0.0);

                    for (size_t K = 0; K < K_size; ++K) {
                        const auto& Kllist =
                            is_alpha(spin)
                                ? lists_.get_alpha_2h_list(class_K, K_start + K, class_Ia)
                                : lists_.get_beta_2h_list(class_K, K_start + K, class_Ib);
                        const auto& Krlist =
                            is_alpha(spin)
                                ? lists_.get_alpha_2h_list(class_K, K_start + K, class_Ja)
                                : lists_.get_beta_2h_list(class_K, K_start + K, class_Jb);
                        for (const auto& [sign_K, p, q, I] : Kllist) {
                            const size_t pq_index = pair_index_gt(p, q);
                            const double sgn = static_cast<double>(sign_K);
                            const auto* src = tl.data() + I * maxL;
                            auto* dst = Kblock1.data() + pq_index * dimKL + K * maxL;
                            for (size_t idx{0}; idx < maxL; ++idx)
                                dst[idx] += sgn * std::conj(src[idx]);
                        }
                        for (const auto& [sign_L, r, s, J] : Krlist) {
                            const size_t rs_index = pair_index_gt(r, s);
                            const double sgn = static_cast<double>(sign_L);
                            const auto* src = tr.data() + J * maxL;
                            auto* dst = Kblock2.data() + (K * maxL) * npairs + rs_index;
                            for (size_t idx{0}; idx < maxL; ++idx)
                                dst[idx * npairs] += sgn * src[idx];
                        }
                    }

                    // gamma[pq,rs] += sum_c B_L[pq,c] B_R^T[c,rs]
                    matrix_product('N', 'N', npairs, npairs, dimKL, 1.0, Kblock1.data(), dimKL,
                                   Kblock2.data(), npairs, 1.0, rdm_data, npairs);
                }
            }
        }
    }

    return matrix::packed_tensor4_to_tensor4<std::complex<double>>(rdm);
}

np_tensor6_complex RelCISigmaBuilder::compute_so_3rdm(np_vector_complex C_left,
                                                      np_vector_complex C_right) const {
    Spin spin = Spin::Alpha; // placeholder spin
    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const auto norb = lists_.norb();

    // if there are less than three orbitals or 3 electrons, return an empty matrix
    if (norb < 3 || na < 3) {
        return make_zeros<nb::numpy, std::complex<double>, 6>({norb, norb, norb, norb, norb, norb});
    }

    const size_t ntriplets = (norb * (norb - 1) * (norb - 2)) / 6;
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({ntriplets, ntriplets});

    auto Cl_span = vector::as_span<std::complex<double>>(C_left);
    auto Cr_span = vector::as_span<std::complex<double>>(C_right);

    auto rdm_data = rdm.data();
    const auto& alpha_address = lists_.alpha_address();
    const auto& beta_address = lists_.beta_address();

    int num_3h_classes = is_alpha(spin) ? lists_.alpha_address_3h()->nclasses()
                                        : lists_.beta_address_3h()->nclasses();

    // Same GEMM reformulation as compute_2rdm, over triplets p>q>r / s>t>u and the 3-hole K
    // classes: gamma[pqr,stu] += sum_c B_L[pqr,c] B_R^T[c,stu] with c = (K, idx).
    for (int class_K = 0; class_K < num_3h_classes; ++class_K) {
        const size_t maxK = is_alpha(spin) ? lists_.alpha_address_3h()->strpcls(class_K)
                                           : lists_.beta_address_3h()->strpcls(class_K);
        if (maxK == 0)
            continue;

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.block_size(nI) == 0)
                continue;

            const size_t maxL =
                is_alpha(spin) ? beta_address->strpcls(class_Ib) : alpha_address->strpcls(class_Ia);
            if (maxL == 0)
                continue;

            auto tl = gather_block(Cl_span, TL, spin, lists_, class_Ia, class_Ib);

            // B_L is ntriplets x dimKL, B_R^T is dimKL x ntriplets.
            auto [Kblock1, Kblock2, K_chunk_size] = get_Kblock_spans(ntriplets * maxL, maxK);

            for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                    (is_beta(spin) and (class_Ia != class_Ja)))
                    continue;
                if (lists_.block_size(nJ) == 0)
                    continue;

                auto tr = gather_block(Cr_span, TR, spin, lists_, class_Ja, class_Jb);

                for (size_t K_start = 0; K_start < maxK; K_start += K_chunk_size) {
                    const size_t K_end = std::min(K_start + K_chunk_size, maxK);
                    const size_t K_size = K_end - K_start;
                    const size_t dimKL = K_size * maxL;

                    std::fill_n(Kblock1.begin(), ntriplets * dimKL, 0.0);
                    std::fill_n(Kblock2.begin(), dimKL * ntriplets, 0.0);

                    for (size_t K = 0; K < K_size; ++K) {
                        const auto& Kllist =
                            is_alpha(spin)
                                ? lists_.get_alpha_3h_list(class_K, K_start + K, class_Ia)
                                : lists_.get_beta_3h_list(class_K, K_start + K, class_Ib);
                        const auto& Krlist =
                            is_alpha(spin)
                                ? lists_.get_alpha_3h_list(class_K, K_start + K, class_Ja)
                                : lists_.get_beta_3h_list(class_K, K_start + K, class_Jb);
                        for (const auto& [sign_K, p, q, r, I] : Kllist) {
                            const size_t pqr_index = triplet_index_gt(p, q, r);
                            const double sgn = static_cast<double>(sign_K);
                            const auto* src = tl.data() + I * maxL;
                            auto* dst = Kblock1.data() + pqr_index * dimKL + K * maxL;
                            for (size_t idx{0}; idx < maxL; ++idx)
                                dst[idx] += sgn * std::conj(src[idx]);
                        }
                        for (const auto& [sign_L, s, t, u, J] : Krlist) {
                            const size_t stu_index = triplet_index_gt(s, t, u);
                            const double sgn = static_cast<double>(sign_L);
                            const auto* src = tr.data() + J * maxL;
                            auto* dst = Kblock2.data() + (K * maxL) * ntriplets + stu_index;
                            for (size_t idx{0}; idx < maxL; ++idx)
                                dst[idx * ntriplets] += sgn * src[idx];
                        }
                    }

                    matrix_product('N', 'N', ntriplets, ntriplets, dimKL, 1.0, Kblock1.data(),
                                   dimKL, Kblock2.data(), ntriplets, 1.0, rdm_data, ntriplets);
                }
            }
        }
    }

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

#include <algorithm>

#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "rel_ci_sigma_builder.h"
#include "ci_hole_contractions.h"
#include "ci_one_rdm.h"

namespace forte2 {

np_matrix_complex RelCISigmaBuilder::compute_so_1rdm(np_vector_complex C_left,
                                                     np_vector_complex C_right) const {
    const auto na = lists_.na();
    const auto norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({norb, norb});

    // skip building the RDM if there are no electrons or there are zero orbitals
    if (na < 1 || norb < 1)
        return rdm;

    accumulate_one_rdm<std::complex<double>>(
        lists_, Spin::Alpha, vector::as_span<std::complex<double>>(C_left),
        vector::as_span<std::complex<double>>(C_right), {}, {}, rdm.view());
    return rdm;
}

template <size_t Rank>
void RelCISigmaBuilder::contract_spinor_rdm(np_vector_complex C_left, np_vector_complex C_right,
                                            std::complex<double>* rdm_data,
                                            const CIOrbitalIrrepBlocks& orbitals) const {
    using Scalar = std::complex<double>;
    const auto left = vector::as_span<Scalar>(C_left);
    const auto right = vector::as_span<Scalar>(C_right);
    const auto& holes = lists_.strings(Spin::Alpha, Rank);
    const auto connections = same_spin_hole_connections<Rank>(lists_, Spin::Alpha);
    if (connections.empty())
        return;

    CIIrrepRDM rdm_blocks(orbitals, rdm_data);
    std::vector<LocalStringAddress> work(lists_.ngas_spaces());
    for (const auto& [classes, links] : connections) {
        const auto [hole_class, spectator] = classes;
        const auto count = holes.size(hole_class);
        const auto& block = orbitals.for_intermediate(lists_, hole_class, spectator);
        const auto nrows = block.size();
        const auto [L, R, chunk] = get_Kblock_spans(nrows, count);
        for (size_t start = 0; start < count;) {
            const size_t width = std::min(chunk, count - start);
            std::fill_n(L.begin(), nrows * width, Scalar{});
            std::fill_n(R.begin(), nrows * width, Scalar{});
            for (const auto& [determinant, substitutions] : links) {
                for (size_t k = 0; k < width; ++k) {
                    substitutions.for_each_hole_substitution(
                        start + k, work, [&](const auto& tuple, int phase, size_t rank) {
                            const auto row = orbitals.local_row(hole_orbital_index(tuple));
                            const auto index = determinant.offset + rank;
                            const double sign = phase;
                            L[row * width + k] = sign * std::conj(left[index]);
                            R[k * nrows + row] = sign * right[index];
                        });
                }
            }
            matrix_product('N', 'N', nrows, nrows, width, 1.0, L.data(), width, R.data(), nrows,
                           1.0, rdm_blocks.data(block), nrows);
            start += width;
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
    contract_spinor_rdm<2>(C_left, C_right, rdm.data(), spinor_pairs_);
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
    contract_spinor_rdm<3>(C_left, C_right, rdm.data(), triplets);

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

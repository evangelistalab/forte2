#include "ci_sigma_builder.h"
#include "ci_hole_contractions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/timer.hpp"

namespace forte2 {

template <size_t AlphaRank, size_t BetaRank>
void CISigmaBuilder::contract_mixed_spin_rdm(np_vector C_left, np_vector C_right, double* rdm_data,
                                             const CIOrbitalIrrepBlocks& orbitals) const {
    const auto connections = mixed_hole_connections<AlphaRank, BetaRank>(lists_);
    if (connections.empty())
        return;

    const auto norb = lists_.norb();
    const auto& alpha = lists_.strings(Spin::Alpha, AlphaRank);
    const auto& beta = lists_.strings(Spin::Beta, BetaRank);
    const auto orbital_index = [norb](auto a, auto b) {
        if constexpr (BetaRank == 2)
            return a[0] * (norb * (norb - 1) / 2) + hole_orbital_index(b);
        else
            return hole_orbital_index(a) * norb + b[0];
    };

    size_t max_rows = 0, max_columns = 0;
    for (const auto& [classes, links] : connections) {
        const auto [ka, kb] = classes;
        max_rows = std::max(max_rows, orbitals.for_intermediate(lists_, ka, kb).size());
        max_columns = std::max(max_columns, checked_ci_product(alpha.size(ka), beta.size(kb)));
    }
    std::vector<double> L, R;
    const size_t chunk = acquire_local_Kblock_buffers(L, R, max_rows, max_columns);
    const auto left = vector::as_span<double>(C_left);
    const auto right = vector::as_span<double>(C_right);
    CIIrrepRDM rdm_blocks(orbitals, rdm_data);
    std::vector<LocalStringAddress> work(lists_.ngas_spaces());

    for (const auto& [classes, links] : connections) {
        const auto [ka, kb] = classes;
        const auto& block = orbitals.for_intermediate(lists_, ka, kb);
        const auto nrows = block.size();
        const auto beta_size = beta.size(kb);
        const auto columns = alpha.size(ka) * beta_size;
        for (size_t start = 0; start < columns;) {
            const size_t width = std::min(chunk, columns - start);
            std::fill_n(L.begin(), nrows * width, 0.0);
            std::fill_n(R.begin(), nrows * width, 0.0);
            for (const auto& [determinant, a, b] : links) {
                for_each_mixed_hole_substitution(
                    a, b, beta_size, start, width, work,
                    [&](size_t col, auto p, auto q, int sign, size_t i, size_t j) {
                        const auto row = orbitals.local_row(orbital_index(p, q));
                        const auto index = determinant.offset + i * determinant.beta_size + j;
                        L[row * width + col] = sign * left[index];
                        R[row * width + col] = sign * right[index];
                    });
            }
            matrix_product('N', 'T', nrows, nrows, width, 1.0, L.data(), width, R.data(), width,
                           1.0, rdm_blocks.data(block), nrows);
            start += width;
        }
    }
    rdm_blocks.unpack();
}

np_tensor4 CISigmaBuilder::compute_ab_2rdm(np_vector C_left, np_vector C_right) const {
    local_timer timer;
    const size_t norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});
    contract_mixed_spin_rdm<1, 1>(C_left, C_right, rdm.data(), opposite_spin_pairs_);
    rdm2_ab_timer_ += timer.elapsed_seconds();
    return rdm;
}

np_tensor4 CISigmaBuilder::compute_aab_3rdm(np_vector C_left, np_vector C_right) const {
    const size_t norb = lists_.norb();
    if (norb < 2)
        return make_zeros<nb::numpy, double, 4>({0, 0, 0, 0});

    const size_t npair = norb * (norb - 1) / 2;
    auto rdm = make_zeros<nb::numpy, double, 4>({npair, norb, npair, norb});
    if (lists_.na() >= 2 && lists_.nb() > 0) {
        const auto triplets = CIOrbitalIrrepBlocks::aab_triplets(lists_);
        contract_mixed_spin_rdm<2, 1>(C_left, C_right, rdm.data(), triplets);
    }
    return rdm;
}

np_tensor4 CISigmaBuilder::compute_abb_3rdm(np_vector C_left, np_vector C_right) const {
    const size_t norb = lists_.norb();
    if (norb < 2)
        return make_zeros<nb::numpy, double, 4>({0, 0, 0, 0});

    const size_t npair = norb * (norb - 1) / 2;
    auto rdm = make_zeros<nb::numpy, double, 4>({norb, npair, norb, npair});
    if (lists_.na() > 0 && lists_.nb() >= 2) {
        const auto triplets = CIOrbitalIrrepBlocks::abb_triplets(lists_);
        contract_mixed_spin_rdm<1, 2>(C_left, C_right, rdm.data(), triplets);
    }
    return rdm;
}

} // namespace forte2

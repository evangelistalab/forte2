#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "helpers/blas.h"
#include "helpers/indexing.hpp"
#include "helpers/np_vector_functions.h"
#include "helpers/timer.hpp"

#include "ci_sigma_builder.h"

namespace forte2 {

namespace {

size_t max_composite_hole_dimension(const ci_flat::StringAddress& alpha_address,
                                    const ci_flat::StringAddress& beta_address,
                                    std::string_view rdm_name) {
    size_t max_alpha = 0;
    for (int class_Ka = 0; class_Ka < alpha_address.nclasses(); ++class_Ka) {
        max_alpha = std::max(max_alpha, alpha_address.strpcls(class_Ka));
    }

    size_t max_beta = 0;
    for (int class_Kb = 0; class_Kb < beta_address.nclasses(); ++class_Kb) {
        max_beta = std::max(max_beta, beta_address.strpcls(class_Kb));
    }

    if ((max_alpha == 0) or (max_beta == 0))
        return 0;
    if (max_alpha > std::numeric_limits<size_t>::max() / max_beta) {
        throw std::overflow_error("The composite " + std::string(rdm_name) +
                                  " hole dimension is too large.");
    }
    return max_alpha * max_beta;
}

/// Two CI vectors share the same signed substitution into the left and right K-blocks.
struct RDMKBlocks {
    std::span<const double> left_coefficients;
    std::span<const double> right_coefficients;
    std::span<double> Kblock_left;
    std::span<double> Kblock_right;

    void set(size_t Kblock_index, size_t coefficient_index, double sign) const {
        Kblock_left[Kblock_index] = sign * left_coefficients[coefficient_index];
        Kblock_right[Kblock_index] = sign * right_coefficients[coefficient_index];
    }
};

/// Gather signed alpha/beta one-hole coefficients into a composite-hole K-block.
void gather_ab_2rdm_block(const ci_flat::CIStrings& lists, int class_Ka, int class_Kb, size_t maxKb,
                          size_t Kblock_start, size_t Kdim, const CIOrbitalIrrepBlocks& pairs,
                          const RDMKBlocks& Kblocks) {
    const size_t norb = lists.norb();
    for (const auto& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        if (lists.block_size(nI) == 0)
            continue;

        const auto maxIb = lists.beta_address()->strpcls(class_Ib);
        const auto coefficient_offset = lists.block_offset(nI);
        const auto& Ka_list = lists.get_alpha_1h_list2(class_Ka, class_Ia);
        const auto& Kb_list = lists.get_beta_1h_list2(class_Kb, class_Ib);
        if (Ka_list.empty() || Kb_list.empty())
            continue;

        for (size_t Kidx = 0; Kidx < Kdim; ++Kidx) {
            const size_t K = Kblock_start + Kidx;
            const size_t Ka = K / maxKb;
            const size_t Kb = K % maxKb;
            for (const auto& [sign_p, p, Ia] : Ka_list[Ka]) {
                const size_t pnorb = p * norb;
                const auto coefficient_Ia_offset = coefficient_offset + Ia * maxIb;
                for (const auto& [sign_q, q, Ib] : Kb_list[Kb]) {
                    Kblocks.set(pairs.local_row(pnorb + q) * Kdim + Kidx,
                                coefficient_Ia_offset + Ib, sign_p * sign_q);
                }
            }
        }
    }
}

/// Gather signed two-alpha-hole/one-beta-hole coefficients into an AAB 3-RDM K-block.
void gather_aab_3rdm_block(const ci_flat::CIStrings& lists, int class_Ka, int class_Kb,
                           size_t maxKb, size_t Kblock_start, size_t Kdim,
                           const CIOrbitalIrrepBlocks& triplets, const RDMKBlocks& Kblocks) {
    const size_t norb = lists.norb();
    for (const auto& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        if (lists.block_size(nI) == 0)
            continue;

        const auto maxIb = lists.beta_address()->strpcls(class_Ib);
        const auto coefficient_offset = lists.block_offset(nI);
        const auto& Kb_list = lists.get_beta_1h_list2(class_Kb, class_Ib);
        if (Kb_list.empty())
            continue;

        for (size_t Kidx = 0; Kidx < Kdim; ++Kidx) {
            const size_t K = Kblock_start + Kidx;
            const size_t Ka = K / maxKb;
            const size_t Kb = K % maxKb;
            const auto& Ka_list = lists.get_alpha_2h_list(class_Ka, Ka, class_Ia);
            if (Ka_list.empty())
                continue;

            for (const auto& [sign_pq, p, q, Ia] : Ka_list) {
                const size_t row = pair_index_gt<size_t>(p, q) * norb;
                const auto coefficient_Ia_offset = coefficient_offset + Ia * maxIb;
                for (const auto& [sign_r, r, Ib] : Kb_list[Kb]) {
                    Kblocks.set(triplets.local_row(row + r) * Kdim + Kidx,
                                coefficient_Ia_offset + Ib, sign_pq * sign_r);
                }
            }
        }
    }
}

/// Gather signed one-alpha-hole/two-beta-hole coefficients into an ABB 3-RDM K-block.
void gather_abb_3rdm_block(const ci_flat::CIStrings& lists, int class_Ka, int class_Kb,
                           size_t maxKb, size_t Kblock_start, size_t Kdim,
                           const CIOrbitalIrrepBlocks& triplets, const RDMKBlocks& Kblocks) {
    const size_t norb = lists.norb();
    const size_t npair = norb * (norb - 1) / 2;
    for (const auto& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        if (lists.block_size(nI) == 0)
            continue;

        const auto maxIb = lists.beta_address()->strpcls(class_Ib);
        const auto coefficient_offset = lists.block_offset(nI);
        const auto& Ka_list = lists.get_alpha_1h_list2(class_Ka, class_Ia);
        if (Ka_list.empty())
            continue;

        for (size_t Kidx = 0; Kidx < Kdim; ++Kidx) {
            const size_t K = Kblock_start + Kidx;
            const size_t Ka = K / maxKb;
            const size_t Kb = K % maxKb;
            const auto& Kb_list = lists.get_beta_2h_list(class_Kb, Kb, class_Ib);
            if (Kb_list.empty())
                continue;

            for (const auto& [sign_p, p, Ia] : Ka_list[Ka]) {
                const size_t row = p * npair;
                const auto coefficient_Ia_offset = coefficient_offset + Ia * maxIb;
                for (const auto& [sign_qr, q, r, Ib] : Kb_list) {
                    const size_t qr_index = pair_index_gt<size_t>(q, r);
                    Kblocks.set(triplets.local_row(row + qr_index) * Kdim + Kidx,
                                coefficient_Ia_offset + Ib, sign_p * sign_qr);
                }
            }
        }
    }
}

} // namespace

template <typename Gather>
void CISigmaBuilder::contract_mixed_spin_rdm(np_vector C_left, np_vector C_right, double* rdm_data,
                                             const CIOrbitalIrrepBlocks& orbitals,
                                             const ci_flat::StringAddress& alpha_holes,
                                             const ci_flat::StringAddress& beta_holes,
                                             std::string_view rdm_name, Gather gather) const {
    const auto& flat_lists = lists_.flat_strings();
    const size_t max_composite_K = max_composite_hole_dimension(alpha_holes, beta_holes, rdm_name);
    if (max_composite_K == 0)
        return;

    std::vector<double> Kblock_left;
    std::vector<double> Kblock_right;
    const size_t Kblock_size = acquire_local_Kblock_buffers(Kblock_left, Kblock_right,
                                                            orbitals.max_rows(), max_composite_K);
    const RDMKBlocks Kblocks{vector::as_span<double>(C_left), vector::as_span<double>(C_right),
                             Kblock_left, Kblock_right};
    CIIrrepRDM rdm_blocks(orbitals, rdm_data);

    for (int class_Ka = 0; class_Ka < alpha_holes.nclasses(); ++class_Ka) {
        const size_t maxKa = alpha_holes.strpcls(class_Ka);
        if (maxKa == 0)
            continue;
        for (int class_Kb = 0; class_Kb < beta_holes.nclasses(); ++class_Kb) {
            const size_t maxKb = beta_holes.strpcls(class_Kb);
            const auto& block = orbitals.for_intermediate(lists_, class_Ka, class_Kb);
            const size_t nrows = block.size();
            if (maxKb == 0 || nrows == 0)
                continue;

            // Both K-blocks use the same flattened hole index K = Ka * maxKb + Kb.
            const size_t maxK = maxKa * maxKb;
            for (size_t Kblock_start = 0; Kblock_start < maxK;) {
                const size_t Kdim = std::min(Kblock_size, maxK - Kblock_start);
                std::fill_n(Kblock_left.begin(), nrows * Kdim, 0.0);
                std::fill_n(Kblock_right.begin(), nrows * Kdim, 0.0);
                gather(flat_lists, class_Ka, class_Kb, maxKb, Kblock_start, Kdim, orbitals,
                       Kblocks);
                matrix_product('N', 'T', nrows, nrows, Kdim, 1.0, Kblock_left.data(), Kdim,
                               Kblock_right.data(), Kdim, 1.0, rdm_blocks.data(block), nrows);
                Kblock_start += Kdim;
            }
        }
    }
    rdm_blocks.unpack();
}

np_tensor4 CISigmaBuilder::compute_ab_2rdm(np_vector C_left, np_vector C_right) const {
    local_timer timer;
    const size_t norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});
    if (norb == 0 || lists_.na() == 0 || lists_.nb() == 0)
        return rdm;

    const auto& flat_lists = lists_.flat_strings();
    contract_mixed_spin_rdm(C_left, C_right, rdm.data(), opposite_spin_pairs_,
                            *flat_lists.alpha_address_1h(), *flat_lists.beta_address_1h(), "2-RDM",
                            gather_ab_2rdm_block);
    rdm2_ab_timer_ += timer.elapsed_seconds();
    return rdm;
}

np_tensor4 CISigmaBuilder::compute_aab_3rdm(np_vector C_left, np_vector C_right) const {
    const size_t norb = lists_.norb();
    if (norb < 2)
        return make_zeros<nb::numpy, double, 4>({0, 0, 0, 0});

    const size_t npair = norb * (norb - 1) / 2;
    auto rdm = make_zeros<nb::numpy, double, 4>({npair, norb, npair, norb});
    if (lists_.na() < 2 || lists_.nb() == 0)
        return rdm;

    const auto triplets = CIOrbitalIrrepBlocks::aab_triplets(lists_);
    const auto& flat_lists = lists_.flat_strings();
    contract_mixed_spin_rdm(C_left, C_right, rdm.data(), triplets, *flat_lists.alpha_address_2h(),
                            *flat_lists.beta_address_1h(), "AAB 3-RDM", gather_aab_3rdm_block);
    return rdm;
}

np_tensor4 CISigmaBuilder::compute_abb_3rdm(np_vector C_left, np_vector C_right) const {
    const size_t norb = lists_.norb();
    if (norb < 2)
        return make_zeros<nb::numpy, double, 4>({0, 0, 0, 0});

    const size_t npair = norb * (norb - 1) / 2;
    auto rdm = make_zeros<nb::numpy, double, 4>({norb, npair, norb, npair});
    if (lists_.na() == 0 || lists_.nb() < 2)
        return rdm;

    const auto triplets = CIOrbitalIrrepBlocks::abb_triplets(lists_);
    const auto& flat_lists = lists_.flat_strings();
    contract_mixed_spin_rdm(C_left, C_right, rdm.data(), triplets, *flat_lists.alpha_address_1h(),
                            *flat_lists.beta_address_2h(), "ABB 3-RDM", gather_abb_3rdm_block);
    return rdm;
}

} // namespace forte2

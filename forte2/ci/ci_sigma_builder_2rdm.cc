#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

namespace forte2 {

namespace {

/// Gather signed alpha/beta one-hole coefficients into a composite-hole K-block.
void gather_ab_2rdm_block(const CIStrings& lists, int class_Ka, int class_Kb, size_t maxKb,
                          size_t Kblock_start, size_t Kdim, size_t norb,
                          std::span<const double> coefficients, std::span<double> Kblock) {
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
                    Kblock[(pnorb + q) * Kdim + Kidx] =
                        sign_p * sign_q * coefficients[coefficient_Ia_offset + Ib];
                }
            }
        }
    }
}

} // namespace

np_matrix CISigmaBuilder::compute_ss_2rdm(np_vector C_left, np_vector C_right, Spin spin) const {
    local_timer timer;

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const size_t norb = lists_.norb();

    // if there are less than two orbitals, return an empty matrix
    if (norb < 2) {
        return make_zeros<nb::numpy, double, 2>({0, 0});
    }

    // calculate the number of pairs of orbitals p > q
    const size_t npairs = (norb * (norb - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});

    // skip building the RDM if there are not enough electrons
    if ((is_alpha(spin) and (na < 2)) or (is_beta(spin) and (nb < 2)))
        return rdm;

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    auto rdm_data = rdm.data();

    const auto& alpha_address = lists_.alpha_address();
    const auto& beta_address = lists_.beta_address();

    int num_2h_classes = is_alpha(spin) ? lists_.alpha_address_2h()->nclasses()
                                        : lists_.beta_address_2h()->nclasses();

    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        size_t maxK = is_alpha(spin) ? lists_.alpha_address_2h()->strpcls(class_K)
                                     : lists_.beta_address_2h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.block_size(nI) == 0)
                continue;

            auto tl = gather_block(Cl_span, TL, spin, lists_, class_Ia, class_Ib);

            for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                    (is_beta(spin) and (class_Ia != class_Ja)))
                    continue;
                if (lists_.block_size(nJ) == 0)
                    continue;

                const size_t maxL = is_alpha(spin) ? beta_address->strpcls(class_Ib)
                                                   : alpha_address->strpcls(class_Ia);
                if (maxL > 0) {
                    // Get a pointer to the correct block of matrix C
                    auto tr = gather_block(Cr_span, TR, spin, lists_, class_Ja, class_Jb);
                    for (size_t K{0}; K < maxK; ++K) {
                        auto& Kllist = is_alpha(spin)
                                           ? lists_.get_alpha_2h_list(class_K, K, class_Ia)
                                           : lists_.get_beta_2h_list(class_K, K, class_Ib);
                        auto& Krlist = is_alpha(spin)
                                           ? lists_.get_alpha_2h_list(class_K, K, class_Ja)
                                           : lists_.get_beta_2h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, q, I] : Kllist) {
                            const size_t pq_index = pair_index_gt(p, q);
                            for (const auto& [sign_L, r, s, J] : Krlist) {
                                const size_t rs_index = pair_index_gt(r, s);
                                const double rdm_element =
                                    dot(maxL, tl.data() + I * maxL, 1, tr.data() + J * maxL, 1);
                                rdm_data[pq_index * npairs + rs_index] +=
                                    sign_K * sign_L * rdm_element;
                            }
                        }
                    }
                }
            }
        }
    }
    rdm2_aa_timer_ += timer.elapsed_seconds();
    return rdm;
}

np_matrix CISigmaBuilder::compute_aa_2rdm(np_vector C_left, np_vector C_right) const {
    return compute_ss_2rdm(C_left, C_right, Spin::Alpha);
}

np_matrix CISigmaBuilder::compute_bb_2rdm(np_vector C_left, np_vector C_right) const {
    return compute_ss_2rdm(C_left, C_right, Spin::Beta);
}

np_tensor4 CISigmaBuilder::compute_ab_2rdm(np_vector C_left, np_vector C_right) const {
    local_timer timer;

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const auto norb = lists_.norb();
    const auto norb2 = norb * norb;

    auto rdm = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});

    // skip building the RDM if there are no electrons or there are zero orbitals
    if ((na < 1) or (nb < 1) or (norb < 1)) {
        return rdm;
    }

    auto rdm_data = rdm.data();

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    const int num_1h_class_Ka = lists_.alpha_address_1h()->nclasses();
    const int num_1h_class_Kb = lists_.beta_address_1h()->nclasses();

    const size_t max_composite_K = max_composite_hole_dimension(*lists_.alpha_address_1h(),
                                                                *lists_.beta_address_1h(), "2-RDM");
    if (max_composite_K == 0) {
        rdm2_ab_timer_ += timer.elapsed_seconds();
        return rdm;
    }

    std::vector<double> Kblock1;
    std::vector<double> Kblock2;
    const size_t Kblock_size =
        acquire_local_Kblock_buffers(Kblock1, Kblock2, norb2, max_composite_K);

    // The contraction gamma[uv,xy] = sum_{Ka,Kb} (sign_u sign_v Cl[Ja,Jb])
    // (sign_x sign_y Cr[Ia,Ib]) is a matrix product over the composite one-hole index
    // K = (Ka, Kb). Gather the signed coefficients into B_L[(uv),K] and B_R[(xy),K],
    // then accumulate gamma += B_L * B_R^T one bounded composite-K chunk at a time.
    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        const auto maxKa = lists_.alpha_address_1h()->strpcls(class_Ka);
        for (int class_Kb = 0; class_Kb < num_1h_class_Kb; ++class_Kb) {
            const auto maxKb = lists_.beta_address_1h()->strpcls(class_Kb);
            if ((maxKa == 0) or (maxKb == 0))
                continue;

            const size_t maxK = maxKa * maxKb;

            for (size_t Kblock_start = 0; Kblock_start < maxK;) {
                const size_t Kdim = std::min(Kblock_size, maxK - Kblock_start);
                const auto temp_dim = norb2 * Kdim;

                // B_L[uv, K] in Kblock1, B_R[xy, K] in Kblock2
                std::fill_n(Kblock1.begin(), temp_dim, 0.0);
                std::fill_n(Kblock2.begin(), temp_dim, 0.0);

                // Gather the signed right and left coefficients into B_R[(xy),K] and B_L[(uv),K].
                gather_ab_2rdm_block(lists_, class_Ka, class_Kb, maxKb, Kblock_start, Kdim, norb,
                                     Cr_span, Kblock2);
                gather_ab_2rdm_block(lists_, class_Ka, class_Kb, maxKb, Kblock_start, Kdim, norb,
                                     Cl_span, Kblock1);

                // gamma[uv,xy] += sum_K B_L[uv,K] B_R[xy,K]
                matrix_product('N', 'T', norb2, norb2, Kdim, 1.0, Kblock1.data(), Kdim,
                               Kblock2.data(), Kdim, 1.0, rdm_data, norb2);
                Kblock_start += Kdim;
            }
        }
    }
    rdm2_ab_timer_ += timer.elapsed_seconds();
    return rdm;
}

np_tensor4 CISigmaBuilder::compute_sf_2rdm(np_vector C_left, np_vector C_right) const {
    size_t norb = lists_.norb();
    auto rdm_sf = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});

    if (norb < 1) {
        return rdm_sf; // No 2-RDM for less than 1 orbitals
    }

    auto rdm_sf_v = rdm_sf.view();
    // Mixed-spin contribution (1 orbital or more)
    {
        auto rdm_ab = compute_ab_2rdm(C_left, C_right);
        auto rdm_ab_v = rdm_ab.view();
        for (size_t p{0}; p < norb; ++p) {
            for (size_t q{0}; q < norb; ++q) {
                for (size_t r{0}; r < norb; ++r) {
                    for (size_t s{0}; s < norb; ++s) {
                        rdm_sf_v(p, q, r, s) += rdm_ab_v(p, q, r, s) + rdm_ab_v(q, p, s, r);
                    }
                }
            }
        }
    }

    if (norb < 2) {
        return rdm_sf; // No same-spin contributions to the 2-RDM for less than 2 orbitals
    }

    // To reduce the  memory footprint, we compute the aa and bb contributions in a packed
    // format and one at a time.
    for (auto spin : {Spin::Alpha, Spin::Beta}) {
        auto rdm_ss = compute_ss_2rdm(C_left, C_right, spin);
        auto rdm_ss_v = rdm_ss.view();
        for (size_t p{1}, pq{0}; p < norb; ++p) {
            for (size_t q{0}; q < p; ++q, ++pq) { // p > q
                for (size_t r{1}, rs{0}; r < norb; ++r) {
                    for (size_t s{0}; s < r; ++s, ++rs) { // r > s
                        auto element = rdm_ss_v(pq, rs);
                        rdm_sf_v(p, q, r, s) += element;
                        rdm_sf_v(q, p, r, s) -= element;
                        rdm_sf_v(p, q, s, r) -= element;
                        rdm_sf_v(q, p, s, r) += element;
                    }
                }
            }
        }
    }

    return rdm_sf;
}

} // namespace forte2

#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

namespace forte2 {

namespace {

/// Gather signed two-alpha-hole/one-beta-hole coefficients into an AAB 3-RDM K-block.
void gather_aab_3rdm_block(const CIStrings& lists, int class_Ka, int class_Kb, size_t maxKb,
                           size_t Kblock_start, size_t Kdim, size_t norb,
                           std::span<const double> coefficients, std::span<double> Kblock) {
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
                    Kblock[(row + r) * Kdim + Kidx] =
                        sign_pq * sign_r * coefficients[coefficient_Ia_offset + Ib];
                }
            }
        }
    }
}

/// Gather signed one-alpha-hole/two-beta-hole coefficients into an ABB 3-RDM K-block.
void gather_abb_3rdm_block(const CIStrings& lists, int class_Ka, int class_Kb, size_t maxKb,
                           size_t Kblock_start, size_t Kdim, size_t npair,
                           std::span<const double> coefficients, std::span<double> Kblock) {
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
                    Kblock[(row + qr_index) * Kdim + Kidx] =
                        sign_p * sign_qr * coefficients[coefficient_Ia_offset + Ib];
                }
            }
        }
    }
}

} // namespace

np_matrix CISigmaBuilder::compute_sss_3rdm(np_vector C_left, np_vector C_right, Spin spin) const {
    local_timer timer;

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const auto norb = lists_.norb();

    // if there are less than three orbitals, return an empty matrix
    if (norb < 3) {
        return make_zeros<nb::numpy, double, 2>({0, 0});
    }

    const size_t ntriplets = (norb * (norb - 1) * (norb - 2)) / 6;
    auto rdm = make_zeros<nb::numpy, double, 2>({ntriplets, ntriplets});

    // skip building the RDM if there are not enough electrons
    if ((is_alpha(spin) and (na < 3)) or (is_beta(spin) and (nb < 3)))
        return rdm;

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    auto rdm_data = rdm.data();
    const auto& alpha_address = lists_.alpha_address();
    const auto& beta_address = lists_.beta_address();

    int num_3h_classes = is_alpha(spin) ? lists_.alpha_address_3h()->nclasses()
                                        : lists_.beta_address_3h()->nclasses();

    for (int class_K = 0; class_K < num_3h_classes; ++class_K) {
        size_t maxK = is_alpha(spin) ? lists_.alpha_address_3h()->strpcls(class_K)
                                     : lists_.beta_address_3h()->strpcls(class_K);

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
                                           ? lists_.get_alpha_3h_list(class_K, K, class_Ia)
                                           : lists_.get_beta_3h_list(class_K, K, class_Ib);
                        auto& Krlist = is_alpha(spin)
                                           ? lists_.get_alpha_3h_list(class_K, K, class_Ja)
                                           : lists_.get_beta_3h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, q, r, I] : Kllist) {
                            const size_t pqr_index = triplet_index_gt(p, q, r);
                            for (const auto& [sign_L, s, t, u, J] : Krlist) {
                                const size_t stu_index = triplet_index_gt(s, t, u);
                                const double rdm_element =
                                    dot(maxL, tl.data() + I * maxL, 1, tr.data() + J * maxL, 1);
                                rdm_data[pqr_index * ntriplets + stu_index] +=
                                    sign_K * sign_L * rdm_element;
                            }
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

np_matrix CISigmaBuilder::compute_aaa_3rdm(np_vector C_left, np_vector C_right) const {
    return compute_sss_3rdm(C_left, C_right, Spin::Alpha);
}

np_matrix CISigmaBuilder::compute_bbb_3rdm(np_vector C_left, np_vector C_right) const {
    return compute_sss_3rdm(C_left, C_right, Spin::Beta);
}

np_tensor4 CISigmaBuilder::compute_aab_3rdm(np_vector C_left, np_vector C_right) const {
    local_timer timer;
    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const auto norb = lists_.norb();

    // if there are less than two orbitals, return an empty matrix
    if (norb < 2) {
        return make_zeros<nb::numpy, double, 4>({0, 0, 0, 0});
    }

    // the number of orbital pairs i > j of the same spin
    const size_t npair = (norb * (norb - 1)) / 2;

    auto rdm = make_zeros<nb::numpy, double, 4>({npair, norb, npair, norb});

    // skip building the RDM if there are not enough electrons
    if ((na < 2) or (nb < 1))
        return rdm;

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    auto rdm_data = rdm.data();

    int num_2h_class_Ka = lists_.alpha_address_2h()->nclasses();
    int num_1h_class_Kb = lists_.beta_address_1h()->nclasses();

    const size_t max_composite_K = max_composite_hole_dimension(
        *lists_.alpha_address_2h(), *lists_.beta_address_1h(), "AAB 3-RDM");
    if (max_composite_K == 0)
        return rdm;

    std::vector<double> Kblock1;
    std::vector<double> Kblock2;

    // The contraction gamma[(uv,w),(xy,z)] is a matrix product over the composite hole
    // index K = (Ka, Kb) (Ka: 2-hole alpha, Kb: 1-hole beta). Gather signed left/right
    // coefficients into B_L[(uv*norb+w),K] and B_R[(xy*norb+z),K], then accumulate
    // gamma += B_L * B_R^T one bounded composite-K chunk at a time.
    const size_t M = npair * norb;
    const size_t Kblock_size = acquire_local_Kblock_buffers(Kblock1, Kblock2, M, max_composite_K);
    for (int class_Ka{0}; class_Ka < num_2h_class_Ka; ++class_Ka) {
        const size_t maxKa = lists_.alpha_address_2h()->strpcls(class_Ka);
        for (int class_Kb{0}; class_Kb < num_1h_class_Kb; ++class_Kb) {
            const size_t maxKb = lists_.beta_address_1h()->strpcls(class_Kb);
            if ((maxKa == 0) or (maxKb == 0))
                continue;

            const size_t maxK = maxKa * maxKb;

            for (size_t Kblock_start = 0; Kblock_start < maxK;) {
                const size_t Kdim = std::min(Kblock_size, maxK - Kblock_start);
                const auto temp_dim = M * Kdim;

                std::fill_n(Kblock1.begin(), temp_dim, 0.0);
                std::fill_n(Kblock2.begin(), temp_dim, 0.0);

                // Gather the signed right and left coefficients into B_R[(xy,z),K] and
                // B_L[(uv,w),K].
                gather_aab_3rdm_block(lists_, class_Ka, class_Kb, maxKb, Kblock_start, Kdim, norb,
                                      Cr_span, Kblock2);
                gather_aab_3rdm_block(lists_, class_Ka, class_Kb, maxKb, Kblock_start, Kdim, norb,
                                      Cl_span, Kblock1);

                matrix_product('N', 'T', M, M, Kdim, 1.0, Kblock1.data(), Kdim, Kblock2.data(),
                               Kdim, 1.0, rdm_data, M);
                Kblock_start += Kdim;
            }
        }
    }
    return rdm;
}

np_tensor4 CISigmaBuilder::compute_abb_3rdm(np_vector C_left, np_vector C_right) const {
    local_timer timer;
    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const auto norb = lists_.norb();

    // if there are less than two orbitals, return an empty matrix
    if (norb < 2) {
        return make_zeros<nb::numpy, double, 4>({0, 0, 0, 0});
    }

    // the number of orbital pairs i > j of the same spin
    const size_t npair = (norb * (norb - 1)) / 2;

    auto rdm = make_zeros<nb::numpy, double, 4>({norb, npair, norb, npair});

    // skip building the RDM if there are not enough electrons
    if ((na < 1) or (nb < 2))
        return rdm;

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    auto rdm_data = rdm.data();

    int num_1h_class_Ka = lists_.alpha_address_1h()->nclasses();
    int num_2h_class_Kb = lists_.beta_address_2h()->nclasses();

    const size_t max_composite_K = max_composite_hole_dimension(
        *lists_.alpha_address_1h(), *lists_.beta_address_2h(), "ABB 3-RDM");
    if (max_composite_K == 0)
        return rdm;

    std::vector<double> Kblock1;
    std::vector<double> Kblock2;

    // GEMM reformulation, mirroring compute_aab_3rdm with the spins swapped: the composite
    // hole index is K = (Ka, Kb) (Ka: 1-hole alpha, Kb: 2-hole beta). Gather signed
    // coefficients into B_L[(u*npair+vw),K] and B_R[(x*npair+yz),K], then accumulate
    // gamma += B_L * B_R^T one bounded composite-K chunk at a time.
    const size_t M = norb * npair;
    const size_t Kblock_size = acquire_local_Kblock_buffers(Kblock1, Kblock2, M, max_composite_K);
    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        const size_t maxKa = lists_.alpha_address_1h()->strpcls(class_Ka);
        for (int class_Kb = 0; class_Kb < num_2h_class_Kb; ++class_Kb) {
            const size_t maxKb = lists_.beta_address_2h()->strpcls(class_Kb);
            if ((maxKa == 0) or (maxKb == 0))
                continue;

            const size_t maxK = maxKa * maxKb;

            for (size_t Kblock_start = 0; Kblock_start < maxK;) {
                const size_t Kdim = std::min(Kblock_size, maxK - Kblock_start);
                const auto temp_dim = M * Kdim;

                std::fill_n(Kblock1.begin(), temp_dim, 0.0);
                std::fill_n(Kblock2.begin(), temp_dim, 0.0);

                // Gather the signed right and left coefficients into B_R[(x,yz),K] and
                // B_L[(u,vw),K].
                gather_abb_3rdm_block(lists_, class_Ka, class_Kb, maxKb, Kblock_start, Kdim, npair,
                                      Cr_span, Kblock2);
                gather_abb_3rdm_block(lists_, class_Ka, class_Kb, maxKb, Kblock_start, Kdim, npair,
                                      Cl_span, Kblock1);

                matrix_product('N', 'T', M, M, Kdim, 1.0, Kblock1.data(), Kdim, Kblock2.data(),
                               Kdim, 1.0, rdm_data, M);
                Kblock_start += Kdim;
            }
        }
    }
    return rdm;
}

} // namespace forte2

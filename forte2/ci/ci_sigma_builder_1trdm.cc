#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"

#include "ci_sigma_builder.h"
#include "ci_strings_makers.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_s_1trdm(const CISigmaBuilder& sigmabuilder_right,
                                          np_vector C_left, np_vector C_right, Spin spin) const {
    const auto& sigmabuilder_left = *this;

    const auto& lists_left = sigmabuilder_left.lists_;
    const auto& lists_right = sigmabuilder_right.lists_;

    // Get the number of electrons
    const auto na_left = lists_left.na();
    const auto nb_left = lists_left.nb();
    const auto na_right = lists_right.na();
    const auto nb_right = lists_right.nb();
    const auto norb = lists_left.norb();

    if (lists_left.norb() != lists_right.norb()) {
        throw std::runtime_error("FCI transition RDMs: The number of MOs must be the same in "
                                 "the two wave functions.");
    }

    if (na_left != na_right or nb_left != nb_right) {
        throw std::runtime_error("FCI transition RDMs: The number of alfa and beta electrons "
                                 "must be the same in the "
                                 "two wave functions.");
    }
    auto rdm = make_zeros<nb::numpy, double, 2>({norb, norb});
    if ((is_alpha(spin) and (na_left < 1)) or (is_beta(spin) and (nb_left < 1)) or (norb < 1))
        return rdm;

    auto rdm_view = rdm.view();

    const auto& alfa_address_left = lists_left.alfa_address();
    const auto& beta_address_left = lists_left.beta_address();
    const auto& alfa_address_right = lists_right.alfa_address();
    const auto& beta_address_right = lists_right.beta_address();

    // Compute the lists that map the strings of the right and left wave functions. We only need
    // strings for the part that is left untouched by the a^+_p a_q operator.
    auto string_list = find_string_map(lists_left, lists_right, spin);
    // Compute the VO lists that map the strings of the right and left wave functions. We only
    // need strings for the part that is affected by the a^+_p a_q operator.
    VOListMap vo_list = find_ov_string_map(lists_left, lists_right, spin);

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    // Here we compute the RDMs for the case of different irreps
    // <Ja|a^{+}_p a_q|Ia> CL_{Ja,K} CR_{Ia,K}

    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_right.determinant_classes()) {
        if (lists_right.block_size(nI) == 0)
            continue;

        auto tr =
            gather_block(Cr_span, sigmabuilder_right.TR, spin, lists_right, class_Ia, class_Ib);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_left.determinant_classes()) {
            // check if the string class on which we don't act is the same for I and J
            // here we cannot assume that the two classes must coincide. So we just check if
            // there are elements in the string list for the given pair of classes
            if (is_alpha(spin)) {
                if (string_list.count(std::make_pair(class_Ib, class_Jb)) == 0)
                    continue;
            } else {
                if (string_list.count(std::make_pair(class_Ia, class_Ja)) == 0)
                    continue;
            }

            if (lists_left.block_size(nJ) == 0)
                continue;

            auto tl = gather_block(Cl_span, TL, spin, lists_left, class_Ja, class_Jb);

            const auto& string_list_block = is_alpha(spin)
                                                ? string_list[std::make_pair(class_Ib, class_Jb)]
                                                : string_list[std::make_pair(class_Ia, class_Ja)];

            const auto& pq_vo_list = is_alpha(spin) ? vo_list[std::make_pair(class_Ia, class_Ja)]
                                                    : vo_list[std::make_pair(class_Ib, class_Jb)];

            const size_t maxL_left = is_alpha(spin) ? beta_address_left->strpcls(class_Jb)
                                                    : alfa_address_left->strpcls(class_Ja);

            const size_t maxL_right = is_alpha(spin) ? beta_address_right->strpcls(class_Ib)
                                                     : alfa_address_right->strpcls(class_Ia);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                double rdm_element = 0.0;
                for (const auto& [sign, I, J] : vo_list) {
                    for (const auto& [Ip, Jp] : string_list_block)
                        rdm_element += sign * tl[J * maxL_left + Jp] * tr[I * maxL_right + Ip];
                }
                rdm_view(p, q) += rdm_element;
            }
        }
    }
    return rdm;
}

np_matrix CISigmaBuilder::compute_a_1trdm(CISigmaBuilder& sigmabuilder_right, np_vector C_left,
                                          np_vector C_right) const {
    return compute_s_1trdm(sigmabuilder_right, C_left, C_right, Spin::Alpha);
}

np_matrix CISigmaBuilder::compute_b_1trdm(CISigmaBuilder& sigmabuilder_right, np_vector C_left,
                                          np_vector C_right) const {
    return compute_s_1trdm(sigmabuilder_right, C_left, C_right, Spin::Beta);
}

np_matrix CISigmaBuilder::compute_sf_1trdm(CISigmaBuilder& sigmabuilder_right, np_vector C_left,
                                           np_vector C_right) const {
    auto sf_1trdm = make_zeros<nb::numpy, double, 2>({lists_.norb(), lists_.norb()});
    if (lists_.norb() > 0) {
        auto a_1trdm = compute_a_1trdm(sigmabuilder_right, C_left, C_right);
        auto b_1trdm = compute_b_1trdm(sigmabuilder_right, C_left, C_right);
        matrix::daxpy<double>(1.0, a_1trdm, sf_1trdm);
        matrix::daxpy<double>(1.0, b_1trdm, sf_1trdm);
    }
    return sf_1trdm;
}

} // namespace forte2

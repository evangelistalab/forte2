#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"

#include "ci_sigma_builder.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_a_1rdm(np_vector C_left, np_vector C_right, bool alfa) const {
    size_t norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, double, 2>({norb, norb});
    auto na = lists_.na();
    auto nb = lists_.nb();

    // skip building the RDM if there are not enough electrons
    if ((alfa and (na < 1)) or ((!alfa) and (nb < 1)))
        return rdm;

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();
    auto rdm_view = rdm.view();
    auto Cl_span = vector::as_span(C_left);
    auto Cr_span = vector::as_span(C_right);

    // loop over blocks of the right state
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        auto tr = gather_block(Cr_span, TR, alfa, lists_, class_Ia, class_Ib);

        // loop over blocks of the left state
        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                continue;
            if (lists_.detpblk(nJ) == 0)
                continue;

            auto tl = gather_block(Cl_span, TL, alfa, lists_, class_Ja, class_Jb);

            const size_t maxL =
                alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

            const auto& pq_vo_list = alfa ? lists_.get_alfa_vo_list(class_Ia, class_Ja)
                                          : lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                double rdm_element = 0.0;
                for (const auto& [sign, I, J] : vo_list) {
                    // Compute the RDM element contribution
                    for (size_t idx{0}; idx != maxL; ++idx) {
                        rdm_element += sign * tr[I * maxL + idx] * tl[J * maxL + idx];
                    }
                }
                rdm_view(p, q) += rdm_element;
            }
        }
    }
    return rdm;
}

np_matrix CISigmaBuilder::compute_sf_1rdm(np_vector C_left, np_vector C_right) const {
    auto rdm_a = compute_a_1rdm(C_left, C_right, true);
    auto rdm_b = compute_a_1rdm(C_left, C_right, false);
    matrix::daxpy(1.0, rdm_a, rdm_b);
    return rdm_b;
}

} // namespace forte2

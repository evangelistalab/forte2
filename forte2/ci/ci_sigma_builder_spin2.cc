#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"

#include "ci_sigma_builder.h"

namespace forte2 {

double CISigmaBuilder::compute_spin2() {
    double spin2 = 0.0;
    for (const auto& [nI, class_Ia, class_Ib] : lists_->determinant_classes()) {
        size_t block_size = alfa_address_->strpcls(class_Ia) * beta_address_->strpcls(class_Ib);
        if (block_size == 0)
            continue;

        // The pq product is totally symmetric so the classes of the result are the same as the
        // classes of the input
        const auto Cr = C_[nI]->pointer();
        for (const auto& [nJ, class_Ja, class_Jb] : lists_->determinant_classes()) {
            auto Cl = C_[nJ]->pointer();
            const auto& pq_vo_alfa = lists_->get_alfa_vo_list(class_Ia, class_Ja);
            const auto& pq_vo_beta = lists_->get_beta_vo_list(class_Ib, class_Jb);
            // loop over the alfa (p,q) pairs
            for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
                const auto& [p, q] = pq;
                // the correspoding beta pair will be (q,p)
                // check if the pair (q,p) is in the list, if not continue
                if (pq_vo_beta.count(std::make_tuple(q, p)) == 0)
                    continue;
                const auto& vo_beta_list = pq_vo_beta.at(std::make_tuple(q, p));
                for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
                    for (const auto& [sign_b, Ib, Jb] : vo_beta_list) {
                        spin2 += Cl[Ja][Jb] * Cr[Ia][Ib] * sign_a * sign_b;
                    }
                }
            }
        }
    }
    double na = alfa_address_->nones();
    double nb = beta_address_->nones();
    return -spin2 + 0.25 * std::pow(na - nb, 2.0) + 0.5 * (na + nb);
}

} // namespace forte2

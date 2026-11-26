#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"

#include "ci_sigma_builder.h"
#include "ci_one_rdm.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_s_1rdm(np_vector C_left, np_vector C_right, Spin spin) const {
    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const auto norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, double, 2>({norb, norb});

    // skip building the RDM if there are no electrons or there are zero orbitals
    if ((is_alpha(spin) and (na < 1)) or (is_beta(spin) and (nb < 1)) or (norb < 1))
        return rdm;

    accumulate_one_rdm<double>(lists_, spin, vector::as_span<double>(C_left),
                               vector::as_span<double>(C_right), TL, TR, rdm.view());
    return rdm;
}

np_matrix CISigmaBuilder::compute_a_1rdm(np_vector C_left, np_vector C_right) const {
    return compute_s_1rdm(C_left, C_right, Spin::Alpha);
}

np_matrix CISigmaBuilder::compute_b_1rdm(np_vector C_left, np_vector C_right) const {
    return compute_s_1rdm(C_left, C_right, Spin::Beta);
}

np_matrix CISigmaBuilder::compute_sf_1rdm(np_vector C_left, np_vector C_right) const {
    auto sf_1rdm = make_zeros<nb::numpy, double, 2>({lists_.norb(), lists_.norb()});
    if (lists_.norb() > 0) {
        auto a_1rdm = compute_a_1rdm(C_left, C_right);
        auto b_1rdm = compute_b_1rdm(C_left, C_right);
        matrix::daxpy<double>(1.0, a_1rdm, sf_1rdm);
        matrix::daxpy<double>(1.0, b_1rdm, sf_1rdm);
    }
    return sf_1rdm;
}

} // namespace forte2

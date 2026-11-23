#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"
#include "ci_hole_contractions.h"

namespace forte2 {

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

    accumulate_same_spin_rdm<2>(lists_, spin, vector::as_span<double>(C_left),
                                vector::as_span<double>(C_right), TL, TR, rdm.data(), npairs);
    rdm2_aa_timer_ += timer.elapsed_seconds();
    return rdm;
}

np_matrix CISigmaBuilder::compute_aa_2rdm(np_vector C_left, np_vector C_right) const {
    return compute_ss_2rdm(C_left, C_right, Spin::Alpha);
}

np_matrix CISigmaBuilder::compute_bb_2rdm(np_vector C_left, np_vector C_right) const {
    return compute_ss_2rdm(C_left, C_right, Spin::Beta);
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

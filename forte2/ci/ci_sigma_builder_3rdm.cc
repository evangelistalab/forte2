#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"
#include "ci_hole_contractions.h"

namespace forte2 {

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

    accumulate_same_spin_rdm<3>(lists_, spin, vector::as_span<double>(C_left),
                                vector::as_span<double>(C_right), TL, TR, rdm.data(), ntriplets);
    return rdm;
}

np_matrix CISigmaBuilder::compute_aaa_3rdm(np_vector C_left, np_vector C_right) const {
    return compute_sss_3rdm(C_left, C_right, Spin::Alpha);
}

np_matrix CISigmaBuilder::compute_bbb_3rdm(np_vector C_left, np_vector C_right) const {
    return compute_sss_3rdm(C_left, C_right, Spin::Beta);
}

np_tensor6 CISigmaBuilder::compute_sf_3rdm(np_vector C_left, np_vector C_right) const {
    auto norb = lists_.norb();
    auto rdm_sf = make_zeros<nb::numpy, double, 6>({norb, norb, norb, norb, norb, norb});

    if (norb < 2) {
        return rdm_sf; // No 3-RDM for less than 2 orbitals
    }

    // Strides of the dense row-major spin-free 6-index tensor rdm_sf[p,q,r,s,t,u].
    const size_t n = norb;
    const size_t n2 = n * n;
    const size_t n3 = n2 * n;
    const size_t n4 = n3 * n;
    const size_t n5 = n4 * n;
    const size_t npair = (norb * (norb - 1)) / 2;
    auto* sf = rdm_sf.data();

    // The aab contribution
    {
        auto rdm_aab = compute_aab_3rdm(C_left, C_right);
        const auto* aab = rdm_aab.data();

        for (size_t p{1}, pq{0}; p < norb; ++p) {
            for (size_t q{0}; q < p; ++q, ++pq) {
                for (size_t r{0}; r < norb; ++r) {
                    for (size_t s{1}, st{0}; s < norb; ++s) {
                        for (size_t t{0}; t < s; ++t, ++st) {
                            const size_t offset = ((pq * n + r) * npair + st) * n;
                            // u has stride 1 in these targets
                            const size_t g1a = p * n5 + q * n4 + r * n3 + s * n2 + t * n;
                            const size_t g1b = p * n5 + q * n4 + r * n3 + t * n2 + s * n;
                            const size_t g1c = q * n5 + p * n4 + r * n3 + s * n2 + t * n;
                            const size_t g1d = q * n5 + p * n4 + r * n3 + t * n2 + s * n;
                            // u has stride n in these targets
                            const size_t g2a = p * n5 + r * n4 + q * n3 + s * n2 + t;
                            const size_t g2b = p * n5 + r * n4 + q * n3 + t * n2 + s;
                            const size_t g2c = q * n5 + r * n4 + p * n3 + s * n2 + t;
                            const size_t g2d = q * n5 + r * n4 + p * n3 + t * n2 + s;
                            // u has stride n2 in these targets
                            const size_t g3a = r * n5 + p * n4 + q * n3 + s * n + t;
                            const size_t g3b = r * n5 + p * n4 + q * n3 + t * n + s;
                            const size_t g3c = r * n5 + q * n4 + p * n3 + s * n + t;
                            const size_t g3d = r * n5 + q * n4 + p * n3 + t * n + s;
                            for (size_t u{0}; u < norb; ++u) {
                                const auto el = aab[offset + u];
                                // G3("pqrstu") += g3aab_("pqrstu");
                                sf[g1a + u] += el;
                                sf[g1b + u] -= el;
                                sf[g1c + u] -= el;
                                sf[g1d + u] += el;
                                // G3("prqsut") += g3aab_("pqrstu");
                                const size_t un = u * n;
                                sf[g2a + un] += el;
                                sf[g2b + un] -= el;
                                sf[g2c + un] -= el;
                                sf[g2d + un] += el;
                                // G3("rpqust") += g3aab_("pqrstu");
                                const size_t un2 = u * n2;
                                sf[g3a + un2] += el;
                                sf[g3b + un2] -= el;
                                sf[g3c + un2] -= el;
                                sf[g3d + un2] += el;
                            }
                        }
                    }
                }
            }
        }
    }

    // The abb contribution
    {
        auto rdm_abb = compute_abb_3rdm(C_left, C_right);
        const auto* abb = rdm_abb.data();
        for (size_t p{0}; p < norb; ++p) {
            for (size_t q{1}, qr{0}; q < norb; ++q) {
                for (size_t r{0}; r < q; ++r, ++qr) {
                    for (size_t s{0}; s < norb; ++s) {
                        const size_t offset = ((p * npair + qr) * n + s) * npair;
                        for (size_t t{1}, tu{0}; t < norb; ++t) {
                            // bases with u at stride 1 (u in 6th slot)
                            const size_t b1 = p * n5 + q * n4 + r * n3 + s * n2 + t * n; //(pqrstu)+
                            const size_t b4 = p * n5 + r * n4 + q * n3 + s * n2 + t * n; //(prqstu)-
                            const size_t b5 = q * n5 + p * n4 + r * n3 + t * n2 + s * n; //(qprtsu)+
                            const size_t b7 = r * n5 + p * n4 + q * n3 + t * n2 + s * n; //(rpqtsu)-
                            // bases with u at stride n (u in 5th slot)
                            const size_t b2 = p * n5 + q * n4 + r * n3 + s * n2 + t;  //(pqrsut)-
                            const size_t b3 = p * n5 + r * n4 + q * n3 + s * n2 + t;  //(prqsut)+
                            const size_t b9 = q * n5 + r * n4 + p * n3 + t * n2 + s;  //(qrptus)+
                            const size_t b11 = r * n5 + q * n4 + p * n3 + t * n2 + s; //(rqptus)-
                            // bases with u at stride n2 (u in 4th slot)
                            const size_t b6 = q * n5 + p * n4 + r * n3 + s * n + t;  //(qprust)-
                            const size_t b8 = r * n5 + p * n4 + q * n3 + s * n + t;  //(rpqust)+
                            const size_t b10 = q * n5 + r * n4 + p * n3 + t * n + s; //(qrputs)-
                            const size_t b12 = r * n5 + q * n4 + p * n3 + t * n + s; //(rqputs)+
                            for (size_t u{0}; u < t; ++u, ++tu) {
                                const auto el = abb[offset + tu];
                                const size_t un = u * n;
                                const size_t un2 = u * n2;
                                sf[b1 + u] += el;
                                sf[b4 + u] -= el;
                                sf[b5 + u] += el;
                                sf[b7 + u] -= el;
                                sf[b2 + un] -= el;
                                sf[b3 + un] += el;
                                sf[b9 + un] += el;
                                sf[b11 + un] -= el;
                                sf[b6 + un2] -= el;
                                sf[b8 + un2] += el;
                                sf[b10 + un2] -= el;
                                sf[b12 + un2] += el;
                            }
                        }
                    }
                }
            }
        }
    }

    if (norb < 3) {
        return rdm_sf; // No same-spin contributions to the 3-RDM for less than 3 orbitals
    }

    // The aaa/bbb (same-spin) contributions. For a fixed bra triplet (p>q>r) the whole (s,t,u)
    // ket sub-block is identical up to the bra-permutation sign, so we assemble that norb^3 block
    // once in a buffer with the ket signs, then add each of the 6
    // bra permutations as a contiguous block onto rdm_sf
    const size_t ntriplets = (norb * (norb - 1) * (norb - 2)) / 6;
    const size_t N3 = n3;
    std::vector<double> ketrow(N3, 0.0);

    auto add_block = [&](bool positive, size_t a, size_t b, size_t c) {
        auto* dst = sf + ((a * n + b) * n + c) * N3;
        if (positive) {
            for (size_t i = 0; i < N3; ++i)
                dst[i] += ketrow[i];
        } else {
            for (size_t i = 0; i < N3; ++i)
                dst[i] -= ketrow[i];
        }
    };

    for (auto spin : {Spin::Alpha, Spin::Beta}) {
        auto rdm_sss = compute_sss_3rdm(C_left, C_right, spin);
        const auto* sss_data = rdm_sss.data();

        for (size_t p{2}, pqr{0}; p < norb; ++p) {
            for (size_t q{1}; q < p; ++q) {
                for (size_t r{0}; r < q; ++r, ++pqr) {
                    const auto* row = sss_data + pqr * ntriplets;
                    // Scatter the ket triplets into the (s,t,u) block with antisymmetric signs.
                    for (size_t s{2}, stu{0}; s < norb; ++s) {
                        for (size_t t{1}; t < s; ++t) {
                            for (size_t u{0}; u < t; ++u, ++stu) {
                                const auto el = row[stu];
                                ketrow[(s * n + t) * n + u] = +el;
                                ketrow[(s * n + u) * n + t] = -el;
                                ketrow[(u * n + s) * n + t] = +el;
                                ketrow[(u * n + t) * n + s] = -el;
                                ketrow[(t * n + u) * n + s] = +el;
                                ketrow[(t * n + s) * n + u] = -el;
                            }
                        }
                    }
                    // Add the 6 bra permutations as contiguous signed accumulations.
                    add_block(true, p, q, r);
                    add_block(false, p, r, q);
                    add_block(true, r, p, q);
                    add_block(false, r, q, p);
                    add_block(true, q, r, p);
                    add_block(false, q, p, r);
                }
            }
        }
    }

    return rdm_sf;
}

} // namespace forte2

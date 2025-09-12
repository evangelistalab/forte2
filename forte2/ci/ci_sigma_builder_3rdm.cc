#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

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

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    auto rdm_data = rdm.data();
    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    int num_3h_classes = is_alpha(spin) ? lists_.alfa_address_3h()->nclasses()
                                        : lists_.beta_address_3h()->nclasses();

    for (int class_K = 0; class_K < num_3h_classes; ++class_K) {
        size_t maxK = is_alpha(spin) ? lists_.alfa_address_3h()->strpcls(class_K)
                                     : lists_.beta_address_3h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.block_size(nI) == 0)
                continue;

            auto tr = gather_block(Cr_span, TR, spin, lists_, class_Ia, class_Ib);

            for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                    (is_beta(spin) and (class_Ia != class_Ja)))
                    continue;
                if (lists_.block_size(nJ) == 0)
                    continue;

                const size_t maxL = is_alpha(spin) ? beta_address->strpcls(class_Ib)
                                                   : alfa_address->strpcls(class_Ia);

                if (maxL > 0) {
                    // Get a pointer to the correct block of matrix C
                    auto tl = gather_block(Cl_span, TL, spin, lists_, class_Ja, class_Jb);

                    for (size_t K{0}; K < maxK; ++K) {
                        auto& Krlist = is_alpha(spin)
                                           ? lists_.get_alfa_3h_list(class_K, K, class_Ia)
                                           : lists_.get_beta_3h_list(class_K, K, class_Ib);
                        auto& Kllist = is_alpha(spin)
                                           ? lists_.get_alfa_3h_list(class_K, K, class_Ja)
                                           : lists_.get_beta_3h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, q, r, I] : Krlist) {
                            const size_t pqr_index = triplet_index_gt(p, q, r);
                            for (const auto& [sign_L, s, t, u, J] : Kllist) {
                                const size_t stu_index = triplet_index_gt(s, t, u);
                                const double rdm_element =
                                    dot(maxL, tr.data() + I * maxL, 1, tl.data() + J * maxL, 1);
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

    auto stride1 = norb;
    auto stride2 = stride1 * npair;
    auto stride3 = stride2 * norb;

    auto index = [stride1, stride2, stride3](size_t pq, size_t r, size_t st, size_t u) {
        return pq * stride3 + r * stride2 + st * stride1 + u;
    };

    // skip building the RDM if there are not enough electrons
    if ((na < 2) or (nb < 1))
        return rdm;

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    auto rdm_data = rdm.data();
    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    int num_2h_class_Ka = lists_.alfa_address_2h()->nclasses();
    int num_1h_class_Kb = lists_.beta_address_1h()->nclasses();

    for (int class_Ka{0}; class_Ka < num_2h_class_Ka; ++class_Ka) {
        size_t maxKa = lists_.alfa_address_2h()->strpcls(class_Ka);

        for (int class_Kb{0}; class_Kb < num_1h_class_Kb; ++class_Kb) {
            size_t maxKb = lists_.beta_address_1h()->strpcls(class_Kb);

            // loop over blocks of matrix C
            for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                if (lists_.block_size(nI) == 0)
                    continue;

                const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                const auto Cr_offset = lists_.block_offset(nI);

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (lists_.block_size(nJ) == 0)
                        continue;

                    const auto maxJb = lists_.beta_address()->strpcls(class_Jb);
                    const auto Cl_offset = lists_.block_offset(nJ);

                    for (size_t Ka = 0; Ka < maxKa; ++Ka) {
                        auto& Ka_right_list = lists_.get_alfa_2h_list(class_Ka, Ka, class_Ia);
                        auto& Ka_left_list = lists_.get_alfa_2h_list(class_Ka, Ka, class_Ja);
                        for (size_t Kb = 0; Kb < maxKb; ++Kb) {
                            auto& Kb_right_list = lists_.get_beta_1h_list(class_Kb, Kb, class_Ib);
                            auto& Kb_left_list = lists_.get_beta_1h_list(class_Kb, Kb, class_Jb);
                            for (const auto& [sign_uv, u, v, Ja] : Ka_left_list) {
                                const auto uv_index = pair_index_gt(u, v);
                                for (const auto& [sign_w, w, Jb] : Kb_left_list) {
                                    const auto ClJ =
                                        sign_uv * sign_w * Cl_span[Cl_offset + Ja * maxJb + Jb];
                                    for (const auto& [sign_xy, x, y, Ia] : Ka_right_list) {
                                        const auto xy_index = pair_index_gt(x, y);
                                        const auto Cr_Ia_offset = Cr_offset + Ia * maxIb;
                                        for (const auto& [sign_z, z, Ib] : Kb_right_list) {
                                            rdm_data[index(uv_index, w, xy_index, z)] +=
                                                sign_xy * sign_z * ClJ * Cr_span[Cr_Ia_offset + Ib];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
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

    auto stride1 = npair;
    auto stride2 = stride1 * norb;
    auto stride3 = stride2 * npair;

    auto index = [stride1, stride2, stride3](size_t p, size_t qr, size_t s, size_t tu) {
        return p * stride3 + qr * stride2 + s * stride1 + tu;
    };

    // skip building the RDM if there are not enough electrons
    if ((na < 1) or (nb < 2))
        return rdm;

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    auto rdm_data = rdm.data();
    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    int num_1h_class_Ka = lists_.alfa_address_1h()->nclasses();
    int num_2h_class_Kb = lists_.beta_address_2h()->nclasses();

    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        size_t maxKa = lists_.alfa_address_1h()->strpcls(class_Ka);

        for (int class_Kb = 0; class_Kb < num_2h_class_Kb; ++class_Kb) {
            size_t maxKb = lists_.beta_address_2h()->strpcls(class_Kb);

            // loop over blocks of matrix C
            for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                if (lists_.block_size(nI) == 0)
                    continue;

                const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                const auto Cr_offset = lists_.block_offset(nI);

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (lists_.block_size(nJ) == 0)
                        continue;

                    const auto maxJb = lists_.beta_address()->strpcls(class_Jb);
                    const auto Cl_offset = lists_.block_offset(nJ);

                    for (size_t Ka = 0; Ka < maxKa; ++Ka) {
                        auto& Ka_right_list = lists_.get_alfa_1h_list(class_Ka, Ka, class_Ia);
                        auto& Ka_left_list = lists_.get_alfa_1h_list(class_Ka, Ka, class_Ja);
                        for (size_t Kb = 0; Kb < maxKb; ++Kb) {
                            auto& Kb_right_list = lists_.get_beta_2h_list(class_Kb, Kb, class_Ib);
                            auto& Kb_left_list = lists_.get_beta_2h_list(class_Kb, Kb, class_Jb);
                            for (const auto& [sign_u, u, Ja] : Ka_left_list) {
                                for (const auto& [sign_vw, v, w, Jb] : Kb_left_list) {
                                    const auto vw_index = pair_index_gt(v, w);
                                    const auto ClJ =
                                        sign_u * sign_vw * Cl_span[Cl_offset + Ja * maxJb + Jb];
                                    for (const auto& [sign_x, x, Ia] : Ka_right_list) {
                                        const auto Cr_Ia_offset = Cr_offset + Ia * maxIb;
                                        for (const auto& [sign_yz, y, z, Ib] : Kb_right_list) {
                                            const auto yz_index = pair_index_gt(y, z);
                                            rdm_data[index(u, vw_index, x, yz_index)] +=
                                                sign_x * sign_yz * ClJ * Cr_span[Cr_Ia_offset + Ib];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

np_tensor6 CISigmaBuilder::compute_sf_3rdm(np_vector C_left, np_vector C_right) const {
    auto norb = lists_.norb();
    auto rdm_sf = make_zeros<nb::numpy, double, 6>({norb, norb, norb, norb, norb, norb});

    if (norb < 2) {
        return rdm_sf; // No 3-RDM for less than 2 orbitals
    }

    auto rdm_sf_v = rdm_sf.view();
    // The aab contribution (2 orbitals or more)
    {
        auto rdm_aab = compute_aab_3rdm(C_left, C_right);
        auto rdm_aab_v = rdm_aab.view();

        for (size_t p{1}, pq{0}; p < norb; ++p) {
            for (size_t q{0}; q < p; ++q, ++pq) {
                for (size_t r{0}; r < norb; ++r) {
                    for (size_t s{1}, st{0}; s < norb; ++s) {
                        for (size_t t{0}; t < s; ++t, ++st) {
                            for (size_t u{0}; u < norb; ++u) {
                                const auto el = rdm_aab_v(pq, r, st, u);
                                // G3("pqrstu") += g3aab_("pqrstu");
                                rdm_sf_v(p, q, r, s, t, u) += el;
                                rdm_sf_v(p, q, r, t, s, u) -= el;
                                rdm_sf_v(q, p, r, s, t, u) -= el;
                                rdm_sf_v(q, p, r, t, s, u) += el;

                                // G3("prqsut") += g3aab_("pqrstu");
                                rdm_sf_v(p, r, q, s, u, t) += el;
                                rdm_sf_v(p, r, q, t, u, s) -= el;
                                rdm_sf_v(q, r, p, s, u, t) -= el;
                                rdm_sf_v(q, r, p, t, u, s) += el;

                                // G3("rpqust") += g3aab_("pqrstu");
                                rdm_sf_v(r, p, q, u, s, t) += el;
                                rdm_sf_v(r, p, q, u, t, s) -= el;
                                rdm_sf_v(r, q, p, u, s, t) -= el;
                                rdm_sf_v(r, q, p, u, t, s) += el;
                            }
                        }
                    }
                }
            }
        }
    }

    // The abb contribution (2 orbitals or more)
    {
        auto rdm_abb = compute_abb_3rdm(C_left, C_right);
        auto rdm_abb_v = rdm_abb.view();
        for (size_t p{0}; p < norb; ++p) {
            for (size_t q{1}, qr{0}; q < norb; ++q) {
                for (size_t r{0}; r < q; ++r, ++qr) {
                    for (size_t s{0}; s < norb; ++s) {
                        for (size_t t{1}, tu{0}; t < norb; ++t) {
                            for (size_t u{0}; u < t; ++u, ++tu) {
                                const auto el = rdm_abb_v(p, qr, s, tu);
                                // G3("pqrstu") += g3abb_("pqrstu");
                                rdm_sf_v(p, q, r, s, t, u) += el;
                                rdm_sf_v(p, q, r, s, u, t) -= el;
                                rdm_sf_v(p, r, q, s, u, t) += el;
                                rdm_sf_v(p, r, q, s, t, u) -= el;

                                // G3("qprtsu") += g3abb_("pqrstu");
                                rdm_sf_v(q, p, r, t, s, u) += el;
                                rdm_sf_v(q, p, r, u, s, t) -= el;
                                rdm_sf_v(r, p, q, t, s, u) -= el;
                                rdm_sf_v(r, p, q, u, s, t) += el;

                                // G3("qrptus") += g3abb_("pqrstu");
                                rdm_sf_v(q, r, p, t, u, s) += el;
                                rdm_sf_v(q, r, p, u, t, s) -= el;
                                rdm_sf_v(r, q, p, t, u, s) -= el;
                                rdm_sf_v(r, q, p, u, t, s) += el;
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

    // To reduce the  memory footprint, we compute the aaa and bbb contributions in a packed
    // format and one at a time.
    for (auto spin : {Spin::Alpha, Spin::Beta}) {
        auto rdm_sss = compute_sss_3rdm(C_left, C_right, spin);
        auto rdm_sss_v = rdm_sss.view();

        for (size_t p{2}, pqr{0}; p < norb; ++p) {
            for (size_t q{1}; q < p; ++q) {
                for (size_t r{0}; r < q; ++r, ++pqr) {
                    for (size_t s{2}, stu{0}; s < norb; ++s) {
                        for (size_t t{1}; t < s; ++t) {
                            for (size_t u{0}; u < t; ++u, ++stu) {
                                // grab the unique element of the 3-RDM
                                const auto el = rdm_sss_v(pqr, stu);

                                // Place the element in all valid 36 antisymmetric index
                                // permutations
                                rdm_sf_v(p, q, r, s, t, u) += el;
                                rdm_sf_v(p, q, r, s, u, t) -= el;
                                rdm_sf_v(p, q, r, u, s, t) += el;
                                rdm_sf_v(p, q, r, u, t, s) -= el;
                                rdm_sf_v(p, q, r, t, u, s) += el;
                                rdm_sf_v(p, q, r, t, s, u) -= el;

                                rdm_sf_v(p, r, q, s, t, u) -= el;
                                rdm_sf_v(p, r, q, s, u, t) += el;
                                rdm_sf_v(p, r, q, u, s, t) -= el;
                                rdm_sf_v(p, r, q, u, t, s) += el;
                                rdm_sf_v(p, r, q, t, u, s) -= el;
                                rdm_sf_v(p, r, q, t, s, u) += el;

                                rdm_sf_v(r, p, q, s, t, u) += el;
                                rdm_sf_v(r, p, q, s, u, t) -= el;
                                rdm_sf_v(r, p, q, u, s, t) += el;
                                rdm_sf_v(r, p, q, u, t, s) -= el;
                                rdm_sf_v(r, p, q, t, u, s) += el;
                                rdm_sf_v(r, p, q, t, s, u) -= el;

                                rdm_sf_v(r, q, p, s, t, u) -= el;
                                rdm_sf_v(r, q, p, s, u, t) += el;
                                rdm_sf_v(r, q, p, u, s, t) -= el;
                                rdm_sf_v(r, q, p, u, t, s) += el;
                                rdm_sf_v(r, q, p, t, u, s) -= el;
                                rdm_sf_v(r, q, p, t, s, u) += el;

                                rdm_sf_v(q, r, p, s, t, u) += el;
                                rdm_sf_v(q, r, p, s, u, t) -= el;
                                rdm_sf_v(q, r, p, u, s, t) += el;
                                rdm_sf_v(q, r, p, u, t, s) -= el;
                                rdm_sf_v(q, r, p, t, u, s) += el;
                                rdm_sf_v(q, r, p, t, s, u) -= el;

                                rdm_sf_v(q, p, r, s, t, u) -= el;
                                rdm_sf_v(q, p, r, s, u, t) += el;
                                rdm_sf_v(q, p, r, u, s, t) -= el;
                                rdm_sf_v(q, p, r, u, t, s) += el;
                                rdm_sf_v(q, p, r, t, u, s) -= el;
                                rdm_sf_v(q, p, r, t, s, u) += el;
                            }
                        }
                    }
                }
            }
        }
    }

    return rdm_sf;
}

np_tensor6 CISigmaBuilder::compute_sf_3cumulant(np_vector C_left, np_vector C_right) const {
    // Compute the spin-free 1-RDM
    auto sf_1rdm = compute_sf_1rdm(C_left, C_right);
    // Compute the spin-free 2-RDM
    auto sf_2rdm = compute_sf_2rdm(C_left, C_right);
    // Compute the spin-free 3-RDM (this will hold the cumulant)
    auto L3 = compute_sf_3rdm(C_left, C_right);

    auto G1_v = sf_1rdm.view();
    auto G2_v = sf_2rdm.view();
    auto L3_v = L3.view();

    const auto norb = lists_.norb();
    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s) {
                    for (size_t t{0}; t < norb; ++t) {
                        for (size_t u{0}; u < norb; ++u) {
                            L3_v(p, q, r, s, t, u) += -G1_v(p, s) * G2_v(q, r, t, u) -
                                                      G1_v(q, t) * G2_v(p, r, s, u) -
                                                      G1_v(r, u) * G2_v(p, q, s, t) +
                                                      0.5 * G1_v(p, t) * G2_v(q, r, s, u) +
                                                      0.5 * G1_v(p, u) * G2_v(q, r, t, s) +
                                                      0.5 * G1_v(q, s) * G2_v(p, r, t, u) +
                                                      0.5 * G1_v(q, u) * G2_v(p, r, s, t) +
                                                      0.5 * G1_v(r, s) * G2_v(p, q, u, t) +
                                                      0.5 * G1_v(r, t) * G2_v(p, q, s, u) +
                                                      2.0 * G1_v(p, s) * G1_v(q, t) * G1_v(r, u) -
                                                      G1_v(p, s) * G1_v(q, u) * G1_v(r, t) -
                                                      G1_v(p, u) * G1_v(q, t) * G1_v(r, s) -
                                                      G1_v(p, t) * G1_v(q, s) * G1_v(r, u) +
                                                      0.5 * G1_v(p, t) * G1_v(q, u) * G1_v(r, s) +
                                                      0.5 * G1_v(p, u) * G1_v(q, s) * G1_v(r, t);
                        }
                    }
                }
            }
        }
    }
    return L3;
}

} // namespace forte2

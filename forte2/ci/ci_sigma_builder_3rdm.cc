#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_3rdm_aaa_same_irrep(np_vector C_left, np_vector C_right,
                                                      bool alfa) const {
    local_timer timer;

    const size_t norb = lists_.norb();
    const size_t ntriplets = (norb * (norb - 1) * (norb - 2)) / 6;

    auto rdm = make_zeros<nb::numpy, double, 2>({ntriplets, ntriplets});

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((alfa and (na < 3)) or ((!alfa) and (nb < 3)))
        return rdm;

    auto Cl_span = vector::as_span(C_left);
    auto Cr_span = vector::as_span(C_right);

    auto rdm_data = rdm.data();
    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    int num_3h_classes =
        alfa ? lists_.alfa_address_3h()->nclasses() : lists_.beta_address_3h()->nclasses();

    for (int class_K = 0; class_K < num_3h_classes; ++class_K) {
        size_t maxK = alfa ? lists_.alfa_address_3h()->strpcls(class_K)
                           : lists_.beta_address_3h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.detpblk(nI) == 0)
                continue;

            auto tr = gather_block(Cr_span, TR, alfa, lists_, class_Ia, class_Ib);

            for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                    continue;
                if (lists_.detpblk(nJ) == 0)
                    continue;

                const size_t maxL =
                    alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

                if (maxL > 0) {
                    // Get a pointer to the correct block of matrix C
                    auto tl = gather_block(Cl_span, TL, alfa, lists_, class_Ja, class_Jb);

                    for (size_t K{0}; K < maxK; ++K) {
                        auto& Krlist = alfa ? lists_.get_alfa_3h_list(class_K, K, class_Ia)
                                            : lists_.get_beta_3h_list(class_K, K, class_Ib);
                        auto& Kllist = alfa ? lists_.get_alfa_3h_list(class_K, K, class_Ja)
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

np_tensor4 CISigmaBuilder::compute_3rdm_aab_same_irrep(np_vector C_left, np_vector C_right) const {
    local_timer timer;

    const size_t norb = lists_.norb();
    // the number of orbital pairs i > j of the same spin
    const size_t npair = (norb * (norb - 1)) / 2;

    auto rdm = make_zeros<nb::numpy, double, 4>({npair, norb, npair, norb});

    auto stride1 = norb;
    auto stride2 = stride1 * npair;
    auto stride3 = stride2 * norb;

    auto index = [stride1, stride2, stride3](size_t pq, size_t r, size_t st, size_t u) {
        return pq * stride3 + r * stride2 + st * stride1 + u;
    };

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((na < 2) or (nb < 1))
        return rdm;

    auto Cl_span = vector::as_span(C_left);
    auto Cr_span = vector::as_span(C_right);

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
                if (lists_.detpblk(nI) == 0)
                    continue;

                const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                const auto Cr_offset = lists_.block_offset(nI);

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (lists_.detpblk(nJ) == 0)
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

np_tensor4 CISigmaBuilder::compute_3rdm_abb_same_irrep(np_vector C_left, np_vector C_right) const {
    local_timer timer;

    const size_t norb = lists_.norb();
    // the number of orbital pairs i > j of the same spin
    const size_t npair = (norb * (norb - 1)) / 2;

    auto rdm = make_zeros<nb::numpy, double, 4>({norb, npair, norb, npair});

    auto stride1 = npair;
    auto stride2 = stride1 * norb;
    auto stride3 = stride2 * npair;

    auto index = [stride1, stride2, stride3](size_t p, size_t qr, size_t s, size_t tu) {
        return p * stride3 + qr * stride2 + s * stride1 + tu;
    };

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((na < 1) or (nb < 2))
        return rdm;

    auto Cl_span = vector::as_span(C_left);
    auto Cr_span = vector::as_span(C_right);

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
                if (lists_.detpblk(nI) == 0)
                    continue;

                const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                const auto Cr_offset = lists_.block_offset(nI);

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (lists_.detpblk(nJ) == 0)
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

// np_tensor6 CISigmaBuilder::compute_sf_3rdm_same_irrep(np_vector C_left, np_vector C_right) {
//     auto norb = lists_.norb();
//     auto rdm_sf = make_zeros<nb::numpy, double, 6>({norb, norb, norb, norb, norb, norb});

//     auto rdm_aaa = compute_3rdm_aaa_same_irrep_full(C_left, C_right, true);
//     auto rdm_bbb = compute_3rdm_aaa_same_irrep_full(C_left, C_right, false);
//     auto rdm_aab = compute_3rdm_aab_same_irrep(C_left, C_right);
//     auto rdm_abb = compute_3rdm_abb_same_irrep(C_left, C_right);

//     auto rdm_sf_v = rdm_sf.view();
//     auto rdm_aaa_v = rdm_aaa.view();
//     auto rdm_bbb_v = rdm_bbb.view();
//     auto rdm_aab_v = rdm_aab.view();
//     auto rdm_abb_v = rdm_abb.view();

//     for (size_t p{2}, pqr{0}; p < norb; ++p) {
//         for (size_t q{1}; q < p; ++q) {
//             for (size_t r{0}; r < q; ++r, ++pqr) {
//                 for (size_t s{2}, stu{0}; s < norb; ++s) {
//                     for (size_t t{1}; t < s; ++t) {
//                         for (size_t u{0}; u < t; ++u, ++stu) {
//                             const auto el = rdm_aaa_v(pqr, stu) + rdm_bbb_v(pqr, stu);
//                             rdm_sf_v(p, q, r, s, t, u) += el;
//                             rdm_sf_v(p, q, r, s, u, t) -= el;
//                             rdm_sf_v(p, q, r, u, s, t) += el;
//                             rdm_sf_v(p, q, r, u, t, s) -= el;
//                             rdm_sf_v(p, q, r, t, u, s) += el;
//                             rdm_sf_v(p, q, r, t, s, u) -= el;

//                             rdm_sf_v(p, r, q, s, t, u) -= el;
//                             rdm_sf_v(p, r, q, s, u, t) += el;
//                             rdm_sf_v(p, r, q, u, s, t) -= el;
//                             rdm_sf_v(p, r, q, u, t, s) += el;
//                             rdm_sf_v(p, r, q, t, u, s) -= el;
//                             rdm_sf_v(p, r, q, t, s, u) += el;

//                             rdm_sf_v(r, p, q, s, t, u) += el;
//                             rdm_sf_v(r, p, q, s, u, t) -= el;
//                             rdm_sf_v(r, p, q, u, s, t) += el;
//                             rdm_sf_v(r, p, q, u, t, s) -= el;
//                             rdm_sf_v(r, p, q, t, u, s) += el;
//                             rdm_sf_v(r, p, q, t, s, u) -= el;

//                             rdm_sf_v(r, q, p, s, t, u) -= el;
//                             rdm_sf_v(r, q, p, s, u, t) += el;
//                             rdm_sf_v(r, q, p, u, s, t) -= el;
//                             rdm_sf_v(r, q, p, u, t, s) += el;
//                             rdm_sf_v(r, q, p, t, u, s) -= el;
//                             rdm_sf_v(r, q, p, t, s, u) += el;

//                             rdm_sf_v(q, r, p, s, t, u) += el;
//                             rdm_sf_v(q, r, p, s, u, t) -= el;
//                             rdm_sf_v(q, r, p, u, s, t) += el;
//                             rdm_sf_v(q, r, p, u, t, s) -= el;
//                             rdm_sf_v(q, r, p, t, u, s) += el;
//                             rdm_sf_v(q, r, p, t, s, u) -= el;

//                             rdm_sf_v(q, p, r, s, t, u) -= el;
//                             rdm_sf_v(q, p, r, s, u, t) += el;
//                             rdm_sf_v(q, p, r, u, s, t) -= el;
//                             rdm_sf_v(q, p, r, u, t, s) += el;
//                             rdm_sf_v(q, p, r, t, u, s) -= el;
//                             rdm_sf_v(q, p, r, t, s, u) += el;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     for (size_t p{1}, pq{0}; p < norb; ++p) {
//         for (size_t q{0}; q < p; ++q, ++pq) {
//             for (size_t r{0}; r < norb; ++r) {
//                 for (size_t s{1}, st{0}; s < norb; ++s) {
//                     for (size_t t{0}; t < s; ++t, ++st) {
//                         for (size_t u{0}; u < norb; ++u) {
//                             const auto el = rdm_aab_v(pq, r, st, u);
//                             // G3("pqrstu") += g3aab_("pqrstu");
//                             rdm_sf_v(p, q, r, s, t, u) += el;
//                             rdm_sf_v(p, q, r, t, s, u) -= el;
//                             rdm_sf_v(q, p, r, s, t, u) -= el;
//                             rdm_sf_v(q, p, r, t, s, u) += el;

//                             // G3("prqsut") += g3aab_("pqrstu");
//                             rdm_sf_v(p, r, q, s, u, t) += el;
//                             rdm_sf_v(p, r, q, t, u, s) -= el;
//                             rdm_sf_v(q, r, p, s, u, t) -= el;
//                             rdm_sf_v(q, r, p, t, u, s) += el;

//                             // G3("rpqust") += g3aab_("pqrstu");
//                             rdm_sf_v(r, p, q, u, s, t) += el;
//                             rdm_sf_v(r, p, q, t, s, u) -= el;
//                             rdm_sf_v(r, q, p, u, s, t) -= el;
//                             rdm_sf_v(r, q, p, t, s, u) += el;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     for (size_t p{0}; p < norb; ++p) {
//         for (size_t q{1}, qr{0}; q < norb; ++q) {
//             for (size_t r{0}; r < q; ++r, ++qr) {
//                 for (size_t s{0}; s < norb; ++s) {
//                     for (size_t t{1}, tu{0}; t < norb; ++t) {
//                         for (size_t u{0}; u < t; ++u, ++tu) {
//                             const auto el = rdm_abb_v(p, qr, s, tu);
//                             // G3("pqrstu") += g3abb_("pqrstu");
//                             rdm_sf_v(p, q, r, s, t, u) += el;
//                             rdm_sf_v(p, q, r, s, u, t) -= el;
//                             rdm_sf_v(p, r, q, s, t, u) -= el;
//                             rdm_sf_v(p, r, q, s, u, t) += el;

//                             // G3("qprtsu") += g3abb_("pqrstu");
//                             rdm_sf_v(q, p, r, t, s, u) += el;
//                             rdm_sf_v(q, p, r, u, s, t) -= el;
//                             rdm_sf_v(r, p, q, t, s, u) -= el;
//                             rdm_sf_v(r, p, q, u, s, t) += el;

//                             // G3("qrptus") += g3abb_("pqrstu");
//                             rdm_sf_v(q, r, p, t, u, s) += el;
//                             rdm_sf_v(q, r, p, u, t, s) -= el;
//                             rdm_sf_v(r, q, p, t, u, s) -= el;
//                             rdm_sf_v(r, q, p, u, t, s) += el;
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     return rdm_sf;
// }

// ambit::Tensor RDMs::SF_L3() const {
//     _test_rdm_level(3, "SF_L3");
//     timer t("make_cumulant_L3");

//     auto G1 = SF_G1();
//     auto G2 = SF_G2();
//     auto L3 = SF_G3().clone();

//     L3("pqrstu") -= G1("ps") * G2("qrtu");
//     L3("pqrstu") -= G1("qt") * G2("prsu");
//     L3("pqrstu") -= G1("ru") * G2("pqst");

//     L3("pqrstu") += 0.5 * G1("pt") * G2("qrsu");
//     L3("pqrstu") += 0.5 * G1("pu") * G2("qrts");

//     L3("pqrstu") += 0.5 * G1("qs") * G2("prtu");
//     L3("pqrstu") += 0.5 * G1("qu") * G2("prst");

//     L3("pqrstu") += 0.5 * G1("rs") * G2("pqut");
//     L3("pqrstu") += 0.5 * G1("rt") * G2("pqsu");

//     L3("pqrstu") += 2.0 * G1("ps") * G1("qt") * G1("ru");

//     L3("pqrstu") -= G1("ps") * G1("qu") * G1("rt");
//     L3("pqrstu") -= G1("pu") * G1("qt") * G1("rs");
//     L3("pqrstu") -= G1("pt") * G1("qs") * G1("ru");

//     L3("pqrstu") += 0.5 * G1("pt") * G1("qu") * G1("rs");
//     L3("pqrstu") += 0.5 * G1("pu") * G1("qs") * G1("rt");

//     L3.set_name("SF_L3");
//     return L3;
// }

} // namespace forte2

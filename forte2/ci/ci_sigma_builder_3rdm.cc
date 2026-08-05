#include <limits>
#include <stdexcept>

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

    size_t max_composite_K = 0;
    for (int class_Ka = 0; class_Ka < num_2h_class_Ka; ++class_Ka) {
        const size_t maxKa = lists_.alpha_address_2h()->strpcls(class_Ka);
        for (int class_Kb = 0; class_Kb < num_1h_class_Kb; ++class_Kb) {
            const size_t maxKb = lists_.beta_address_1h()->strpcls(class_Kb);
            if ((maxKa == 0) or (maxKb == 0))
                continue;
            if (maxKa > std::numeric_limits<size_t>::max() / maxKb) {
                throw std::overflow_error("The composite AAB 3-RDM hole dimension is too large.");
            }
            max_composite_K = std::max(max_composite_K, maxKa * maxKb);
        }
    }
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

                // Gather the (signed) right coefficients into B_R[(xy*norb+z), K]
                for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                    if (lists_.block_size(nI) == 0)
                        continue;
                    const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                    const auto Cr_offset = lists_.block_offset(nI);
                    const auto& Kb_right_list = lists_.get_beta_1h_list2(class_Kb, class_Ib);
                    if (Kb_right_list.empty())
                        continue;
                    for (size_t Kidx = 0; Kidx < Kdim; ++Kidx) {
                        const size_t K = Kblock_start + Kidx;
                        const size_t Ka = K / maxKb;
                        const size_t Kb = K % maxKb;
                        const auto& Ka_right_list =
                            lists_.get_alpha_2h_list(class_Ka, Ka, class_Ia);
                        if (Ka_right_list.empty())
                            continue;
                        const auto& KbR = Kb_right_list[Kb];
                        for (const auto& [sign_xy, x, y, Ia] : Ka_right_list) {
                            const auto xy_index = pair_index_gt<size_t>(x, y);
                            const size_t row_xy = xy_index * norb;
                            const auto Cr_Ia_offset = Cr_offset + Ia * maxIb;
                            for (const auto& [sign_z, z, Ib] : KbR) {
                                Kblock2[(row_xy + z) * Kdim + Kidx] =
                                    sign_xy * sign_z * Cr_span[Cr_Ia_offset + Ib];
                            }
                        }
                    }
                }

                // Gather the (signed) left coefficients into B_L[(uv*norb+w), K]
                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (lists_.block_size(nJ) == 0)
                        continue;
                    const auto maxJb = lists_.beta_address()->strpcls(class_Jb);
                    const auto Cl_offset = lists_.block_offset(nJ);
                    const auto& Kb_left_list = lists_.get_beta_1h_list2(class_Kb, class_Jb);
                    if (Kb_left_list.empty())
                        continue;
                    for (size_t Kidx = 0; Kidx < Kdim; ++Kidx) {
                        const size_t K = Kblock_start + Kidx;
                        const size_t Ka = K / maxKb;
                        const size_t Kb = K % maxKb;
                        const auto& Ka_left_list = lists_.get_alpha_2h_list(class_Ka, Ka, class_Ja);
                        if (Ka_left_list.empty())
                            continue;
                        const auto& KbL = Kb_left_list[Kb];
                        for (const auto& [sign_uv, u, v, Ja] : Ka_left_list) {
                            const auto uv_index = pair_index_gt<size_t>(u, v);
                            const size_t row_uv = uv_index * norb;
                            const auto Cl_Ja_offset = Cl_offset + Ja * maxJb;
                            for (const auto& [sign_w, w, Jb] : KbL) {
                                Kblock1[(row_uv + w) * Kdim + Kidx] =
                                    sign_uv * sign_w * Cl_span[Cl_Ja_offset + Jb];
                            }
                        }
                    }
                }

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

    size_t max_composite_K = 0;
    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        const size_t maxKa = lists_.alpha_address_1h()->strpcls(class_Ka);
        for (int class_Kb = 0; class_Kb < num_2h_class_Kb; ++class_Kb) {
            const size_t maxKb = lists_.beta_address_2h()->strpcls(class_Kb);
            if ((maxKa == 0) or (maxKb == 0))
                continue;
            if (maxKa > std::numeric_limits<size_t>::max() / maxKb) {
                throw std::overflow_error("The composite ABB 3-RDM hole dimension is too large.");
            }
            max_composite_K = std::max(max_composite_K, maxKa * maxKb);
        }
    }
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

                // Gather the (signed) right coefficients into B_R[(x*npair+yz), K]
                for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                    if (lists_.block_size(nI) == 0)
                        continue;
                    const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                    const auto Cr_offset = lists_.block_offset(nI);
                    const auto& Ka_right_list = lists_.get_alpha_1h_list2(class_Ka, class_Ia);
                    if (Ka_right_list.empty())
                        continue;
                    for (size_t Kidx = 0; Kidx < Kdim; ++Kidx) {
                        const size_t K = Kblock_start + Kidx;
                        const size_t Ka = K / maxKb;
                        const size_t Kb = K % maxKb;
                        const auto& Kb_right_list = lists_.get_beta_2h_list(class_Kb, Kb, class_Ib);
                        if (Kb_right_list.empty())
                            continue;
                        const auto& KaR = Ka_right_list[Ka];
                        for (const auto& [sign_x, x, Ia] : KaR) {
                            const size_t row_x = x * npair;
                            const auto Cr_Ia_offset = Cr_offset + Ia * maxIb;
                            for (const auto& [sign_yz, y, z, Ib] : Kb_right_list) {
                                const auto yz_index = pair_index_gt<size_t>(y, z);
                                Kblock2[(row_x + yz_index) * Kdim + Kidx] =
                                    sign_x * sign_yz * Cr_span[Cr_Ia_offset + Ib];
                            }
                        }
                    }
                }

                // Gather the (signed) left coefficients into B_L[(u*npair+vw), K]
                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (lists_.block_size(nJ) == 0)
                        continue;
                    const auto maxJb = lists_.beta_address()->strpcls(class_Jb);
                    const auto Cl_offset = lists_.block_offset(nJ);
                    const auto& Ka_left_list = lists_.get_alpha_1h_list2(class_Ka, class_Ja);
                    if (Ka_left_list.empty())
                        continue;
                    for (size_t Kidx = 0; Kidx < Kdim; ++Kidx) {
                        const size_t K = Kblock_start + Kidx;
                        const size_t Ka = K / maxKb;
                        const size_t Kb = K % maxKb;
                        const auto& Kb_left_list = lists_.get_beta_2h_list(class_Kb, Kb, class_Jb);
                        if (Kb_left_list.empty())
                            continue;
                        const auto& KaL = Ka_left_list[Ka];
                        for (const auto& [sign_u, u, Ja] : KaL) {
                            const size_t row_u = u * npair;
                            const auto Cl_Ja_offset = Cl_offset + Ja * maxJb;
                            for (const auto& [sign_vw, v, w, Jb] : Kb_left_list) {
                                const auto vw_index = pair_index_gt<size_t>(v, w);
                                Kblock1[(row_u + vw_index) * Kdim + Kidx] =
                                    sign_u * sign_vw * Cl_span[Cl_Ja_offset + Jb];
                            }
                        }
                    }
                }

                matrix_product('N', 'T', M, M, Kdim, 1.0, Kblock1.data(), Kdim, Kblock2.data(),
                               Kdim, 1.0, rdm_data, M);
                Kblock_start += Kdim;
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

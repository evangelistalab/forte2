#include <iostream>
#include <iomanip>

#include "helpers/timer.hpp"
#include "helpers/np_vector_functions.h"
#include "helpers/np_matrix_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"
#include "helpers/logger.h"

#include "ci_sigma_builder.h"

namespace forte2 {

double gather_a_time = 0.0;
double gather_b_time = 0.0;
double gather_b_time_read = 0.0;
double dgemm_time = 0.0;
double scatter_a_time = 0.0;
double scatter_b_time = 0.0;
double scatter_b_time_write = 0.0;
double transpose_1_time = 0.0;
double transpose_2_time = 0.0;

CISigmaBuilder::CISigmaBuilder(const CIStrings& lists, double E, np_matrix& H, np_tensor4& V,
                               int log_level)
    : lists_(lists), E_(E), H_(H), V_(V), slater_rules_(lists.norb(), E, H, V),
      log_level_(log_level) {
    // Find the size of the largest symmetry block
    size_t max_size = 0;
    for (auto const& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        max_size = std::max(lists.block_size(nI), max_size);
    }

    LOG(log_level_) << "\nAllocating CI temporary buffers of size 2 x " << max_size << " ("
                    << 2 * max_size * sizeof(double) / (1024 * 1024) << " MB).\n";

    // Resize the TR and TL vectors to the maximum block size
    TR.resize(max_size);
    TL.resize(max_size);

    set_Hamiltonian(E, H, V);
}

void CISigmaBuilder::set_algorithm(const std::string& algorithm) {
    if (algorithm == "kh" or algorithm == "knowles-handy") {
        algorithm_ = CIAlgorithm::Knowles_Handy;
    } else if (algorithm == "hz" or algorithm == "harrison-zarrabian") {
        algorithm_ = CIAlgorithm::Harrison_Zarrabian;
    } else {
        throw std::runtime_error("CI algorithm " + algorithm + " not valid.");
    }
}

std::string CISigmaBuilder::get_algorithm() const {
    switch (algorithm_) {
    case CIAlgorithm::Knowles_Handy:
        return "knowles-handy";
    case CIAlgorithm::Harrison_Zarrabian:
        return "harrison-zarrabian";
    default:
        throw std::runtime_error("Unknown CI algorithm.");
    }
}

void CISigmaBuilder::set_memory(int mb) {
    memory_size_ = mb * 1024 * 1024; // Convert MB to bytes
}

void CISigmaBuilder::set_Hamiltonian(double E, np_matrix H, np_tensor4 V) {
    E_ = E;

    if (H.ndim() != 2) {
        throw std::runtime_error("H must be a 2D matrix.");
    }
    if (H.shape(0) != lists_.norb() || H.shape(1) != lists_.norb()) {
        throw std::runtime_error("H shape does not match the number of orbitals.");
    }
    H_ = H;

    // Initialize the one-electron integrals h_hz and h_kh

    const size_t norb = lists_.norb();
    h_kh.resize(norb * norb);
    h_hz.resize(norb * norb);
    auto h = H.view();
    auto v = V.view();
    for (size_t p = 0; p < norb; ++p) {
        for (size_t q = 0; q < norb; ++q) {
            h_hz[p * norb + q] = h(p, q);
            h_kh[p * norb + q] = h(p, q);
            for (size_t r = 0; r < norb; ++r) {
                h_kh[p * norb + q] -= 0.5 * v(p, q, r, r);
            }
        }
    }

    // Initialize the two-electron integrals v_pr_qs and v_pr_qs_a
    if (V.ndim() != 4) {
        throw std::runtime_error("V must be a 4D tensor.");
    }
    if (V.shape(0) != lists_.norb() || V.shape(1) != lists_.norb() || V.shape(2) != lists_.norb() ||
        V.shape(3) != lists_.norb()) {
        throw std::runtime_error("V shape does not match the number of orbitals.");
    }
    V_ = V;

    const size_t norb2 = norb * norb;
    const size_t npairs = (norb * (norb - 1)) / 2;    // Number of pairs (p, r) with p > r
    const size_t ngeqpairs = (norb * (norb + 1)) / 2; // Number of pairs (p, r) with p >= r
    v_pr_qs.resize(norb2 * norb2);
    v_pr_qs_a.resize(npairs * npairs);
    v_ijkl_hk.resize(ngeqpairs * ngeqpairs);

    // Loop over all pairs (p, r) and (q, s) to fill v_pr_qs
    for (size_t p = 0; p < norb; ++p) {
        for (size_t r = 0; r < norb; ++r) {
            const auto pr_index = p * norb + r;
            for (size_t q = 0; q < norb; ++q) {
                for (size_t s = 0; s < norb; ++s) {
                    const auto qs_index = q * norb + s;
                    v_pr_qs[pr_index * norb2 + qs_index] = v(p, r, q, s);
                }
            }
        }
    }
    // Loop over all pairs (p, r) and (q, s) to fill v_pr_qs_a with p > r and q > s
    for (int p = 1; p < norb; ++p) {
        for (int r = 0; r < p; ++r) {
            const auto pr_index = (p * (p - 1)) / 2 + r;
            for (int q = 1; q < norb; ++q) {
                for (int s = 0; s < q; ++s) {
                    const auto qs_index = pair_index_gt(q, s);
                    v_pr_qs_a[pr_index * npairs + qs_index] = v(p, r, q, s) - v(p, r, s, q);
                }
            }
        }
    }
    // Loop over all pairs (i, j) and (k, l) to fill v_ijkl_hk with i >= j and k >= l
    for (size_t i = 0; i < norb; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            const auto ij_index = pair_index_geq(i, j);
            for (size_t k = 0; k < norb; ++k) {
                for (size_t l = 0; l <= k; ++l) {
                    const auto kl_index = pair_index_geq(k, l);
                    double dij = (i == j ? 2 : 1);
                    double dkl = (k == l ? 2 : 1);
                    v_ijkl_hk[ij_index * ngeqpairs + kl_index] = v(i, k, j, l) / (dij * dkl);
                }
            }
        }
    }
}

void CISigmaBuilder::Hamiltonian(np_vector basis, np_vector sigma) const {
    local_timer t;
    vector::zero(sigma);
    auto b_span = vector::as_span(basis);
    auto s_span = vector::as_span(sigma);

    H0(b_span, s_span);
    if (algorithm_ == CIAlgorithm::Knowles_Handy) {
        H1_kh(b_span, s_span, Spin::Alpha);
        H1_kh(b_span, s_span, Spin::Beta);
        local_timer h_aabb_timer;
        H2_kh(b_span, s_span);
        haabb_timer_ += h_aabb_timer.elapsed_seconds();
    } else {
        H1_hz(b_span, s_span, Spin::Alpha, h_hz);
        H1_hz(b_span, s_span, Spin::Beta, h_hz);

        local_timer h_aabb_timer;
        H2_hz_opposite_spin(b_span, s_span);
        haabb_timer_ += h_aabb_timer.elapsed_seconds();

        local_timer h_aaaa_timer;
        H2_hz_same_spin(b_span, s_span, Spin::Alpha);
        haaaa_timer_ += h_aaaa_timer.elapsed_seconds();

        local_timer h_bbbb_timer;
        H2_hz_same_spin(b_span, s_span, Spin::Beta);
        hbbbb_timer_ += h_bbbb_timer.elapsed_seconds();
    }
    hdiag_timer_ += t.elapsed_seconds();
    build_count_++;
}

void CISigmaBuilder::H0(std::span<double> basis, std::span<double> sigma) const {
    add(basis.size(), E_, basis.data(), 1, sigma.data(), 1);
}

std::span<double> gather_block(std::span<double> source, std::span<double> dest, Spin spin,
                               const CIStrings& lists, int class_Ia, int class_Ib) {
    const auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    const auto offset = lists.block_offset(block_index);
    const auto maxIa = lists.alfa_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);

    if (is_alpha(spin)) {
        std::span<double> dest_span(source.data() + offset, maxIa * maxIb);
        return dest_span;
    }
    for (size_t Ia{0}; Ia < maxIa; ++Ia)
        for (size_t Ib{0}; Ib < maxIb; ++Ib)
            dest[Ib * maxIa + Ia] = source[offset + Ia * maxIb + Ib];
    return dest;
}

void zero_block(std::span<double> dest, Spin spin, const CIStrings& lists, int class_Ia,
                int class_Ib) {
    const auto maxIa = lists.alfa_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);

    if (is_alpha(spin)) {
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                dest[Ia * maxIb + Ib] = 0.0;
    } else {
        for (size_t Ib{0}; Ib < maxIb; ++Ib)
            for (size_t Ia{0}; Ia < maxIa; ++Ia)
                dest[Ib * maxIa + Ia] = 0.0;
    }
}

void scatter_block(std::span<double> source, std::span<double> dest, Spin spin,
                   const CIStrings& lists, int class_Ia, int class_Ib) {
    size_t maxIa = lists.alfa_address()->strpcls(class_Ia);
    size_t maxIb = lists.beta_address()->strpcls(class_Ib);

    auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    auto offset = lists.block_offset(block_index);

    if (is_alpha(spin)) {
        // Add m to C
        for (size_t I{0}, maxI{maxIa * maxIb}; I < maxI; ++I)
            // for (size_t Ib{0}; Ib < maxIb; ++Ib)
            dest[offset + I] += source[I];
    } else {
        // Add m transposed to C
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                dest[offset + Ia * maxIb + Ib] += source[Ib * maxIa + Ia];
    }
}

double CISigmaBuilder::slater_rules_csf(const std::vector<Determinant>& dets,
                                        const CISpinAdapter& spin_adapter, size_t I,
                                        size_t J) const {
    double matrix_element = 0.0;
    for (const auto& [det_add_I, c_I] : spin_adapter.csf(I)) {
        for (const auto& [det_add_J, c_J] : spin_adapter.csf(J)) {
            if (det_add_I == det_add_J) {
                matrix_element += c_I * c_J * slater_rules_.energy(dets[det_add_I]);
            } else if (det_add_I < det_add_J) {
                if (c_I * c_J != 0.0) {
                    matrix_element += 2.0 * c_I * c_J *
                                      slater_rules_.slater_rules(dets[det_add_I], dets[det_add_J]);
                }
            }
        }
    }
    return matrix_element;
}

np_vector CISigmaBuilder::form_Hdiag_csf(const std::vector<Determinant>& dets,
                                         const CISpinAdapter& spin_adapter,
                                         bool spin_adapt_full_preconditioner) const {
    auto Hdiag = make_zeros<nb::numpy, double, 1>({spin_adapter.ncsf()});
    auto Hdiag_view = Hdiag.view();
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    if (spin_adapt_full_preconditioner) {
        for (size_t i{0}, imax{spin_adapter.ncsf()}; i < imax; ++i) {
            double energy = 0.0;
            int I = 0;
            for (const auto& [det_add_I, c_I] : spin_adapter.csf(i)) {
                int J = 0;
                for (const auto& [det_add_J, c_J] : spin_adapter.csf(i)) {
                    if (I == J) {
                        energy += c_I * c_J * slater_rules_.energy(dets[det_add_I]);
                    } else if (I < J) {
                        if (c_I * c_J != 0.0) {
                            energy += 2.0 * c_I * c_J *
                                      slater_rules_.slater_rules(dets[det_add_I], dets[det_add_J]);
                        }
                    }
                    J++;
                }
                I++;
            }
            Hdiag_view(i) = energy;
        }
    } else {
        for (size_t i{0}, imax{spin_adapter.ncsf()}; i < imax; ++i) {
            double energy = 0.0;
            for (const auto& [det_add_I, c_I] : spin_adapter.csf(i)) {
                energy += c_I * c_I * slater_rules_.energy(dets[det_add_I]);
            }
            Hdiag_view(i) = energy;
        }
    }
    return Hdiag;
}

} // namespace forte2

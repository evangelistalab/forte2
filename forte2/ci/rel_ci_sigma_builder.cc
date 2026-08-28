#include <iostream>
#include <iomanip>

#include "helpers/timer.hpp"
#include "helpers/np_vector_functions.h"
#include "helpers/np_matrix_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"
#include "helpers/logger.h"

#include "rel_ci_sigma_builder.h"

namespace forte2 {

RelCISigmaBuilder::RelCISigmaBuilder(const CIStrings& lists, double E, np_matrix_complex& H,
                                     np_tensor4_complex& V, int log_level,
                                     const std::string& algorithm)
    : lists_(lists), E_(E), rel_slater_rules_(lists.norb(), E, H, V), log_level_(log_level) {
    if (algorithm == "hz" or algorithm == "harrison-zarrabian") {
        algorithm_ = CIAlgorithm::Harrison_Zarrabian;
    } else if (algorithm == "kh" or algorithm == "knowles-handy") {
        throw std::runtime_error("Knowles-Handy algorithm is not implemented for "
                                 "RelCISigmaBuilder; use 'hz' (Harrison-Zarrabian).");
    } else {
        throw std::runtime_error("CI algorithm " + algorithm + " not valid.");
    }

    // Two-component (relativistic) CI treats every electron as an alpha spinor, so the beta space
    // is the single vacuum string (nb == 0). The sigma/RDM builders rely on this: the opposite-spin
    // spectator string count is always 1.
    if (lists.nb() != 0)
        throw std::runtime_error("RelCISigmaBuilder requires nb == 0 (two-component CI treats all "
                                 "electrons as alpha spinors).");

    // Find the size of the largest symmetry block
    size_t max_size = 0;
    for (auto const& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        max_size = std::max(lists.block_size(nI), max_size);
    }

    LOG(log_level_) << "\nAllocating CI temporary buffers of size 2 x " << max_size << " ("
                    << 2 * max_size * sizeof(std::complex<double>) / (1024 * 1024) << " MB).\n";

    // Resize the TR and TL vectors to the maximum block size
    TR.resize(max_size);
    TL.resize(max_size);

    set_Hamiltonian(E, H, V);
}

std::string RelCISigmaBuilder::get_algorithm() const {
    switch (algorithm_) {
    case CIAlgorithm::Harrison_Zarrabian:
        return "Harrison-Zarrabian";
    default:
        throw std::runtime_error("Unknown CI algorithm.");
    }
}

void RelCISigmaBuilder::set_memory(int mb) {
    if (mb < 0) {
        throw std::invalid_argument("CI builder memory must be non-negative.");
    }
    memory_size_ = static_cast<size_t>(mb) * 1024 * 1024; // Convert MB to bytes
    std::vector<std::complex<double>>{}.swap(Kblock1_);
    std::vector<std::complex<double>>{}.swap(Kblock2_);
}

void RelCISigmaBuilder::set_Hamiltonian(std::optional<double> E, std::optional<np_matrix_complex> H,
                                        std::optional<np_tensor4_complex> V) {
    if (E) {
        E_ = *E;
    }

    const size_t norb = lists_.norb();

    if (H) {
        if (H->ndim() != 2) {
            throw std::runtime_error("H must be a 2D matrix.");
        }
        if (H->shape(0) != norb || H->shape(1) != norb) {
            throw std::runtime_error("H shape does not match the number of orbitals.");
        }
        update_h_hz(*H);
    }

    if (V) {
        if (V->ndim() != 4) {
            throw std::runtime_error("V must be a 4D tensor.");
        }
        if (V->shape(0) != norb || V->shape(1) != norb || V->shape(2) != norb ||
            V->shape(3) != norb) {
            throw std::runtime_error("V shape does not match the number of orbitals.");
        }
        update_v_hz(*V);
    }

    // optional containers forwarded to slater rules update,
    // where partial updates are also supported
    if (E || H || V) {
        rel_slater_rules_.update_integrals(static_cast<int>(norb), E, H, V);
    }
}

void RelCISigmaBuilder::update_h_hz(np_matrix_complex& H) {
    const size_t norb = lists_.norb();
    h_hz.resize(norb * norb);
    auto h = H.view();
    for (size_t p = 0; p < norb; ++p) {
        for (size_t q = 0; q < norb; ++q) {
            h_hz[p * norb + q] = h(p, q);
        }
    }
}

void RelCISigmaBuilder::update_v_hz(np_tensor4_complex& V) {
    const size_t norb = lists_.norb();
    const size_t npairs = (norb * (norb - 1)) / 2; // Number of pairs (p, r) with p > r
    v_pr_qs.resize(npairs * npairs);
    auto v = V.view();

    // Loop over all pairs (p, r) and (q, s) to fill v_pr_qs with p > r and q > s.
    // V is given in physicist's notation <pq|rs> and antisymmetrized here on the fly.
    for (int p = 1; p < norb; ++p) {
        for (int r = 0; r < p; ++r) {
            const auto pr_index = (p * (p - 1)) / 2 + r;
            for (int q = 1; q < norb; ++q) {
                for (int s = 0; s < q; ++s) {
                    const auto qs_index = pair_index_gt(q, s);
                    v_pr_qs[pr_index * npairs + qs_index] = v(p, r, q, s) - v(p, r, s, q);
                }
            }
        }
    }
}

void RelCISigmaBuilder::Hamiltonian(np_vector_complex basis, np_vector_complex sigma) const {
    vector::zero<std::complex<double>>(sigma);
    auto b_span = vector::as_span<std::complex<double>>(basis);
    auto s_span = vector::as_span<std::complex<double>>(sigma);

    H0(b_span, s_span);
    H1_hz(b_span, s_span, h_hz);
    H2_hz_same_spin(b_span, s_span);
}

void RelCISigmaBuilder::sigma_one_electron(np_vector_complex basis, np_vector_complex sigma) const {
    vector::zero<std::complex<double>>(sigma);
    auto b_span = vector::as_span<std::complex<double>>(basis);
    auto s_span = vector::as_span<std::complex<double>>(sigma);

    H0(b_span, s_span);
    H1_hz(b_span, s_span, h_hz);
}

void RelCISigmaBuilder::sigma_two_electron(np_vector_complex basis, np_vector_complex sigma) const {
    vector::zero<std::complex<double>>(sigma);
    auto b_span = vector::as_span<std::complex<double>>(basis);
    auto s_span = vector::as_span<std::complex<double>>(sigma);

    H2_hz_same_spin(b_span, s_span);
}

void RelCISigmaBuilder::H0(std::span<std::complex<double>> basis,
                           std::span<std::complex<double>> sigma) const {
    add(basis.size(), static_cast<std::complex<double>>(E_), basis.data(), 1, sigma.data(), 1);
}

std::span<std::complex<double>> gather_block(std::span<std::complex<double>> source,
                                             std::span<std::complex<double>> dest, Spin spin,
                                             const CIStrings& lists, int class_Ia, int class_Ib) {
    const auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    const auto offset = lists.block_offset(block_index);
    const auto maxIa = lists.alpha_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);

    if (is_alpha(spin)) {
        std::span<std::complex<double>> dest_span(source.data() + offset, maxIa * maxIb);
        return dest_span;
    }
    for (size_t Ia{0}; Ia < maxIa; ++Ia)
        for (size_t Ib{0}; Ib < maxIb; ++Ib)
            dest[Ib * maxIa + Ia] = source[offset + Ia * maxIb + Ib];
    return dest;
}

void zero_block(std::span<std::complex<double>> dest, Spin spin, const CIStrings& lists,
                int class_Ia, int class_Ib) {
    const auto maxIa = lists.alpha_address()->strpcls(class_Ia);
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

void scatter_block(std::span<std::complex<double>> source, std::span<std::complex<double>> dest,
                   Spin spin, const CIStrings& lists, int class_Ia, int class_Ib) {
    size_t maxIa = lists.alpha_address()->strpcls(class_Ia);
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

np_vector_complex RelCISigmaBuilder::form_Hdiag(const std::vector<Determinant>& dets) const {
    auto Hdiag = make_zeros<nb::numpy, std::complex<double>, 1>({dets.size()});
    auto Hdiag_view = Hdiag.view();
    // Compute the diagonal elements of the Hamiltonian in the determinantal basis
    for (size_t i{0}, imax{dets.size()}; i < imax; ++i) {
        Hdiag_view(i) = rel_slater_rules_.energy(dets[i]);
    }
    return Hdiag;
}

std::complex<double> RelCISigmaBuilder::slater_rules(const std::vector<Determinant>& dets, size_t I,
                                                     size_t J) const {
    double matrix_element = 0.0;
    if (I == J) {
        return rel_slater_rules_.energy(dets[I]);
    } else {
        return rel_slater_rules_.slater_rules(dets[I], dets[J]);
    }
}

} // namespace forte2

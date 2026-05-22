#include "determinant/slater_rules.h"

#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
// Specialized Slater connection functions for alpha and beta spins.

std::optional<std::uint32_t> screen_slater_connection_alpha(const forte2::Determinant& lhs,
                                                            const forte2::Determinant& rhs) {
    return screen_slater_connection_impl<0, forte2::Determinant::storage_words_per_spin>(lhs, rhs);
}

std::optional<std::uint32_t> screen_slater_connection_beta(const forte2::Determinant& lhs,
                                                           const forte2::Determinant& rhs) {
    return screen_slater_connection_impl<forte2::Determinant::storage_words_per_spin,
                                         forte2::Determinant::nwords_>(lhs, rhs);
}

std::tuple<std::size_t, std::size_t> find_single_connection_alpha(const forte2::Determinant& lhs,
                                                                  const forte2::Determinant& rhs) {
    return find_single_connection_impl<0, forte2::Determinant::storage_words_per_spin>(lhs, rhs);
}

std::tuple<std::size_t, std::size_t> find_single_connection_beta(const forte2::Determinant& lhs,
                                                                 const forte2::Determinant& rhs) {
    return find_single_connection_impl<forte2::Determinant::storage_words_per_spin,
                                       forte2::Determinant::nwords_>(lhs, rhs);
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
find_double_connection_alpha(const forte2::Determinant& lhs, const forte2::Determinant& rhs) {
    return find_double_connection_impl<0, forte2::Determinant::storage_words_per_spin>(lhs, rhs);
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
find_double_connection_beta(const forte2::Determinant& lhs, const forte2::Determinant& rhs) {
    return find_double_connection_impl<forte2::Determinant::storage_words_per_spin,
                                       forte2::Determinant::nwords_>(lhs, rhs);
}
} // namespace

namespace forte2 {

SlaterRules::SlaterRules(int norb, double scalar_energy, np_matrix one_electron_integrals,
                         np_tensor4 two_electron_integrals)
    : norb_(norb), norb2_(norb * norb), norb3_(norb * norb * norb), scalar_energy_(scalar_energy) {
    if (norb < 0) {
        throw std::invalid_argument("SlaterRules: norb must be non-negative, got " +
                                    std::to_string(norb));
    }

    // Precompute the one-electron, Coulomb and Exchange integrals
    h_.resize(norb_ * norb_);
    J_.resize(norb_ * norb_);
    JK_.resize(norb_ * norb_);
    f_J_.resize(norb_ * norb_ * norb_);
    f_JK_.resize(norb_ * norb_ * norb_);
    v_.resize(norb_ * norb_ * norb_ * norb_);
    va_.resize(norb_ * norb_ * norb_ * norb_);
    auto h_view = one_electron_integrals.view();
    auto v_view = two_electron_integrals.view();

    for (std::size_t p = 0; p < norb_; ++p) {
        for (std::size_t q = 0; q < norb_; ++q) {
            h_[p * norb_ + q] = h_view(p, q);                             // <p|h|q>
            J_[p * norb_ + q] = v_view(p, q, p, q);                       // <pq|pq>
            JK_[p * norb_ + q] = v_view(p, q, p, q) - v_view(p, q, q, p); // <pq|pq> - <pq|qp>
            for (std::size_t r = 0; r < norb_; ++r) {
                f_J_[p * norb2_ + q * norb_ + r] = v_view(p, r, q, r); // <pr|qr>
                f_JK_[p * norb2_ + q * norb_ + r] =
                    v_view(p, r, q, r) - v_view(p, r, r, q); // <pr|qr> - <pr|rq>
                for (std::size_t s = 0; s < norb_; ++s) {
                    v_[p * norb3_ + q * norb2_ + r * norb_ + s] = v_view(p, q, r, s); // <pq|rs>
                    va_[p * norb3_ + q * norb2_ + r * norb_ + s] =
                        v_view(p, q, r, s) - v_view(p, q, s, r); // <pq||rs> = <pq|rs> - <pq|sr>
                }
            }
        }
    }
}

double SlaterRules::energy(const Determinant& det) const {
    double energy = scalar_energy_;

    det.for_each_a_occ([&](size_t p) {
        energy += h(p, p);

        det.for_each_a_occ([&](size_t q) {
            if (q >= p) {
                return false;
            }
            energy += JK(q, p); // <pq|pq> - <pq|qp>
            return true;
        });

        det.for_each_b_occ([&](size_t q) {
            energy += J(p, q); // <pq|pq>
            return true;
        });
        return true;
    });

    det.for_each_b_occ([&](size_t p) {
        energy += h(p, p);

        det.for_each_b_occ([&](size_t q) {
            if (q >= p) {
                return false;
            }
            energy += JK(q, p); // <pq|pq> - <pq|qp>
            return true;
        });
        return true;
    });

    return energy;
}

// The same-spin loop iterates over all alpha occupations of `d`, not just the lhs and rhs
// intersection used in the textbook Slater rule. The extra particle-orbital term contributes
// f_JK(i, a, a) = <ia||aa> = 0 by antisymmetry of the two-electron integrals, so the result is
// unchanged. Do not copy this shortcut to RelSlaterRules when V is not antisymmetric.
double SlaterRules::singles_coupling_a(size_t i, size_t a, const Determinant& d) const noexcept {
    double coupling = h(i, a);
    d.for_each_a_occ([&](size_t j) { coupling += f_JK(i, a, j); });
    d.for_each_b_occ([&](size_t j) { coupling += f_J(i, a, j); });
    return coupling;
}

// The same-spin loop iterates over all beta occupations of `d`, not just the lhs and rhs
// intersection used in the textbook Slater rule. The extra particle-orbital term contributes
// f_JK(i, a, a) = <ia||aa> = 0 by antisymmetry of the two-electron integrals, so the result is
// unchanged. The cross-spin f_J loop does not rely on this cancellation.
double SlaterRules::singles_coupling_b(size_t i, size_t a, const Determinant& d) const noexcept {
    double coupling = h(i, a);
    d.for_each_a_occ([&](size_t j) { coupling += f_J(i, a, j); });
    d.for_each_b_occ([&](size_t j) { coupling += f_JK(i, a, j); });
    return coupling;
}

np_vector SlaterRules::energies(const std::vector<Determinant>& dets) const {
    auto energies = make_zeros<nb::numpy, double, 1>({dets.size()});
    auto energies_view = energies.view();
    for (size_t i{0}; i < dets.size(); ++i) {
        energies_view(i) = energy(dets[i]);
    }
    return energies;
}

double SlaterRules::slater_rules(const Determinant& lhs, const Determinant& rhs) const {
    const auto count_alpha = screen_slater_connection_alpha(lhs, rhs);
    const auto count_beta = screen_slater_connection_beta(lhs, rhs);
    if (!count_alpha.has_value() or !count_beta.has_value()) {
        return 0.0;
    }

    const auto ndiff_alpha = count_alpha.value();
    const auto ndiff_beta = count_beta.value();

    if (ndiff_alpha + ndiff_beta > 4) {
        return 0.0;
    }

    if (ndiff_alpha == 2 and ndiff_beta == 2) {
        const auto [i, a] = find_single_connection_alpha(lhs, rhs);
        const auto [j, b] = find_single_connection_beta(lhs, rhs);
        const double sign = lhs.slater_sign_aa(i, a) * lhs.slater_sign_bb(j, b);
        return sign * v(i, j, a, b); // <ia|jb>
    }

    if (ndiff_alpha == 4 and ndiff_beta == 0) {
        const auto [i, j, a, b] = find_double_connection_alpha(lhs, rhs);
        const double sign = lhs.slater_sign_aaaa(i, j, a, b);
        return sign * va(i, j, a, b); // <ij||ab>
    }

    if (ndiff_alpha == 0 and ndiff_beta == 4) {
        const auto [i, j, a, b] = find_double_connection_beta(lhs, rhs);
        const double sign = lhs.slater_sign_bbbb(i, j, a, b);
        return sign * va(i, j, a, b); // <ij||ab>
    }

    if (ndiff_alpha == 2 and ndiff_beta == 0) {
        const auto [i, a] = find_single_connection_alpha(lhs, rhs);
        const double sign = lhs.slater_sign_aa(i, a);
        return sign * singles_coupling_a(i, a, rhs);
    }

    if (ndiff_alpha == 0 and ndiff_beta == 2) {
        const auto [i, a] = find_single_connection_beta(lhs, rhs);
        const double sign = lhs.slater_sign_bb(i, a);
        return sign * singles_coupling_b(i, a, rhs);
    }

    return energy(lhs);
}
} // namespace forte2

#include "determinant/rel_slater_rules.h"
#include "determinant/determinant_helpers.h"

namespace {

std::optional<int> screen_rel_slater_connection(const forte2::Determinant& lhs,
                                                const forte2::Determinant& rhs) {
    int diff_count{0}; // number of spinors that differ between the two determinants
    for (std::size_t w = 0; w < forte2::Determinant::nwords_; ++w) {
        diff_count += std::popcount(lhs.get_word(w) ^ rhs.get_word(w));
        // early exit if more than 4 differences
        if (diff_count > 4) {
            return std::optional<int>();
        }
    }
    int total_count{0}; //  number of lhs - rhs occupied spinors
    for (std::size_t w = 0; w < forte2::Determinant::nwords_; ++w) {
        total_count += std::popcount(lhs.get_word(w)) - std::popcount(rhs.get_word(w));
    }
    // early exit if the determinants have different numbers of electrons
    if (total_count != 0) {
        return std::optional<int>();
    }
    return diff_count;
}

/// @brief Find the single connection between two determinants, which must differ by exactly one
/// occupied and one unoccupied spinor.
/// @param lhs the left determinant
/// @param rhs the right determinant
/// @return a tuple containing the indices of the connected spinors: (i, a) where i is the index of
/// the occupied spinor in lhs and a is the index of the unoccupied spinor in rhs
std::tuple<std::size_t, std::size_t> find_single_connection(const forte2::Determinant& lhs,
                                                            const forte2::Determinant& rhs) {
    std::size_t i, a; // namespace
    for (std::size_t w = 0; w < forte2::Determinant::nwords_; ++w) {
        const std::uint64_t lhs_word = lhs.get_word(w);
        const std::uint64_t rhs_word = rhs.get_word(w);
        const std::uint64_t lhs_only = lhs_word & ~rhs_word;
        const std::uint64_t rhs_only = rhs_word & ~lhs_word;
        if (lhs_only) {
            i = w * forte2::Determinant::bits_per_word + std::countr_zero(lhs_only);
        }
        if (rhs_only) {
            a = w * forte2::Determinant::bits_per_word + std::countr_zero(rhs_only);
        }
    }
    return {i, a};
}

/// @brief Find the double connection between two determinants, which must differ by exactly two
/// occupied and two unoccupied spinors.
/// @param lhs the left determinant
/// @param rhs the right determinant
/// @return a tuple containing the indices of the connected spinors: (i, j, a, b) where i and j are
/// the indices of the occupied spinors in lhs and a and b are the indices of the unoccupied spinors
/// in rhs
std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
find_double_connection(const forte2::Determinant& lhs, const forte2::Determinant& rhs) {
    constexpr std::size_t not_filled = std::numeric_limits<size_t>::max();
    std::size_t i{not_filled}, j, a{not_filled}, b; // mark i and j as not filled
    for (std::size_t w = 0; w < forte2::Determinant::nwords_; ++w) {
        const std::uint64_t lhs_word = lhs.get_word(w);
        const std::uint64_t rhs_word = rhs.get_word(w);
        std::uint64_t lhs_only = lhs_word & ~rhs_word;
        std::uint64_t rhs_only = rhs_word & ~lhs_word;
        while (lhs_only) {
            if (i == not_filled) {
                i = w * forte2::Determinant::bits_per_word + std::countr_zero(lhs_only);
            } else {
                j = w * forte2::Determinant::bits_per_word + std::countr_zero(lhs_only);
            }
            ui64_clear_lowest_one_bit(lhs_only); // Clear the lowest set bit
        }
        while (rhs_only) {
            if (a == not_filled) {
                a = w * forte2::Determinant::bits_per_word + std::countr_zero(rhs_only);
            } else {
                b = w * forte2::Determinant::bits_per_word + std::countr_zero(rhs_only);
            }
            ui64_clear_lowest_one_bit(rhs_only); // Clear the lowest set bit
        }
    }
    return {i, j, a, b};
}
} // namespace

namespace forte2 {

RelSlaterRules::RelSlaterRules(int nspinor, double scalar_energy,
                               np_matrix_complex one_electron_integrals,
                               np_tensor4_complex two_electron_integrals, bool tei_is_asym)
    : nspinor_(nspinor), scalar_energy_(scalar_energy),
      one_electron_integrals_(one_electron_integrals),
      two_electron_integrals_(two_electron_integrals), tei_is_asym_(tei_is_asym) {}

double RelSlaterRules::energy(const Determinant& det) const {
    std::complex<double> energy = scalar_energy_;
    auto h = one_electron_integrals_.view();
    auto v = two_electron_integrals_.view();
    if (tei_is_asym_) {
        det.for_each_occ([&](size_t p) {
            energy += h(p, p);
            det.for_each_occ([&](size_t q) {
                if (q >= p) {
                    return false;
                }
                energy += v(p, q, p, q); // <pq|pq> - <pq|qp>
                return true;
            });
            return true;
        });
    } else {
        det.for_each_occ([&](size_t p) {
            energy += h(p, p);
            det.for_each_occ([&](size_t q) {
                if (q >= p) {
                    return false;
                }
                energy += v(p, q, p, q) - v(p, q, q, p); // <pq|pq> - <pq|qp>
                return true;
            });
            return true;
        });
    }

    return energy.real();
}

np_vector RelSlaterRules::energies(const std::vector<Determinant>& dets) const {
    auto energies = make_zeros<nb::numpy, double, 1>({dets.size()});
    auto energies_view = energies.view();
    for (size_t i{0}; i < dets.size(); ++i) {
        energies_view(i) = energy(dets[i]);
    }
    return energies;
}

std::complex<double> RelSlaterRules::slater_rules(const Determinant& lhs,
                                                  const Determinant& rhs) const {
    // Early exit for disconnected pairs or if the determinants have different numbers of electrons
    const auto count = screen_rel_slater_connection(lhs, rhs);
    if (!count.has_value()) {
        return 0.0;
    }
    const int ndiff = count.value();

    if (ndiff == 4) {
        auto v = two_electron_integrals_.view();
        const auto [i, j, a, b] = find_double_connection(lhs, rhs);
        auto v_el = tei_is_asym_ ? v(i, j, a, b) : v(i, j, a, b) - v(i, j, b, a); // <ij||ab>
        const double sign = lhs.slater_sign_aaaa(i, j, a, b);
        return sign * v_el;
    }

    if (ndiff == 2) {
        auto h = one_electron_integrals_.view();
        auto v = two_electron_integrals_.view();
        const auto [i, a] = find_single_connection(lhs, rhs);
        std::complex<double> matrix_element = h(i, a); // <i|a>
        if (tei_is_asym_) {
            lhs.for_each_occ([&](size_t j) {
                matrix_element += v(i, j, a, j); // \sum_j<ij||aj>
            });
        } else {
            lhs.for_each_occ([&](size_t j) {
                matrix_element += v(i, j, a, j) - v(i, j, j, a); // \sum_j<ij||aj>
            });
        }
        const double sign = lhs.slater_sign_aa(i, a);
        return sign * matrix_element;
    }

    if (ndiff == 0) {
        return energy(lhs);
    }

    return 0.0;
}
} // namespace forte2

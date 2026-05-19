#include "determinant/rel_slater_rules.h"
#include "determinant/determinant_helpers.h"

namespace {

/// Holds the connection between two determinants over the stored determinant words.
///
/// The lhs_only arrays contain occupied orbitals in lhs that are unoccupied in rhs. The
/// rhs_only arrays contain the matching occupied orbitals in rhs. At most two indices are stored
/// per spin/direction because Slater rules only need explicit orbital labels through doubles; the
/// counters still record the full popcount so disconnected higher-rank pairs are detected.
struct RelSlaterConnection {
    std::array<std::uint16_t, 2> lhs_only;
    std::array<std::uint16_t, 2> rhs_only;
    std::uint16_t n_lhs_only = 0;
    std::uint16_t n_rhs_only = 0;
};

/// Count set bits in one word and store the first two global bit indices seen.
///
/// Words are scanned in ascending order, so the first two stored entries are also the first two
/// orbital indices. The counter is incremented for every set bit, even after the fixed array is
/// full, because the excitation rank check still needs the complete count.
void collect_connection_bits(std::uint64_t bits, std::size_t base,
                             std::array<std::uint16_t, 2>& indices, std::uint16_t& count) {
    constexpr std::uint16_t max_stored_indices = 2;
    while (bits) {
        if (count < max_stored_indices) {
            indices[count] = static_cast<std::uint16_t>(base + std::countr_zero(bits));
        }
        ++count;
        bits &= bits - 1;
    }
}

/// Build connection indices and popcounts for lhs and rhs.
///
/// The result gives a compact classification of the connection between two determinants:
/// equal determinants, singles, doubles, or disconnected pairs. It avoids storing full
/// temporary bit strings and records only the orbital labels Slater rules need.
RelSlaterConnection build_rel_slater_connection(const forte2::Determinant& lhs,
                                                const forte2::Determinant& rhs) {
    RelSlaterConnection result;
    for (std::size_t word_idx = 0; word_idx < forte2::Determinant::nwords_; ++word_idx) {
        const std::uint64_t lhs_word = lhs.get_word(word_idx);
        const std::uint64_t rhs_word = rhs.get_word(word_idx);

        const std::uint64_t lhs_only = lhs_word & ~rhs_word;
        const std::uint64_t rhs_only = rhs_word & ~lhs_word;

        const std::size_t base = word_idx * forte2::Determinant::bits_per_word;
        collect_connection_bits(lhs_only, base, result.lhs_only, result.n_lhs_only);
        collect_connection_bits(rhs_only, base, result.rhs_only, result.n_rhs_only);

        const int raw_diff_count = result.n_lhs_only + result.n_rhs_only;
        if (raw_diff_count > 4) {
            return result;
        }
    }
    return result;
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
                energy += 0.5 * v(p, q, p, q); // <pq|pq> - <pq|qp>
                return true;
            });
            return true;
        });
    } else {
        det.for_each_occ([&](size_t p) {
            energy += h(p, p);
            det.for_each_occ([&](size_t q) {
                energy += 0.5 * (v(p, q, p, q) - v(p, q, q, p)); // <pq|pq> - <pq|qp>
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
    const auto connection = build_rel_slater_connection(lhs, rhs);

    if ((connection.n_lhs_only != connection.n_rhs_only)) {
        return 0.0;
    }

    const std::uint16_t ndiff = connection.n_lhs_only;
    if (ndiff > 2) {
        return 0.0;
    }

    if (ndiff == 0) {
        return energy(lhs);
    }

    std::complex<double> matrix_element = 0.0;
    auto h = one_electron_integrals_.view();
    auto v = two_electron_integrals_.view();

    if (ndiff == 1) {
        const std::size_t i = connection.lhs_only[0];
        const std::size_t a = connection.rhs_only[0];
        double sign = lhs.slater_sign_aa(i, a);
        matrix_element += h(i, a); // <i|a>
        if (tei_is_asym_) {
            lhs.for_each_occ([&](size_t j) {
                matrix_element += v(i, j, a, j); // \sum_j<ij||aj>
            });
        } else {
            lhs.for_each_occ([&](size_t j) {
                matrix_element += v(i, j, a, j) - v(i, j, j, a); // \sum_j<ij||aj>
            });
        }
        matrix_element *= sign;
    }

    if (ndiff == 2) {
        const std::size_t i = connection.lhs_only[0];
        const std::size_t j = connection.lhs_only[1];
        const std::size_t a = connection.rhs_only[0];
        const std::size_t b = connection.rhs_only[1];
        double sign = lhs.slater_sign_aaaa(i, j, a, b);
        auto v_el = tei_is_asym_ ? v(i, j, a, b) : v(i, j, a, b) - v(i, j, b, a); // <ij||ab>
        matrix_element += sign * v_el;                                            // <ij||ab>
    }

    return matrix_element;
}
} // namespace forte2

#include "determinant/rel_slater_rules.h"
#include "determinant/determinant_helpers.h"

namespace {

/// Holds the connection between two determinants over the stored determinant words.
///
/// The *_lhs_only arrays contain occupied orbitals in lhs that are unoccupied in rhs. The
/// *_rhs_only arrays contain the matching occupied orbitals in rhs. At most two indices are stored
/// per spin/direction because Slater rules only need explicit orbital labels through doubles; the
/// counters still record the full popcount so disconnected higher-rank pairs are detected.
struct RelSlaterConnection {
    std::array<std::size_t, 2> lhs_only{ui64_bit_not_found, ui64_bit_not_found};
    std::array<std::size_t, 2> rhs_only{ui64_bit_not_found, ui64_bit_not_found};
    int n_lhs_only = 0;
    int n_rhs_only = 0;
};

/// Count set bits in one word and store the first two global bit indices seen.
///
/// Words are scanned in ascending order, so the first two stored entries are also the first two
/// orbital indices. The counter is incremented for every set bit, even after the fixed array is
/// full, because the excitation rank check still needs the complete count.
void collect_connection_bits(std::uint64_t bits, std::size_t base,
                             std::array<std::size_t, 2>& indices, int& count) {
    constexpr int max_stored_indices = 2;
    while (bits) {
        if (count < max_stored_indices) {
            indices[count] = base + std::countr_zero(bits);
        }
        ++count;
        bits &= bits - 1;
    }
}

/// Build alpha/beta connection indices and popcounts for lhs and rhs.
///
/// The result gives a compact classification of the connection between two determinants:
/// equal determinants, alpha/beta singles, doubles, or disconnected pairs. It avoids storing full
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

    const int ndiff = connection.n_lhs_only;
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
        const auto i = connection.lhs_only[0];
        const auto a = connection.rhs_only[0];
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
        const auto i = connection.lhs_only[0];
        const auto j = connection.lhs_only[1];
        const auto a = connection.rhs_only[0];
        const auto b = connection.rhs_only[1];
        double sign = lhs.slater_sign_aaaa(i, j, a, b);
        auto v_el = tei_is_asym_ ? v(i, j, a, b) : v(i, j, a, b) - v(i, j, b, a); // <ij||ab>
        matrix_element += sign * v_el;                                            // <ij||ab>
    }

    return matrix_element;
}

// double RelSlaterRules::energy(const Determinant& det) const {
//     std::complex<double> energy = scalar_energy_;

//     auto h = one_electron_integrals_.view();
//     auto v = two_electron_integrals_.view();

//     auto occ = det.get_alpha_occ();
//     if (tei_is_asym_) {
//         for (auto p : occ) {
//             energy += h(p, p); // <p|p>
//             for (auto q : occ) {
//                 energy += 0.5 * v(p, q, p, q); // <pq||pq>
//             }
//         }
//     } else {
//         for (auto p : occ) {
//             energy += h(p, p); // <p|p>
//             for (auto q : occ) {
//                 energy += 0.5 * (v(p, q, p, q) - v(p, q, q, p)); // <pq||pq>
//             }
//         }
//     }

//     return energy.real();
// }

// std::complex<double> RelSlaterRules::slater_rules(const Determinant& lhs,
//                                                   const Determinant& rhs) const {
//     // make sure the two determinants have the same number of electrons
//     if (lhs.count_alpha() != rhs.count_alpha())
//         return 0.0;

//     int ndiff = lhs.symmetric_difference_count(rhs) / 2;
//     if (ndiff > 2)
//         return 0.0;

//     // Slater rule 1 PhiI = PhiJ
//     if (ndiff == 0) {
//         return energy(lhs);
//     }

//     // excitation_connection stores the creation and annihilation operators
//     // that need to be applied to rhs to obtain lhs:
//     // if <LHS|pa^+ qb^+ sa rb|RHS> = +- 1 then excitation_connection = [[s, p], [r, q]]
//     // [[alpha annihilation], [alpha creation],
//     //  [beta annihilation],  [beta creation]]
//     auto connection = excitation_connection(lhs, rhs);

//     std::complex<double> matrix_element = 0.0;
//     auto h = one_electron_integrals_.view();
//     auto v = two_electron_integrals_.view();

//     if (ndiff == 1) {
//         size_t i = connection[0][0];
//         size_t a = connection[1][0];
//         double sign = lhs.slater_sign_aa(i, a);
//         matrix_element += h(i, a); // <i|a>

//         auto occ = lhs.get_alpha_occ();

//         if (tei_is_asym_) {
//             for (auto j : occ) {
//                 matrix_element += v(i, j, a, j); // \sum_j<ij||aj>
//             }
//             matrix_element *= sign;
//         } else {
//             for (auto j : occ) {
//                 matrix_element += v(i, j, a, j) - v(i, j, j, a); // \sum_j<ij||aj>
//             }
//             matrix_element *= sign;
//         }
//     }

//     if (ndiff == 2) {
//         size_t i = connection[0][0];
//         size_t j = connection[0][1];
//         size_t a = connection[1][0];
//         size_t b = connection[1][1];
//         double sign = lhs.slater_sign_aaaa(i, j, a, b);
//         auto v_el = tei_is_asym_ ? v(i, j, a, b) : v(i, j, a, b) - v(i, j, b, a); // <ij||ab>
//         matrix_element += sign * v_el;                                            // <ij||ab>
//     }

//     return matrix_element;
// }

} // namespace forte2

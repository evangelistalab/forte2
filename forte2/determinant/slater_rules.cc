#include "determinant/slater_rules.h"

#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

/// Holds the connection between two determinants over the stored determinant words.
///
/// The *_lhs_only arrays contain occupied orbitals in lhs that are unoccupied in rhs. The
/// *_rhs_only arrays contain the matching occupied orbitals in rhs. At most two indices are stored
/// per spin/direction because Slater rules only need explicit orbital labels through doubles; the
/// counters still record the full popcount so disconnected higher-rank pairs are detected.
struct SlaterConnection {
    std::array<std::size_t, 2> a_lhs_only{ui64_bit_not_found, ui64_bit_not_found};
    std::array<std::size_t, 2> a_rhs_only{ui64_bit_not_found, ui64_bit_not_found};
    std::array<std::size_t, 2> b_lhs_only{ui64_bit_not_found, ui64_bit_not_found};
    std::array<std::size_t, 2> b_rhs_only{ui64_bit_not_found, ui64_bit_not_found};
    int na_lhs_only = 0;
    int na_rhs_only = 0;
    int nb_lhs_only = 0;
    int nb_rhs_only = 0;
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
SlaterConnection build_slater_connection(const forte2::Determinant& lhs,
                                         const forte2::Determinant& rhs) {
    SlaterConnection result;
    for (std::size_t word_idx = 0; word_idx < forte2::Determinant::storage_words_per_spin;
         ++word_idx) {
        const std::uint64_t lhs_a = lhs.get_word(word_idx);
        const std::uint64_t rhs_a = rhs.get_word(word_idx);
        const std::uint64_t lhs_b =
            lhs.get_word(word_idx + forte2::Determinant::storage_words_per_spin);
        const std::uint64_t rhs_b =
            rhs.get_word(word_idx + forte2::Determinant::storage_words_per_spin);

        const std::uint64_t a_lhs_only = lhs_a & ~rhs_a;
        const std::uint64_t a_rhs_only = rhs_a & ~lhs_a;
        const std::uint64_t b_lhs_only = lhs_b & ~rhs_b;
        const std::uint64_t b_rhs_only = rhs_b & ~lhs_b;

        const std::size_t base = word_idx * forte2::Determinant::bits_per_word;
        collect_connection_bits(a_lhs_only, base, result.a_lhs_only, result.na_lhs_only);
        collect_connection_bits(a_rhs_only, base, result.a_rhs_only, result.na_rhs_only);
        collect_connection_bits(b_lhs_only, base, result.b_lhs_only, result.nb_lhs_only);
        collect_connection_bits(b_rhs_only, base, result.b_rhs_only, result.nb_rhs_only);

        const int raw_diff_count =
            result.na_lhs_only + result.na_rhs_only + result.nb_lhs_only + result.nb_rhs_only;
        if (raw_diff_count > 4) {
            return result;
        }
    }
    return result;
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
    const auto connection = build_slater_connection(lhs, rhs);

    if ((connection.na_lhs_only != connection.na_rhs_only) or
        (connection.nb_lhs_only != connection.nb_rhs_only)) {
        return 0.0;
    }

    const int nadiff = connection.na_lhs_only;
    const int nbdiff = connection.nb_lhs_only;
    if (nadiff + nbdiff > 2) {
        return 0.0;
    }

    if (nadiff == 0 and nbdiff == 0) {
        return energy(lhs);
    }

    if (nadiff == 1 and nbdiff == 0) {
        const auto i = connection.a_lhs_only[0];
        const auto j = connection.a_rhs_only[0];
        const double sign = lhs.slater_sign_aa(i, j);
        return sign * singles_coupling_a(i, j, rhs);
    }

    if (nadiff == 0 and nbdiff == 1) {
        const auto i = connection.b_lhs_only[0];
        const auto j = connection.b_rhs_only[0];
        const double sign = lhs.slater_sign_bb(i, j);
        return sign * singles_coupling_b(i, j, rhs);
    }

    if (nadiff == 2 and nbdiff == 0) {
        const auto i = connection.a_lhs_only[0];
        const auto j = connection.a_lhs_only[1];
        const auto k = connection.a_rhs_only[0];
        const auto l = connection.a_rhs_only[1];
        const double sign = lhs.slater_sign_aaaa(i, j, k, l);
        return sign * va(i, j, k, l); // <ij||kl>
    }

    if (nadiff == 0 and nbdiff == 2) {
        const auto i = connection.b_lhs_only[0];
        const auto j = connection.b_lhs_only[1];
        const auto k = connection.b_rhs_only[0];
        const auto l = connection.b_rhs_only[1];
        const double sign = lhs.slater_sign_bbbb(i, j, k, l);
        return sign * va(i, j, k, l); // <ij||kl>
    }

    if (nadiff == 1 and nbdiff == 1) {
        const auto i = connection.a_lhs_only[0];
        const auto j = connection.b_lhs_only[0];
        const auto k = connection.a_rhs_only[0];
        const auto l = connection.b_rhs_only[0];
        const double sign = lhs.slater_sign_aa(i, k) * lhs.slater_sign_bb(j, l);
        return sign * v(i, j, k, l); // <ij|kl>
    }

    return 0.0;
}

double SlaterRules::slater_rules_reference(const Determinant& lhs, const Determinant& rhs) const {
    // we first check that the two determinants have equal Ms
    if ((lhs.count_alpha() != rhs.count_alpha()) or (lhs.count_beta() != rhs.count_beta()))
        return 0.0;

    int nadiff = 0;
    int nbdiff = 0;
    // Count how many differences in mos are there
    for (size_t n = 0; n < norb_; ++n) {
        if (lhs.na(n) != rhs.na(n))
            nadiff++;
        if (lhs.nb(n) != rhs.nb(n))
            nbdiff++;
        if (nadiff + nbdiff > 4)
            return 0.0; // Get out of this as soon as possible
    }
    nadiff /= 2;
    nbdiff /= 2;

    double matrix_element = 0.0;
    // Slater rule 1 PhiI = PhiJ
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element = scalar_energy_;
        for (size_t p = 0; p < norb_; ++p) {
            if (lhs.na(p))
                matrix_element += h(p, p);
            if (lhs.nb(p))
                matrix_element += h(p, p);
            for (size_t q = 0; q < norb_; ++q) {
                if (lhs.na(p) and lhs.na(q))
                    matrix_element += 0.5 * JK(p, q); // <pq|pq> - <pq|qp>
                if (lhs.nb(p) and lhs.nb(q))
                    matrix_element += 0.5 * JK(p, q); // <pq|pq> - <pq|qp>
                if (lhs.na(p) and lhs.nb(q))
                    matrix_element += J(p, q); // <pq|pq>
            }
        }
    }

    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 0)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        for (size_t p = 0; p < norb_; ++p) {
            if ((lhs.na(p) != rhs.na(p)) and lhs.na(p))
                i = p;
            if ((lhs.na(p) != rhs.na(p)) and rhs.na(p))
                j = p;
        }
        // double sign = SlaterSign(I, i, j);
        double sign = lhs.slater_sign_aa(i, j);
        matrix_element = sign * h(i, j);
        for (size_t p = 0; p < norb_; ++p) {
            if (lhs.na(p) and rhs.na(p)) {
                matrix_element += sign * va(i, p, j, p); // <ip|jp> - <ip|pj>
            }
            if (lhs.nb(p) and rhs.nb(p)) {
                matrix_element += sign * v(i, p, j, p); // <ip|jp>
            }
        }
    }
    // Slater rule 2 PhiI = j_b^+ i_b PhiJ
    if ((nadiff == 0) and (nbdiff == 1)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        for (size_t p = 0; p < norb_; ++p) {
            if ((lhs.nb(p) != rhs.nb(p)) and lhs.nb(p))
                i = p;
            if ((lhs.nb(p) != rhs.nb(p)) and rhs.nb(p))
                j = p;
        }
        // double sign = SlaterSign(I, norb_ + i, norb_ + j);
        double sign = lhs.slater_sign_bb(i, j);
        matrix_element = sign * h(i, j); // oei_b_[i * norb_ + j];
        for (size_t p = 0; p < norb_; ++p) {
            if (lhs.na(p) and rhs.na(p)) {
                matrix_element += sign * v(p, i, p, j); // <pi|pj>
            }
            if (lhs.nb(p) and rhs.nb(p)) {
                matrix_element += sign * va(i, p, j, p); // <ip|jp> - <ip|pj>
            }
        }
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 2) and (nbdiff == 0)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        size_t k = 0;
        size_t l = 0;
        bool found_i = false;
        bool found_k = false;
        for (size_t p = 0; p < norb_; ++p) {
            if ((lhs.na(p) != rhs.na(p)) and lhs.na(p)) {
                if (not found_i) {
                    i = p;
                    found_i = true;
                } else {
                    j = p;
                }
            }
            if ((lhs.na(p) != rhs.na(p)) and rhs.na(p)) {
                if (not found_k) {
                    k = p;
                    found_k = true;
                } else {
                    l = p;
                }
            }
        }
        double sign = lhs.slater_sign_aaaa(i, j, k, l);
        matrix_element = sign * va(i, j, k, l); // <ij||kl>
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 0) and (nbdiff == 2)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        size_t k = 0;
        size_t l = 0;
        bool found_i = false;
        bool found_k = false;
        for (size_t p = 0; p < norb_; ++p) {
            if ((lhs.nb(p) != rhs.nb(p)) and lhs.nb(p)) {
                if (not found_i) {
                    i = p;
                    found_i = true;
                } else {
                    j = p;
                }
            }
            if ((lhs.nb(p) != rhs.nb(p)) and rhs.nb(p)) {
                if (not found_k) {
                    k = p;
                    found_k = true;
                } else {
                    l = p;
                }
            }
        }
        double sign = lhs.slater_sign_bbbb(i, j, k, l);
        matrix_element = sign * va(i, j, k, l); // <ij||kl>
    }

    // Slater rule 3 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        size_t k = 0;
        size_t l = 0;
        for (size_t p = 0; p < norb_; ++p) {
            if ((lhs.na(p) != rhs.na(p)) and lhs.na(p))
                i = p;
            if ((lhs.nb(p) != rhs.nb(p)) and lhs.nb(p))
                j = p;
            if ((lhs.na(p) != rhs.na(p)) and rhs.na(p))
                k = p;
            if ((lhs.nb(p) != rhs.nb(p)) and rhs.nb(p))
                l = p;
        }
        double sign = lhs.slater_sign_aa(i, k) * lhs.slater_sign_bb(j, l);
        matrix_element = sign * v(i, j, k, l); // <ij|kl>
    }

    return (matrix_element);
}

} // namespace forte2

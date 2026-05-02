#include "slater_rules.h"

#include <array>
#include <bit>
#include <cstdint>

namespace {

/// Holds the orbital differences between two determinants, restricted to the active
/// spatial-orbital range used by this SlaterRules object.
///
/// The *_lhs_only strings contain occupied orbitals in lhs that are unoccupied in rhs.
/// The *_rhs_only strings contain the matching occupied orbitals in rhs. These are the
/// annihilation and creation orbital sets needed to connect rhs to lhs. The *_common
/// strings contain orbitals occupied in both determinants and are used in single-excitation
/// contractions.
struct ExcitationWords {
    forte2::String a_lhs_only = forte2::String::zero();
    forte2::String a_rhs_only = forte2::String::zero();
    forte2::String b_lhs_only = forte2::String::zero();
    forte2::String b_rhs_only = forte2::String::zero();
    forte2::String a_common = forte2::String::zero();
    forte2::String b_common = forte2::String::zero();
    int na_lhs_only = 0;
    int na_rhs_only = 0;
    int nb_lhs_only = 0;
    int nb_rhs_only = 0;
};

/// Return a word mask that keeps only spatial orbitals below norb.
///
/// Determinants can store more orbitals than a given SlaterRules active space. This mask lets
/// the excitation analysis ignore bits outside the integral arrays without mutating either input
/// determinant. For full words inside the active space the mask is all ones; for words beyond the
/// active space it is zero.
std::uint64_t active_word_mask(std::size_t word_idx, std::size_t norb) {
    constexpr auto bits_per_word = forte2::Determinant::bits_per_word;
    const std::size_t first_bit = word_idx * bits_per_word;
    if (first_bit >= norb) {
        return 0;
    }
    const std::size_t remaining_bits = norb - first_bit;
    if (remaining_bits >= bits_per_word) {
        return ~std::uint64_t(0);
    }
    return (std::uint64_t(1) << remaining_bits) - std::uint64_t(1);
}

/// Return the first two set-bit indices in ascending order.
///
/// SlaterRules only needs the explicit orbital indices for single and double excitations. The
/// caller has already counted the excitation rank, so this helper intentionally stops after two
/// bits and leaves missing entries as ui64_bit_not_found.
std::array<std::size_t, 2> first_two_set_bits(const forte2::String& bit_string) {
    std::array<std::size_t, 2> result{ui64_bit_not_found, ui64_bit_not_found};
    std::size_t n = 0;
    bit_string.for_each_set_bit([&](std::size_t p) {
        if (n < result.size()) {
            result[n++] = p;
        }
        return n < result.size();
    });
    return result;
}

/// Build masked alpha/beta difference words and their popcounts for lhs and rhs.
///
/// The result gives a compact classification of the connection between two determinants:
/// equal determinants, alpha/beta singles, doubles, or disconnected pairs. This avoids scanning
/// each orbital with na()/nb() in the fast Slater-rule path while preserving the original
/// spin-separated excitation cases.
ExcitationWords build_excitation_words(const forte2::Determinant& lhs,
                                       const forte2::Determinant& rhs, std::size_t norb) {
    ExcitationWords result;
    for (std::size_t word_idx = 0; word_idx < forte2::Determinant::nwords_half; ++word_idx) {
        const std::uint64_t mask = active_word_mask(word_idx, norb);
        const std::uint64_t lhs_a = lhs.get_word(word_idx) & mask;
        const std::uint64_t rhs_a = rhs.get_word(word_idx) & mask;
        const std::uint64_t lhs_b =
            lhs.get_word(word_idx + forte2::Determinant::nwords_half) & mask;
        const std::uint64_t rhs_b =
            rhs.get_word(word_idx + forte2::Determinant::nwords_half) & mask;

        const std::uint64_t a_lhs_only = lhs_a & ~rhs_a;
        const std::uint64_t a_rhs_only = rhs_a & ~lhs_a;
        const std::uint64_t b_lhs_only = lhs_b & ~rhs_b;
        const std::uint64_t b_rhs_only = rhs_b & ~lhs_b;

        result.a_lhs_only.set_word(word_idx, a_lhs_only);
        result.a_rhs_only.set_word(word_idx, a_rhs_only);
        result.b_lhs_only.set_word(word_idx, b_lhs_only);
        result.b_rhs_only.set_word(word_idx, b_rhs_only);
        result.a_common.set_word(word_idx, lhs_a & rhs_a);
        result.b_common.set_word(word_idx, lhs_b & rhs_b);

        result.na_lhs_only += std::popcount(a_lhs_only);
        result.na_rhs_only += std::popcount(a_rhs_only);
        result.nb_lhs_only += std::popcount(b_lhs_only);
        result.nb_rhs_only += std::popcount(b_rhs_only);
    }
    return result;
}

} // namespace

namespace forte2 {

SlaterRules::SlaterRules(int norb, double scalar_energy, np_matrix one_electron_integrals,
                         np_tensor4 two_electron_integrals)
    : norb_(norb), norb2_(norb * norb), norb3_(norb * norb * norb),
      scalar_energy_(scalar_energy) {

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

double SlaterRules::singles_coupling_a(size_t i, size_t a, const Determinant& d) const noexcept {
    double coupling = h(i, a);
    d.for_each_a_occ([&](size_t j) {
        coupling += f_JK(i, a, j);
    });
    d.for_each_b_occ([&](size_t j) {
        coupling += f_J(i, a, j);
    });
    return coupling;
}

double SlaterRules::singles_coupling_b(size_t i, size_t a, const Determinant& d) const noexcept {
    double coupling = h(i, a);
    d.for_each_a_occ([&](size_t j) {
        coupling += f_J(i, a, j);
    });
    d.for_each_b_occ([&](size_t j) {
        coupling += f_JK(i, a, j);
    });
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
    const auto excitation = build_excitation_words(lhs, rhs, norb_);

    if ((excitation.na_lhs_only != excitation.na_rhs_only) or
        (excitation.nb_lhs_only != excitation.nb_rhs_only)) {
        return 0.0;
    }

    const int excitation_rank = excitation.na_lhs_only + excitation.nb_lhs_only;
    if (excitation_rank > 2) {
        return 0.0;
    }

    if (excitation_rank == 0) {
        return energy(lhs);
    }

    if ((excitation.na_lhs_only == 1) and (excitation.nb_lhs_only == 0)) {
        const auto lhs_a = first_two_set_bits(excitation.a_lhs_only);
        const auto rhs_a = first_two_set_bits(excitation.a_rhs_only);
        const auto i = lhs_a[0];
        const auto j = rhs_a[0];
        const double sign = lhs.slater_sign_aa(static_cast<int>(i), static_cast<int>(j));
        return sign * singles_coupling_a(i, j, rhs);
    }

    if ((excitation.na_lhs_only == 0) and (excitation.nb_lhs_only == 1)) {
        const auto lhs_b = first_two_set_bits(excitation.b_lhs_only);
        const auto rhs_b = first_two_set_bits(excitation.b_rhs_only);
        const auto i = lhs_b[0];
        const auto j = rhs_b[0];
        const double sign = lhs.slater_sign_bb(static_cast<int>(i), static_cast<int>(j));
        return sign * singles_coupling_b(i, j, rhs);
    }

    if ((excitation.na_lhs_only == 2) and (excitation.nb_lhs_only == 0)) {
        const auto lhs_a = first_two_set_bits(excitation.a_lhs_only);
        const auto rhs_a = first_two_set_bits(excitation.a_rhs_only);
        const auto i = lhs_a[0];
        const auto j = lhs_a[1];
        const auto k = rhs_a[0];
        const auto l = rhs_a[1];
        const double sign = lhs.slater_sign_aaaa(static_cast<int>(i), static_cast<int>(j),
                                                 static_cast<int>(k), static_cast<int>(l));
        return sign * va(i, j, k, l); // <ij||kl>
    }

    if ((excitation.na_lhs_only == 0) and (excitation.nb_lhs_only == 2)) {
        const auto lhs_b = first_two_set_bits(excitation.b_lhs_only);
        const auto rhs_b = first_two_set_bits(excitation.b_rhs_only);
        const auto i = lhs_b[0];
        const auto j = lhs_b[1];
        const auto k = rhs_b[0];
        const auto l = rhs_b[1];
        const double sign = lhs.slater_sign_bbbb(static_cast<int>(i), static_cast<int>(j),
                                                 static_cast<int>(k), static_cast<int>(l));
        return sign * va(i, j, k, l); // <ij||kl>
    }

    if ((excitation.na_lhs_only == 1) and (excitation.nb_lhs_only == 1)) {
        const auto lhs_a = first_two_set_bits(excitation.a_lhs_only);
        const auto rhs_a = first_two_set_bits(excitation.a_rhs_only);
        const auto lhs_b = first_two_set_bits(excitation.b_lhs_only);
        const auto rhs_b = first_two_set_bits(excitation.b_rhs_only);
        const auto i = lhs_a[0];
        const auto j = lhs_b[0];
        const auto k = rhs_a[0];
        const auto l = rhs_b[0];
        const double sign = lhs.slater_sign_aa(static_cast<int>(i), static_cast<int>(k)) *
                            lhs.slater_sign_bb(static_cast<int>(j), static_cast<int>(l));
        return sign * v(i, j, k, l); // <ij|kl>
    }

    return 0.0;
}

double SlaterRules::slater_rules_reference(const Determinant& lhs, const Determinant& rhs) const {
    // we first check that the two determinants have equal Ms
    if ((lhs.count_a() != rhs.count_a()) or (lhs.count_b() != rhs.count_b()))
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
        int i = -1;
        int j = 0;
        int k = -1;
        int l = 0;
        for (size_t p = 0; p < norb_; ++p) {
            if ((lhs.na(p) != rhs.na(p)) and lhs.na(p)) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((lhs.na(p) != rhs.na(p)) and rhs.na(p)) {
                if (k == -1) {
                    k = p;
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
        int i, j, k, l;
        i = -1;
        j = -1;
        k = -1;
        l = -1;
        for (size_t p = 0; p < norb_; ++p) {
            if ((lhs.nb(p) != rhs.nb(p)) and lhs.nb(p)) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((lhs.nb(p) != rhs.nb(p)) and rhs.nb(p)) {
                if (k == -1) {
                    k = p;
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
        int i, j, k, l;
        i = j = k = l = -1;
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

    auto occ = det.get_alfa_occ(nspinor_);
    if (tei_is_asym_) {
        for (auto p : occ) {
            energy += h(p, p); // <p|p>
            for (auto q : occ) {
                energy += 0.5 * v(p, q, p, q); // <pq||pq>
            }
        }
    } else {
        for (auto p : occ) {
            energy += h(p, p); // <p|p>
            for (auto q : occ) {
                energy += 0.5 * (v(p, q, p, q) - v(p, q, q, p)); // <pq||pq>
            }
        }
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
    // make sure the two determinants have the same number of electrons
    if (lhs.count_a() != rhs.count_a())
        return 0.0;

    int ndiff = lhs.fast_a_xor_b_count(rhs) / 2;
    if (ndiff > 2)
        return 0.0;

    // Slater rule 1 PhiI = PhiJ
    if (ndiff == 0) {
        return energy(lhs);
    }

    // excitation_connection stores the creation and annihilation operators
    // that need to be applied to rhs to obtain lhs:
    // if <LHS|pa^+ qb^+ sa rb|RHS> = +- 1 then excitation_connection = [[s, p], [r, q]]
    // [[alpha annihilation], [alpha creation],
    //  [beta annihilation],  [beta creation]]
    auto excitation_connection = lhs.excitation_connection(rhs);

    std::complex<double> matrix_element = 0.0;
    auto h = one_electron_integrals_.view();
    auto v = two_electron_integrals_.view();

    if (ndiff == 1) {
        size_t i = excitation_connection[0][0];
        size_t a = excitation_connection[1][0];
        double sign = lhs.slater_sign_aa(i, a);
        matrix_element += h(i, a); // <i|a>

        auto occ = lhs.get_alfa_occ(nspinor_);

        if (tei_is_asym_) {
            for (auto j : occ) {
                matrix_element += v(i, j, a, j); // \sum_j<ij||aj>
            }
            matrix_element *= sign;
        } else {
            for (auto j : occ) {
                matrix_element += v(i, j, a, j) - v(i, j, j, a); // \sum_j<ij||aj>
            }
            matrix_element *= sign;
        }
    }

    if (ndiff == 2) {
        size_t i = excitation_connection[0][0];
        size_t j = excitation_connection[0][1];
        size_t a = excitation_connection[1][0];
        size_t b = excitation_connection[1][1];
        double sign = lhs.slater_sign_aaaa(i, j, a, b);
        auto v_el = tei_is_asym_ ? v(i, j, a, b) : v(i, j, a, b) - v(i, j, b, a); // <ij||ab>
        matrix_element += sign * v_el;                                            // <ij||ab>
    }

    return matrix_element;
}

} // namespace forte2

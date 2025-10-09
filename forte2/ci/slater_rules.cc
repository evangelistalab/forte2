#include "slater_rules.h"

namespace forte2 {

SlaterRules::SlaterRules(int norb, double scalar_energy, np_matrix one_electron_integrals,
                         np_tensor4 two_electron_integrals)
    : norb_(norb), norb2_(norb * norb), norb3_(norb * norb * norb), scalar_energy_(scalar_energy) {

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

    for (int p = 0; p < norb_; ++p) {
        for (int q = 0; q < norb_; ++q) {
            h_[p * norb_ + q] = h_view(p, q);                             // <p|h|q>
            J_[p * norb_ + q] = v_view(p, q, p, q);                       // <pq|pq>
            JK_[p * norb_ + q] = v_view(p, q, p, q) - v_view(p, q, q, p); // <pq|pq> - <pq|qp>
            for (int r = 0; r < norb_; ++r) {
                f_J_[p * norb2_ + q * norb_ + r] = v_view(p, r, q, r); // <pr|qr>
                f_JK_[p * norb2_ + q * norb_ + r] =
                    v_view(p, r, q, r) - v_view(p, r, r, q); // <pr|qr> - <pr|rq>
                for (int s = 0; s < norb_; ++s) {
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
    String Ia = det.a_string();
    String Ib = det.b_string();
    String tempI;

    int naocc = Ia.count();
    int nbocc = Ib.count();

    for (int A = 0; A < naocc; ++A) {
        int p = Ia.find_and_clear_first_one();
        energy += h(p, p);

        tempI = Ia;
        for (int AA = A + 1; AA < naocc; ++AA) {
            int q = tempI.find_and_clear_first_one();
            energy += JK(p, q); // <pq||pq> - <pq|qp>
        }

        tempI = Ib;
        for (int B = 0; B < nbocc; ++B) {
            int q = tempI.find_and_clear_first_one();
            energy += J(p, q); // <pq|pq>
        }
    }

    for (int B = 0; B < nbocc; ++B) {
        int p = Ib.find_and_clear_first_one();
        energy += h(p, p);
        tempI = Ib;
        for (int BB = B + 1; BB < nbocc; ++BB) {
            int q = tempI.find_and_clear_first_one();
            energy += JK(p, q); // <pq||pq> - <pq|qp>
        }
    }

    return energy;
}

double SlaterRules::slater_rules(const Determinant& lhs, const Determinant& rhs) const {
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
                    matrix_element += 0.5 * JK(p, q); // <pq||pq> - <pq|qp>
                if (lhs.nb(p) and lhs.nb(q))
                    matrix_element += 0.5 * JK(p, q); // <pq||pq> - <pq|qp>
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

double SlaterRules::singles_coupling(size_t i, size_t a, const std::vector<size_t>& same_spin_occ,
                                     const std::vector<size_t>& opposite_spin_occ) const {
    double coupling = h(i, a);
    for (const auto& j : same_spin_occ) {
        coupling += f_JK(i, a, j); // <ij||aj>
    }
    for (const auto& j : opposite_spin_occ) {
        coupling += f_J(i, a, j); // <ij|aj>
    }
    return coupling;
}

RelSlaterRules::RelSlaterRules(int nspinor, double scalar_energy,
                               np_matrix_complex one_electron_integrals,
                               np_tensor4_complex two_electron_integrals)
    : nspinor_(nspinor), scalar_energy_(scalar_energy),
      one_electron_integrals_(one_electron_integrals),
      two_electron_integrals_(two_electron_integrals) {}

double RelSlaterRules::energy(const Determinant& det) const {
    std::complex<double> energy = scalar_energy_;

    auto h = one_electron_integrals_.view();
    auto v = two_electron_integrals_.view();

    auto occ = det.get_alfa_occ(nspinor_);
    for (auto p : occ) {
        energy += h(p, p); // <p|p>
        for (auto q : occ) {
            energy += 0.5 * (v(p, q, p, q) - v(p, q, q, p)); // <pq||pq>
        }
    }

    return energy.real();
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

        for (auto j : occ) {
            matrix_element += v(i, j, a, j) - v(i, j, j, a); // \sum_j<ij||aj>
        }
        matrix_element *= sign;
    }

    if (ndiff == 2) {
        size_t i = excitation_connection[0][0];
        size_t j = excitation_connection[0][1];
        size_t a = excitation_connection[1][0];
        size_t b = excitation_connection[1][1];
        double sign = lhs.slater_sign_aaaa(i, j, a, b);
        matrix_element += sign * (v(i, j, a, b) - v(i, j, b, a)); // <ij||ab>
    }

    return matrix_element;
}

} // namespace forte2
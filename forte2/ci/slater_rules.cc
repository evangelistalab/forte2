
// #define FAST_SLATER_RULES 0

#include "slater_rules.h"

namespace forte2 {

SlaterRules::SlaterRules(int norb, double scalar_energy, np_matrix one_electron_integrals,
                         np_tensor4 two_electron_integrals)
    : norb_(norb), scalar_energy_(scalar_energy), one_electron_integrals_(one_electron_integrals),
      two_electron_integrals_(two_electron_integrals) {}

double SlaterRules::energy(const Determinant& det) const {
    double energy = scalar_energy_;

    auto h = one_electron_integrals_.view();
    auto v = two_electron_integrals_.view();

    String Ia = det.get_alfa_bits();
    String Ib = det.get_beta_bits();
    String Iac;
    String Ibc;

    int naocc = Ia.count();
    int nbocc = Ib.count();

    for (int A = 0; A < naocc; ++A) {
        int p = Ia.find_and_clear_first_one();
        energy += h(p, p);

        Iac = Ia;
        for (int AA = A + 1; AA < naocc; ++AA) {
            int q = Iac.find_and_clear_first_one();
            energy += v(p, q, p, q) - v(p, q, q, p); // <pq||pq> - <pq|qp>
        }

        Ibc = Ib;
        for (int B = 0; B < nbocc; ++B) {
            int q = Ibc.find_and_clear_first_one();
            energy += v(p, q, p, q); // <pq|pq>
        }
    }

    for (int B = 0; B < nbocc; ++B) {
        int p = Ib.find_and_clear_first_one();
        energy += h(p, p);
        Ibc = Ib;
        for (int BB = B + 1; BB < nbocc; ++BB) {
            int q = Ibc.find_and_clear_first_one();
            energy += v(p, q, p, q) - v(p, q, q, p); // <pq||pq> - <pq|qp>
        }
    }

    return energy;
}

/*
double ActiveSpaceIntegrals::slater_rules(const Determinant& lhs, const Determinant& rhs) const {
    // we first check that the two determinants have equal Ms
    if ((lhs.count_alfa() != rhs.count_alfa()) or (lhs.count_beta() != rhs.count_beta()))
        return 0.0;

#if FAST_SLATER_RULES
#else
    int nadiff = 0;
    int nbdiff = 0;
    // Count how many differences in mos are there
    for (size_t n = 0; n < nmo_; ++n) {
        if (lhs.get_alfa_bit(n) != rhs.get_alfa_bit(n))
            nadiff++;
        if (lhs.get_beta_bit(n) != rhs.get_beta_bit(n))
            nbdiff++;
        if (nadiff + nbdiff > 4)
            return 0.0; // Get out of this as soon as possible
    }
    nadiff /= 2;
    nbdiff /= 2;

    double matrix_element = 0.0;
    // Slater rule 1 PhiI = PhiJ
    if ((nadiff == 0) and (nbdiff == 0)) {
        // matrix_element += frozen_core_energy_ + this->energy(rhs);
        matrix_element = frozen_core_energy_;
        for (size_t p = 0; p < nmo_; ++p) {
            if (lhs.get_alfa_bit(p))
                matrix_element += oei_a_[p * nmo_ + p];
            if (lhs.get_beta_bit(p))
                matrix_element += oei_b_[p * nmo_ + p];
            for (size_t q = 0; q < nmo_; ++q) {
                if (lhs.get_alfa_bit(p) and lhs.get_alfa_bit(q))
                    matrix_element += 0.5 * tei_aa_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
                if (lhs.get_beta_bit(p) and lhs.get_beta_bit(q))
                    matrix_element += 0.5 * tei_bb_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
                if (lhs.get_alfa_bit(p) and lhs.get_beta_bit(q))
                    matrix_element += tei_ab_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
            }
        }
    }

    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 0)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and lhs.get_alfa_bit(p))
                i = p;
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and rhs.get_alfa_bit(p))
                j = p;
        }
        // double sign = SlaterSign(I, i, j);
        double sign = lhs.slater_sign_aa(i, j);
        matrix_element = sign * oei_a_[i * nmo_ + j];
        for (size_t p = 0; p < nmo_; ++p) {
            if (lhs.get_alfa_bit(p) and rhs.get_alfa_bit(p)) {
                matrix_element += sign * tei_aa_[i * nmo3_ + p * nmo2_ + j * nmo_ + p];
            }
            if (lhs.get_beta_bit(p) and rhs.get_beta_bit(p)) {
                matrix_element += sign * tei_ab_[i * nmo3_ + p * nmo2_ + j * nmo_ + p];
            }
        }
    }
    // Slater rule 2 PhiI = j_b^+ i_b PhiJ
    if ((nadiff == 0) and (nbdiff == 1)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and lhs.get_beta_bit(p))
                i = p;
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and rhs.get_beta_bit(p))
                j = p;
        }
        // double sign = SlaterSign(I, nmo_ + i, nmo_ + j);
        double sign = lhs.slater_sign_bb(i, j);
        matrix_element = sign * oei_b_[i * nmo_ + j];
        for (size_t p = 0; p < nmo_; ++p) {
            if (lhs.get_alfa_bit(p) and rhs.get_alfa_bit(p)) {
                matrix_element += sign * tei_ab_[p * nmo3_ + i * nmo2_ + p * nmo_ + j];
            }
            if (lhs.get_beta_bit(p) and rhs.get_beta_bit(p)) {
                matrix_element += sign * tei_bb_[i * nmo3_ + p * nmo2_ + j * nmo_ + p];
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
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and lhs.get_alfa_bit(p)) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and rhs.get_alfa_bit(p)) {
                if (k == -1) {
                    k = p;
                } else {
                    l = p;
                }
            }
        }
        double sign = lhs.slater_sign_aaaa(i, j, k, l);
        matrix_element = sign * tei_aa_[i * nmo3_ + j * nmo2_ + k * nmo_ + l];
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 0) and (nbdiff == 2)) {
        // Diagonal contribution
        int i, j, k, l;
        i = -1;
        j = -1;
        k = -1;
        l = -1;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and lhs.get_beta_bit(p)) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and rhs.get_beta_bit(p)) {
                if (k == -1) {
                    k = p;
                } else {
                    l = p;
                }
            }
        }
        double sign = lhs.slater_sign_bbbb(i, j, k, l);
        matrix_element = sign * tei_bb_[i * nmo3_ + j * nmo2_ + k * nmo_ + l];
    }

    // Slater rule 3 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Diagonal contribution
        int i, j, k, l;
        i = j = k = l = -1;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and lhs.get_alfa_bit(p))
                i = p;
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and lhs.get_beta_bit(p))
                j = p;
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and rhs.get_alfa_bit(p))
                k = p;
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and rhs.get_beta_bit(p))
                l = p;
        }
        double sign = lhs.slater_sign_aa(i, k) * lhs.slater_sign_bb(j, l);
        matrix_element = sign * tei_ab_[i * nmo3_ + j * nmo2_ + k * nmo_ + l];
    }
#endif
    return (matrix_element);
}

double ActiveSpaceIntegrals::slater_rules_single_alpha(const Determinant& det, int i, int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double sign = det.slater_sign_aa(i, a);
    double matrix_element = oei_a_[i * nmo_ + a];
    for (size_t p = 0; p < nmo_; ++p) {
        if (det.get_alfa_bit(p)) {
            matrix_element += tei_aa_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
        if (det.get_beta_bit(p)) {
            matrix_element += tei_ab_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
    }
    return sign * matrix_element;
}

double ActiveSpaceIntegrals::slater_rules_single_alpha_abs(const Determinant& det, int i,
                                                           int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double matrix_element = oei_a_[i * nmo_ + a];
    for (size_t p = 0; p < nmo_; ++p) {
        if (det.get_alfa_bit(p)) {
            matrix_element += tei_aa_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
        if (det.get_beta_bit(p)) {
            matrix_element += tei_ab_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
    }
    return matrix_element;
}

double ActiveSpaceIntegrals::slater_rules_single_beta(const Determinant& det, int i, int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double sign = det.slater_sign_bb(i, a);
    double matrix_element = oei_b_[i * nmo_ + a];
    for (size_t p = 0; p < nmo_; ++p) {
        if (det.get_alfa_bit(p)) {
            matrix_element += tei_ab_[p * nmo3_ + i * nmo2_ + p * nmo_ + a];
        }
        if (det.get_beta_bit(p)) {
            matrix_element += tei_bb_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
    }
    return sign * matrix_element;
}

double ActiveSpaceIntegrals::slater_rules_single_beta_abs(const Determinant& det, int i,
                                                          int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double matrix_element = oei_b_[i * nmo_ + a];
    for (size_t p = 0; p < nmo_; ++p) {
        if (det.get_alfa_bit(p)) {
            matrix_element += tei_ab_[p * nmo3_ + i * nmo2_ + p * nmo_ + a];
        }
        if (det.get_beta_bit(p)) {
            matrix_element += tei_bb_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
    }
    return matrix_element;
}

void ActiveSpaceIntegrals::add(std::shared_ptr<ActiveSpaceIntegrals> as_ints, const double factor) {
    if (as_ints->active_mo_symmetry() != active_mo_symmetry_)
        throw std::runtime_error("Inconsistent active orbitals cannot be added!");

    scalar_energy_ += factor * as_ints->scalar_energy();

    auto add_op = [&factor](double lhs, double rhs) { return lhs + factor * rhs; };

    std::transform(oei_a_.begin(), oei_a_.end(), as_ints->oei_a_vector().begin(), oei_a_.begin(),
                   add_op);
    std::transform(oei_b_.begin(), oei_b_.end(), as_ints->oei_b_vector().begin(), oei_b_.begin(),
                   add_op);

    std::transform(tei_aa_.begin(), tei_aa_.end(), as_ints->tei_aa_vector().begin(),
                   tei_aa_.begin(), add_op);
    std::transform(tei_ab_.begin(), tei_ab_.end(), as_ints->tei_ab_vector().begin(),
                   tei_ab_.begin(), add_op);
    std::transform(tei_bb_.begin(), tei_bb_.end(), as_ints->tei_bb_vector().begin(),
                   tei_bb_.begin(), add_op);
}

void ActiveSpaceIntegrals::print() {
    psi::outfile->Printf("\n\n  ==> Active Space Integrals <==\n");
    psi::outfile->Printf("\n  Nuclear repulsion energy:   %20.12f\n", nuclear_repulsion_energy());
    psi::outfile->Printf("  Frozen core energy:         %20.12f\n", frozen_core_energy());
    psi::outfile->Printf("  Scalar energy:              %20.12f\n", scalar_energy());

    psi::outfile->Printf("\nOne-electron integrals (alpha) <p|h|q> (includes restricted docc)\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            if (std::fabs(oei_a(p, q)) > 1e-12)
                psi::outfile->Printf("  <%2d|h|%2d> = %20.12f\n", p, q, oei_a(p, q));
        }
    }
    psi::outfile->Printf("\nOne-electron integrals (beta) <p|h|q> (includes restricted docc)\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            if (std::fabs(oei_b(p, q)) > 1e-12)
                psi::outfile->Printf("  <%2d|h|%2d> = %20.12f\n", p, q, oei_b(p, q));
        }
    }

    psi::outfile->Printf("\nAntisymmetrized two-electron integrals (alpha-alpha) <pq||rs>\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::fabs(tei_aa(p, q, r, s)) > 1e-12)
                        psi::outfile->Printf("  <%2d %2d|%2d %2d> = %20.12f\n", p, q, r, s,
                                             tei_aa(p, q, r, s));
                }
            }
        }
    }
    psi::outfile->Printf("\nAntisymmetrized two-electron integrals (beta-beta) <pq||rs>\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::fabs(tei_bb(p, q, r, s)) > 1e-12)
                        psi::outfile->Printf("  <%2d %2d|%2d %2d> = %20.12f\n", p, q, r, s,
                                             tei_bb(p, q, r, s));
                }
            }
        }
    }
    psi::outfile->Printf("\nTwo-electron integrals (alpha-beta) <pq|rs>\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::fabs(tei_ab(p, q, r, s)) > 1e-12)
                        psi::outfile->Printf("  <%2d %2d|%2d %2d> = %20.12f\n", p, q, r, s,
                                             tei_ab(p, q, r, s));
                }
            }
        }
    }
}

std::shared_ptr<ActiveSpaceIntegrals>
make_active_space_ints(std::shared_ptr<MOSpaceInfo> mo_space_info,
                       std::shared_ptr<ForteIntegrals> ints, const std::string& active_space,
                       const std::vector<std::string>& core_spaces) {

    bool updated_ints = ints->update_ints_if_needed();
    if (updated_ints) {
        psi::outfile->Printf(
            "\n\n  The integrals are not consistent with the orbitals. Re-transforming them.\n");
    }

    // get the active/core vectors
    auto active_mo = mo_space_info->corr_absolute_mo(active_space);
    auto active_mo_symmetry = mo_space_info->symmetry(active_space);
    std::vector<size_t> core_mo;
    for (const auto& space : core_spaces) {
        auto mos = mo_space_info->corr_absolute_mo(space);
        core_mo.insert(core_mo.end(), mos.begin(), mos.end());
    }

    // allocate the active space integral object
    auto as_ints =
        std::make_shared<ActiveSpaceIntegrals>(ints, active_mo, active_mo_symmetry, core_mo);

    // grab the integrals from the ForteIntegrals object
    if (ints->spin_restriction() == IntegralSpinRestriction::Restricted) {
        auto tei_active_ab = ints->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
        auto tei_active_aa = tei_active_ab.clone();
        tei_active_aa("pqrs") = tei_active_ab("pqrs") - tei_active_ab("pqsr");
        tei_active_aa.set_name("tei_active_aa");
        as_ints->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_aa);
    } else {
        auto tei_active_aa = ints->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
        auto tei_active_ab = ints->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
        auto tei_active_bb = ints->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);
        as_ints->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    }
    as_ints->compute_restricted_one_body_operator();
    return as_ints;
}*/

} // namespace forte2

#include "determinant/rel_slater_rules.h"

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

    int ndiff = lhs.symmetric_difference_count(rhs) / 2;
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

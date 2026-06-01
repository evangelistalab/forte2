#include "determinant/rel_slater_rules.h"
#include "determinant/determinant_helpers.h"

namespace {
std::optional<std::uint32_t> screen_slater_connection(const forte2::Determinant& lhs,
                                                      const forte2::Determinant& rhs) {

    return screen_slater_connection_impl<0, forte2::Determinant::nwords_>(lhs, rhs);
}

std::tuple<std::size_t, std::size_t> find_single_connection(const forte2::Determinant& lhs,
                                                            const forte2::Determinant& rhs) {
    return find_single_connection_impl<0, forte2::Determinant::nwords_>(lhs, rhs);
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
find_double_connection(const forte2::Determinant& lhs, const forte2::Determinant& rhs) {
    return find_double_connection_impl<0, forte2::Determinant::nwords_>(lhs, rhs);
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
    // Early exit for disconnected pairs or if the determinants have different numbers of
    // electrons
    const auto count = screen_slater_connection(lhs, rhs);
    if (!count.has_value()) {
        return 0.0;
    }
    const auto ndiff = count.value();

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

    return energy(lhs);
}
} // namespace forte2

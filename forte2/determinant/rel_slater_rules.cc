#include <stdexcept>
#include <string>

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
                               np_tensor4_complex two_electron_integrals) {
    update_integrals(nspinor, scalar_energy, one_electron_integrals, two_electron_integrals);
}

void RelSlaterRules::update_integrals(int nspinor, std::optional<double> scalar_energy,
                                      std::optional<np_matrix_complex> one_electron_integrals,
                                      std::optional<np_tensor4_complex> two_electron_integrals) {
    if (nspinor < 0) {
        throw std::invalid_argument("RelSlaterRules: nspinor must be non-negative, got " +
                                    std::to_string(nspinor));
    }
    const auto new_nspinor = static_cast<std::size_t>(nspinor);

    if (one_electron_integrals) {
        if (one_electron_integrals->ndim() != 2) {
            throw std::runtime_error("RelSlaterRules: H must be a 2D matrix.");
        }
        if (one_electron_integrals->shape(0) != new_nspinor ||
            one_electron_integrals->shape(1) != new_nspinor) {
            throw std::runtime_error(
                "RelSlaterRules: H shape does not match the number of spinors.");
        }
    }
    if (two_electron_integrals) {
        if (two_electron_integrals->ndim() != 4) {
            throw std::runtime_error("RelSlaterRules: V must be a 4D tensor.");
        }
        if (two_electron_integrals->shape(0) != new_nspinor ||
            two_electron_integrals->shape(1) != new_nspinor ||
            two_electron_integrals->shape(2) != new_nspinor ||
            two_electron_integrals->shape(3) != new_nspinor) {
            throw std::runtime_error(
                "RelSlaterRules: V shape does not match the number of spinors.");
        }
    }
    if (new_nspinor != static_cast<std::size_t>(nspinor_) &&
        (!one_electron_integrals || !two_electron_integrals)) {
        throw std::runtime_error(
            "RelSlaterRules: changing nspinor requires one_electron_integrals and "
            "two_electron_integrals to be given together, since this class stores them "
            "directly and a stale one would be read out of bounds.");
    }

    nspinor_ = nspinor;
    if (scalar_energy) {
        scalar_energy_ = *scalar_energy;
    }
    if (one_electron_integrals) {
        h_.resize(new_nspinor * new_nspinor);
        auto h_view = one_electron_integrals->view();
        for (std::size_t p = 0; p < new_nspinor; ++p) {
            for (std::size_t q = 0; q < new_nspinor; ++q) {
                h_[p * new_nspinor + q] = h_view(p, q);
            }
        }
    }
    if (two_electron_integrals) {
        v_.resize(new_nspinor * new_nspinor * new_nspinor * new_nspinor);
        auto v_view = two_electron_integrals->view();
        for (std::size_t p = 0, pqrs = 0; p < new_nspinor; ++p) {
            for (std::size_t q = 0; q < new_nspinor; ++q) {
                for (std::size_t r = 0; r < new_nspinor; ++r) {
                    for (std::size_t s = 0; s < new_nspinor; ++s, ++pqrs) {
                        v_[pqrs] = v_view(p, q, r, s);
                    }
                }
            }
        }
    }
}

double RelSlaterRules::energy(const Determinant& det) const {
    std::complex<double> energy = scalar_energy_;
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
        const auto [i, j, a, b] = find_double_connection(lhs, rhs);
        auto v_el = v(i, j, a, b) - v(i, j, b, a); // <ij||ab>
        const double sign = lhs.slater_sign_aaaa(i, j, a, b);
        return sign * v_el;
    }

    if (ndiff == 2) {
        const auto [i, a] = find_single_connection(lhs, rhs);
        std::complex<double> matrix_element = h(i, a); // <i|a>
        lhs.for_each_occ([&](size_t j) {
            matrix_element += v(i, j, a, j) - v(i, j, j, a); // \sum_j<ij||aj>
        });
        const double sign = lhs.slater_sign_aa(i, a);
        return sign * matrix_element;
    }

    return energy(lhs);
}
} // namespace forte2

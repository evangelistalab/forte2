#include "ints/fock_builder.h"

#include "ints/basis.h"
#include "ints/two_electron.h"

namespace forte2 {

FockBuilder::FockBuilder(const Basis& basis, const Basis& auxiliary_basis)
    : basis_(basis), auxiliary_basis_(auxiliary_basis) {
    integrals_ = coulomb_4c(basis_, basis_, basis_, basis_);
}

std::pair<std::vector<np_matrix>, std::vector<np_matrix>>
FockBuilder::build(std::vector<np_matrix>& density_matrices) {
    // Get the number of basis functions in each basis
    const std::size_t nb = basis_.size();
    const std::size_t nd = density_matrices.size();

    // Allocate the J and K matrices
    std::vector<np_matrix> J(nd);
    std::vector<np_matrix> K(nd);
    for (std::size_t k = 0; k < density_matrices.size(); ++k) {
        J[k] = make_ndarray<nb::numpy, double, 2>({nb, nb});
        K[k] = make_ndarray<nb::numpy, double, 2>({nb, nb});
    }

    opt_fock_build(density_matrices, integrals_, J, K);

    return {J, K};
}

void FockBuilder::naive_fock_build(const std::vector<np_matrix>& density_matrices,
                                   const nb::ndarray<nb::numpy, double, nb::ndim<4>>& integrals,
                                   std::vector<np_matrix>& J, std::vector<np_matrix>& K) {
    const std::size_t nd = density_matrices.size();

    // Compute the J and K matrices
    auto v = integrals.view();
    for (size_t a = 0; a < nd; ++a) {
        auto Ja = J[a].view();
        auto Ka = K[a].view();
        auto D = density_matrices[a].view();
        for (size_t i = 0; i < v.shape(0); ++i)
            for (size_t j = 0; j < v.shape(1); ++j) {
                double J_ij = 0.0;
                double K_ij = 0.0;
                for (size_t k = 0; k < v.shape(2); ++k) {
                    for (size_t l = 0; l < v.shape(3); ++l) {
                        J_ij += v(i, j, k, l) * D(k, l);
                        K_ij += v(i, k, j, l) * D(k, l);
                    }
                }
                Ja(i, j) = J_ij;
                Ka(i, j) = K_ij;
            }
    }
}

void FockBuilder::opt_fock_build(const std::vector<np_matrix>& density_matrices,
                                 const nb::ndarray<nb::numpy, double, nb::ndim<4>>& integrals,
                                 std::vector<np_matrix>& J, std::vector<np_matrix>& K) {
    const std::size_t nd = density_matrices.size();

    // Compute the J and K matrices
    auto v = integrals.view();
    for (size_t a = 0; a < nd; ++a) {
        auto Ja = J[a].view();
        auto Ka = K[a].view();
        auto D = density_matrices[a].view();
        for (size_t i = 0; i < v.shape(0); ++i)
            for (size_t j = i; j < v.shape(1); ++j) {
                double J_ij = 0.0;
                double K_ij = 0.0;
                for (size_t k = 0; k < v.shape(2); ++k) {
                    for (size_t l = k + 1; l < v.shape(3); ++l) {
                        J_ij += 2.0 * v(i, j, k, l) * D(k, l);
                        K_ij += (v(i, k, j, l) + v(i, l, j, k)) * D(k, l);
                    }
                    J_ij += v(i, j, k, k) * D(k, k);
                    K_ij += v(i, k, j, k) * D(k, k);
                }
                Ja(i, j) = Ja(j, i) = J_ij;
                Ka(i, j) = Ka(j, i) = K_ij;
            }
    }
}

} // namespace forte2
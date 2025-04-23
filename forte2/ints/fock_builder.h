#pragma once

// #include <array>

// #include <nanobind/nanobind.h>
// #include <nanobind/ndarray.h>

// namespace nb = nanobind;

#include "helpers/ndarray.h"
#include "ints/basis.h"

namespace forte2 {
class Basis;

class FockBuilder {
  public:
    FockBuilder(const Basis& basis, const Basis& auxiliary_basis = Basis());

    /// @brief Build the Coulomb and Exchange matrices.
    std::pair<std::vector<np_matrix>, std::vector<np_matrix>>
    build(std::vector<np_matrix>& density_matrices);

  private:
    Basis basis_;
    Basis auxiliary_basis_;
    nb::ndarray<nb::numpy, double, nb::ndim<4>> integrals_;
    nb::ndarray<nb::numpy, double, nb::ndim<2>> PQ;
    nb::ndarray<nb::numpy, double, nb::ndim<3>> mnP;

    void naive_fock_build(const std::vector<np_matrix>& density_matrices,
                          const nb::ndarray<nb::numpy, double, nb::ndim<4>>& integrals,
                          std::vector<np_matrix>& J, std::vector<np_matrix>& K);

    void opt_fock_build(const std::vector<np_matrix>& density_matrices,
                        const nb::ndarray<nb::numpy, double, nb::ndim<4>>& integrals,
                        std::vector<np_matrix>& J, std::vector<np_matrix>& K);

    // void df_fock_build(const std::vector<np_matrix>& density_matrices,
    //                    nb::ndarray<nb::numpy, double, nb::ndim<2>>& PQ,
    //                    nb::ndarray<nb::numpy, double, nb::ndim<3>>& mnP, std::vector<np_matrix>&
    //                    J, std::vector<np_matrix>& K);
};

} // namespace forte2
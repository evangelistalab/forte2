#pragma once

// #include <array>

// #include <nanobind/nanobind.h>
// #include <nanobind/ndarray.h>

// namespace nb = nanobind;

#include "helpers/ndarray.h"
#include "integrals/basis.h"

namespace forte2 {
class Basis;

class FockBuilder {
  public:
    FockBuilder(const Basis& basis, const Basis& auxiliary_basis = Basis());

    /// @brief Build the Coulomb and Exchange matrices.
    std::pair<std::vector<ndarray<double, 2>>, std::vector<ndarray<double, 2>>>
    build(std::vector<ndarray<double, 2>>& density_matrices);

  private:
    Basis basis_;
    Basis auxiliary_basis_;
    ndarray<double, 4> integrals_;
    ndarray<double, 2> PQ;
    ndarray<double, 3> mnP;

    void naive_fock_build(const std::vector<ndarray<double, 2>>& density_matrices,
                          const ndarray<double, 4>& integrals, std::vector<ndarray<double, 2>>& J,
                          std::vector<ndarray<double, 2>>& K);

    void opt_fock_build(const std::vector<ndarray<double, 2>>& density_matrices,
                        const ndarray<double, 4>& integrals, std::vector<ndarray<double, 2>>& J,
                        std::vector<ndarray<double, 2>>& K);

    // void df_fock_build(const std::vector<ndarray<double, 2>>& density_matrices,
    //                    ndarray<double, 2>& PQ,
    //                    ndarray<double, 3>& mnP, std::vector<ndarray<double, 2>>&
    //                    J, std::vector<ndarray<double, 2>>& K);
};

} // namespace forte2
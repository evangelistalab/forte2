#pragma once

#include <vector>

#include <libint2/shell.h>

#include "helpers/ndarray.h"
namespace forte2 {
class Basis {
  public:
    // Default constructor
    Basis();

    // Construct from shells
    Basis(const std::vector<libint2::Shell>& shells);

    // Add shells
    void add(const libint2::Shell& shell);

    /// @return the number of basis functions in the basis set
    std::size_t size() const;

    /// @return the number of shells in the basis set
    std::size_t nshells() const;

    /// @return the max angular momentum of the basis set
    int max_l() const;

    /// @return the max number of primitives in a shell
    std::size_t max_nprim() const;

    /// @return the i-th shell in the basis set
    const libint2::Shell& operator[](size_t i) const;

    /// @brief Evaluate the basis functions at the given points.
    /// @param points a vector of points at which to evaluate the basis functions
    /// @return a 2D array of shape (npoints, nbasis) containing the values of the basis
    /// functions
    np_matrix value_at_points(const std::vector<std::array<double, 3>>& points) const;

    /// @brief Evaluate the product of basis functions times a coefficient matrix at the given
    /// points.
    /// @param points a vector of points at which to evaluate the basis functions
    /// @param C a 2D array of shape (nbasis, norb) containing the coefficients
    /// @return a 2D array of shape (npoints, norb) containing the values of the basis
    /// functions times the coefficients
    /// @note The shape of C must be (nbasis, norb), where nbasis is the number of basis
    /// functions in the basis set and norb is the number of orbitals.
    np_matrix value_at_points_C(const std::vector<std::array<double, 3>>& points,
                                np_matrix C) const;

    /// @return a vector of pairs of the first index and size of each shell
    ///         in the basis set. The first index is the index of the first basis function
    ///         in the shell, and the size is the number of basis functions in the shell.
    std::vector<std::pair<std::size_t, std::size_t>> shell_first_and_size() const;

    /// @return a vector of pairs of the first and last index of the basis functions on a given
    /// center in the basis set.
    std::vector<std::pair<std::size_t, std::size_t>> center_first_and_last() const;

  private:
    std::vector<libint2::Shell> shells_;
    size_t size_ = 0;      // total number of basis functions
    size_t max_nprim_ = 0; // max number of primitives in shells
    int max_l_ = 0;        // max angular momentum of shells
};

void evaluate_shell(const libint2::Shell& shell, const std::array<double, 3>& point,
                    double* buffer);

} // namespace forte2
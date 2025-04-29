#pragma once

#include <vector>

#include <libint2/shell.h>

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
    /// @param out a vector to store the evaluated basis functions
    /// @note The size of the output vector must be at least size() * points.size().
    ///       The output is stored in a column-major order, i.e. the first size() values
    ///       correspond to the first point, the next size() values to the second point, etc.
    ///       The output is not resized, so it must be large enough to hold the results.
    void value_at_points(const std::vector<std::array<double, 3>>& points,
                         std::vector<double>& out) const;

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

void evaluate_shell(const libint2::Shell& shell, const std::vector<std::array<double, 3>>& points,
                    std::vector<double>& values);

} // namespace forte2
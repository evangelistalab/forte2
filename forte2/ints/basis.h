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
    /// @return a vector of pairs of the first index and size of each shell
    ///         in the basis set. The first index is the index of the first basis function
    ///         in the shell, and the size is the number of basis functions in the shell.
    std::vector<std::pair<std::size_t, std::size_t>> shell_first_and_size() const;

  private:
    std::vector<libint2::Shell> shells_;
    size_t size_ = 0;      // total number of basis functions
    size_t max_nprim_ = 0; // max number of primitives in shells
    int max_l_ = 0;        // max angular momentum of shells
};
} // namespace forte2
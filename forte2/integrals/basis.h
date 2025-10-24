#pragma once

#include <vector>
#include <string>

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

    /// Set the name of the basis set
    void set_name(const std::string& name) { name_ = name; }

    /// @return the number of basis functions in the basis set
    std::size_t size() const;

    /// @return the number of shells in the basis set
    std::size_t nshells() const;

    /// @return the max angular momentum of the basis set
    int max_l() const;

    /// @return the name of the basis set
    std::string name() const;

    /// @return the max number of primitives in a shell
    std::size_t max_nprim() const;

    /// @return the vector of shells in the basis set
    const std::vector<libint2::Shell>& shells() const;

    /// @return the vector of shells in the basis set
    std::vector<libint2::Shell>& shells();

    /// @return the i-th shell in the basis set
    const libint2::Shell& operator[](size_t i) const;

    /// @return a vector of pairs of the first index and size of each shell
    ///         in the basis set. The first index is the index of the first basis function
    ///         in the shell, and the size is the number of basis functions in the shell.
    std::vector<std::pair<std::size_t, std::size_t>> shell_first_and_size() const;

    /// @return a vector of pairs of the first and last index of the basis functions on a given
    /// center in the basis set.
    std::vector<std::pair<std::size_t, std::size_t>> center_first_and_last() const;

    /// Set Libcint data structures
    void set_cint_atm(np_matrix_int atm) { cint_atm = atm; }
    void set_cint_bas(np_matrix_int bas) { cint_bas = bas; }
    void set_cint_env(np_vector env) { cint_env = env; }

    /// Return Libcint data structures
    np_matrix_int get_cint_atm() const { return cint_atm; }
    np_matrix_int get_cint_bas() const { return cint_bas; }
    np_vector get_cint_env() const { return cint_env; }

  private:
    std::vector<libint2::Shell> shells_; // vector of shells in the basis set
    size_t size_ = 0;                    // total number of basis functions
    size_t max_nprim_ = 0;               // max number of primitives in shells
    int max_l_ = 0;                      // max angular momentum of shells
    std::string name_ = "unnamed basis"; // name of the basis set

    // Data for Libcint
    np_matrix_int cint_atm; // Libcint atom array
    np_matrix_int cint_bas; // Libcint basis (shell) array
    np_vector cint_env;     // Libcint environment vector
};

/// @brief Evaluate a shell at a given point in space.
/// @param shell The shell to evaluate.
/// @param point The point in space (x, y, z) at which to evaluate the shell.
/// @param buffer A pointer to a buffer where the results will be stored.
void evaluate_shell(const libint2::Shell& shell, const std::array<double, 3>& point,
                    double* buffer);

/// @brief The atomic orbital label for a given angular momentum (l) and index (idx) matching the
/// forte2/libint2 convention.
std::string shell_label(int l, int idx);

} // namespace forte2
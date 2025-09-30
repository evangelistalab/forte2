#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <span>

#include "helpers/ndarray.h"
#include "helpers/unordered_dense.h"
#include "determinant.h"
#include "ci/slater_rules.h"

namespace forte2 {

class SortedStringList {
  public:
    SortedStringList(size_t norb, std::vector<Determinant>&& sorted_dets);
    SortedStringList() = default;

    /// @return The number of orbitals
    size_t norb() const { return norb_; }
    /// @return The number of determinants
    size_t ndets() const { return ndets_; }
    /// @return The sorted determinants
    const std::vector<Determinant>& sorted_dets() const { return sorted_dets_; }
    /// @return The i-th sorted first string
    const String& sorted_first_string(size_t i) const { return sorted_first_string_[i]; }
    /// @return The i-th sorted second string
    const String& sorted_second_string(size_t i) const { return sorted_second_string_[i]; }
    /// @return The permutation that sorts the determinants (perm[i] gives the index in the original
    /// det ordering)
    const std::vector<size_t>& det_permutation() const { return det_permutation_; }
    /// @return The range of indices of the determinants that go with the i-th unique first string
    /// as (start, end) indices in sorted_dets_
    const std::pair<size_t, size_t> range(size_t i) const { return first_string_range_[i]; }
    /// @return The index sorted second string that corresponds to the i-th determinant in
    /// sorted_dets_
    const size_t sorted_dets_second_string(size_t i) const { return sorted_dets_second_string_[i]; }
    /// @return The number of unique first strings
    size_t first_string_size() const { return sorted_first_string_.size(); }

    /// @return For a given first string, this maps the second string index to the determinant index
    /// in the original ordering
    const std::vector<ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>>>&
    second_string_to_det_index() const {
        return second_string_to_det_index_;
    }

    const std::vector<String>& one_hole_strings() const { return one_hole_strings_; }

    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_string_list() const {
        return one_hole_string_list_;
    }

    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_string_list_inv() const {
        return one_hole_string_list_inv_;
    }

    const std::vector<String>& one_hole_second_strings() const { return one_hole_second_strings_; }
    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_second_string_list() const {
        return one_hole_second_string_list_;
    }
    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_second_string_list_inv() const {
        return one_hole_second_string_list_inv_;
    }

    const std::vector<String>& two_hole_strings() const { return two_hole_strings_; }

    const std::vector<std::vector<std::tuple<size_t, size_t, size_t, double>>>&
    two_hole_string_list() const {
        return two_hole_string_list_;
    }
    const std::vector<std::vector<std::tuple<size_t, size_t, size_t, double>>>&
    two_hole_string_list_inv() const {
        return two_hole_string_list_inv_;
    }

  protected:
    // == Class Protected Variables ==
    /// @brief Number of orbitals
    size_t norb_ = 0;
    /// @brief Number of determinants
    size_t ndets_ = 0;
    /// @brief The sorted determinants
    std::vector<Determinant> sorted_dets_;
    /// @brief The permutation that sorts the determinants
    /// det_permutation_[i] gives the index in the original det ordering
    std::vector<size_t> det_permutation_;
    /// @brief The range of each unique first string. first_string_range_[i] = (start, end)
    /// where start and end are indices in sorted_dets_ and sorted_dets_second_string_.
    std::vector<std::pair<size_t, size_t>> first_string_range_;
    /// @brief The unique first strings
    std::vector<String> sorted_first_string_;
    /// @brief The unique second strings
    std::vector<String> sorted_second_string_;
    /// @brief The unique addresses of the second strings corresponding to each determinant in
    /// sorted_dets_
    std::vector<size_t> sorted_dets_second_string_;
    /// @brief Map from first string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> first_string_index_;
    /// @brief Map from second string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> second_string_index_;
    /// @brief For each unique first string, map from second string index to determinant index
    std::vector<ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>>>
        second_string_to_det_index_;

    /// @brief Precomputed list of one-hole strings
    std::vector<String> one_hole_strings_;
    /// @brief Map from one-hole string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> one_hole_strings_index_;
    /// @brief Precomputed list of one-particle strings with sign for each orbital
    /// Stores I -> tuples of (orbital, K, sign) where K is the index of the one-hole string
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> one_hole_string_list_;
    /// @brief Precomputed list of one-particle strings with sign for each orbital (inverse)
    /// Stores K -> tuples of (orbital, I, sign) where I is the index of the string
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> one_hole_string_list_inv_;

    /// @brief Precomputed list of one-hole strings
    std::vector<String> one_hole_second_strings_;
    /// @brief Map from one-hole string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> one_hole_second_strings_index_;
    /// @brief Precomputed list of one-particle strings with sign for each orbital
    /// Stores I -> tuples of (orbital, K, sign) where K is the index of the one-hole string
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> one_hole_second_string_list_;
    /// @brief Precomputed list of one-particle strings with sign for each orbital (inverse)
    /// Stores K -> tuples of (orbital, I, sign) where I is the index of the string
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> one_hole_second_string_list_inv_;

    /// @brief Two-hole strings
    std::vector<String> two_hole_strings_;
    /// @brief Map from two-hole string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> two_hole_strings_index_;
    /// @brief Precomputed list of two-hole strings with sign for each pair of orbitals
    /// Stores I -> tuples of (orbital1, orbital2, K, sign) where K is the index of the two-hole
    /// string
    std::vector<std::vector<std::tuple<size_t, size_t, size_t, double>>> two_hole_string_list_;
    /// @brief Precomputed list of two-hole strings with sign for each pair of orbitals (inverse)
    /// Stores K -> tuples of (orbital1, orbital2, I, sign) where I is the index of the string
    std::vector<std::vector<std::tuple<size_t, size_t, size_t, double>>> two_hole_string_list_inv_;
};

class SelectedCIHelper {
  public:
    // == Class Constructor ==
    SelectedCIHelper(size_t norb, const std::vector<Determinant>& dets, np_matrix& c, double E,
                     np_matrix& H, np_tensor4& V, int log_level = 3);

    // == Class Public Methods ==

    /// @brief Return the determinants in the variational space
    const std::vector<Determinant>& get_variational_dets() const { return dets_; }

    /// @brief Set the Hamiltonian integrals
    void set_Hamiltonian(double E, np_matrix H, np_tensor4 V);

    np_matrix fullHamiltonian() const;

    void set_c(np_matrix& c);

    /// @brief Perform CIPSI selection with the given threshold
    void select_cipsi(double threshold);

    /// @brief Perform HBCI selection with the given threshold
    void select_hbci(double threshold);

    void Hamiltonian(np_vector basis, np_vector sigma) const;

    np_vector Hdiag() const;

  private:
    // == Class Private Methods ==
    void prepare_sigma_build();
    void H0(std::span<double> basis, std::span<double> sigma) const;
    void H1a(std::span<double> basis, std::span<double> sigma) const;
    void H1b(std::span<double> basis, std::span<double> sigma) const;
    void H2a(std::span<double> basis, std::span<double> sigma) const;
    void H2b(std::span<double> basis, std::span<double> sigma) const;
    void H2ab(std::span<double> basis, std::span<double> sigma) const;
    void find_matching_dets(std::span<double> basis, std::span<double> sigma,
                            const SortedStringList& list, size_t i, size_t j,
                            double int_sign) const;

    // == Class Private Variables ==

    /// @brief logging level for the class
    int log_level_ = 3;

    /// @brief Number of orbitals
    const size_t norb_;
    const size_t norb2_;
    const size_t norb3_;

    /// @brief The scalar energy
    double E_;
    /// @brief One-electron integrals in the form of a matrix H[p][q] = <p|H|q> = h_pq
    np_matrix H_;
    /// @brief Two-electron integrals in the form of a tensor V[p][q][r][s] = <pq|rs> = (pr|qs)
    np_tensor4 V_;

    SlaterRules slater_rules_;

    /// @brief Orbital energies: e[p] = <p|H|p>
    std::vector<double> epsilon_;
    /// @brief One-electron integrals: H[p][q] = <p|H|q> = h_pq
    std::vector<double> h_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq|rs> = (pr|qs)
    std::vector<double> v_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq||rs> = (pr|qs) - (ps|qr)
    std::vector<double> v_a_;

    std::vector<std::vector<std::tuple<double, size_t, size_t>>> v_sorted_;
    std::vector<std::vector<std::tuple<double, size_t, size_t>>> va_sorted_;

    inline const double& V(std::size_t i, std::size_t j, std::size_t a,
                           std::size_t b) const noexcept {
        return v_[i * norb3_ + j * norb2_ + a * norb_ + b];
    }

    inline const double& Va(std::size_t i, std::size_t j, std::size_t a,
                            std::size_t b) const noexcept {
        return v_a_[i * norb3_ + j * norb2_ + a * norb_ + b];
    }

    /// @brief The determinants in the reference space
    size_t nroots_;
    /// @brief The initial guess for the CI coefficients
    np_matrix c_guess_;
    /// @brief The initial guess for the determinants
    std::vector<Determinant> guess_dets_;

    /// @brief The determinants in the variational space
    std::vector<Determinant> dets_;
    /// @brief The energies of the determinants in the variational space
    std::vector<double> det_energies_;
    /// @brief The CI coefficients of the determinants in the variational space
    /// Stored as a flat vector of size dets_.size() * nroots_, where the coefficients for each
    /// root are stored contiguously. E.g., the coefficient for determinant i and root r is at
    /// index i * nroots_ + r.
    std::vector<double> c_;

    SortedStringList ab_list_;
    SortedStringList ba_list_;
};

} // namespace forte2

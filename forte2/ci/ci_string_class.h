#pragma once

#include <vector>
#include <unordered_map>
#include <map>

namespace forte2 {
class StringClass {
  public:
    StringClass(size_t symmetry, const std::vector<std::vector<int>>& orbital_symmetry,
                const std::vector<std::array<int, 6>>& alfa_occupation,
                const std::vector<std::array<int, 6>>& beta_occupation,
                const std::vector<std::pair<size_t, size_t>>& occupations);

    /// @brief Return the symmetry of the MOs
    const std::vector<int>& mo_sym() const;
    /// @brief Return the symmetry of a string
    size_t symmetry(const String& s) const;
    /// @brief Return the class of an alpha string
    size_t alfa_string_class(const String& s) const;
    /// @brief Return the class of a beta string
    size_t beta_string_class(const String& s) const;
    /// @brief Return the number of alpha strings classes
    size_t num_alfa_classes() const;
    /// @brief Return the number of beta strings classes
    size_t num_beta_classes() const;
    /// @brief Return the alpha string classes
    const std::vector<std::pair<size_t, size_t>>& alfa_string_classes() const;
    /// @brief Return the beta string classes
    const std::vector<std::pair<size_t, size_t>>& beta_string_classes() const;
    /// @brief Return a list of tuples of the form (class_idx, class_Ia, class_Ib)
    const std::vector<std::tuple<size_t, size_t, size_t>>& determinant_classes() const;
    /// @brief Return block index
    int block_index(int class_Ia, int class_Ib) const;

  private:
    /// The number of irreps
    size_t nirrep_;
    /// The symmetry of each MO
    std::vector<int> mo_sym_;
    /// A map from the occupation of each GAS space to the class for the alpha strings
    std::map<std::array<int, 6>, size_t> alfa_occupation_group_;
    /// A map from the occupation of each GAS space to the class for the beta strings
    std::map<std::array<int, 6>, size_t> beta_occupation_group_;
    /// A list of all possible occupations of the alpha and beta GAS spaces
    std::vector<std::pair<size_t, size_t>> occupations_;
    /// A list of the classes of the alpha strings stored as tuple
    /// (occupation group idx, symmetry)
    std::vector<std::pair<size_t, size_t>> alfa_string_classes_;
    /// A list of the classes of the beta strings stored as tuple
    /// (occupation group idx, symmetry)
    std::vector<std::pair<size_t, size_t>> beta_string_classes_;
    /// A list of the product of alpha and beta string classes stored as tuple
    /// (alfa class index, beta class index)
    std::map<std::pair<size_t, size_t>, size_t> alfa_string_classes_map_;
    /// A list of the classes of the beta strings stored as tuple
    /// (occupation group idx, symmetry)
    std::map<std::pair<size_t, size_t>, size_t> beta_string_classes_map_;
    /// A list of the product of alpha and beta string classes stored as tuple
    /// (product class index, alfa class index, beta class index)
    std::vector<std::tuple<size_t, size_t, size_t>> determinant_classes_;
    /// A map from the class of alpha and beta string classes to the block index
    std::map<std::pair<size_t, size_t>, int> block_index_;
    /// A mask used to count the number of 1s in a string that belongs to a given GAS space
    std::array<String, 6> gas_masks_;
};

} // namespace forte2

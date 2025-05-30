#pragma once

#include <vector>
#include <unordered_map>
#include <map>

#include "determinant.h"

namespace forte2 {
/**
 * @brief The StringAddress class
 * This class computes the address of a string
 */
class StringAddress {
  public:
    // ==> Class Constructor and Destructor <==
    /// @brief Default constructor
    /// @param strings a vector of vectors of strings.
    /// Each vector collects the strings of a given symmetry
    StringAddress(const std::vector<int>& gas_size, int ne,
                  const std::vector<std::vector<String>>& strings);

    /// @brief Default destructor
    ~StringAddress() = default;

    // ==> Class Interface <==
    /// @brief Add a string with a given irrep
    void push_back(const String& s, int irrep);
    /// @brief Return the address of a string within an irrep
    size_t add(const String& s) const;
    /// @brief
    std::unordered_map<String, std::pair<uint32_t, uint32_t>, String::Hash>::const_iterator
    find(const String& s) const;
    std::unordered_map<String, std::pair<uint32_t, uint32_t>, String::Hash>::const_iterator
    end() const;
    /// @brief Return the irrep of a string
    int sym(const String& s) const;
    /// @brief Return the address and irrep of a string
    const std::pair<uint32_t, uint32_t>& address_and_class(const String& s) const;
    /// @brief Return the number of string classes
    int nclasses() const;
    /// @brief Return the number of strings in a given class
    size_t strpcls(int h) const;
    /// @brief Return the number of bits in the string
    int nbits() const;
    /// @brief Return the number of 1s in the string
    int nones() const;

  private:
    // ==> Class Data <==
    /// number of string classes
    int nclasses_;
    /// number of strings
    size_t nstr_;
    /// number of strings in each class
    std::vector<size_t> strpcls_;
    /// Map from string to address and class
    std::unordered_map<String, std::pair<uint32_t, uint32_t>, String::Hash> address_;
    int nones_; // number of 1s
    /// the number of orbitals in each gas space
    const std::vector<int> gas_size_;
};

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

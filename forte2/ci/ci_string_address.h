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

} // namespace forte2

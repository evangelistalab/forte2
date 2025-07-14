#pragma once

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "ci/ci_string_defs.h"
#include "ci/ci_string_address.h"
#include "ci/ci_string_class.h"

namespace forte2 {

class CIStrings {
  public:
    // ==> Constructor and Destructor <==
    /// @brief The CIStrings constructor
    CIStrings(size_t na, size_t nb, int symmetry, std::vector<std::vector<int>> orbital_symmetry,
              const std::vector<int> gas_min, const std::vector<int> gas_max);

    //   // ==> Class Public Functions <==

    /// @return the number of alpha electrons
    size_t na() const { return na_; }

    /// @return the number of beta electrons
    size_t nb() const { return nb_; }

    /// @return the number of alpha strings
    size_t nas() const { return nas_; }

    /// @return the number of beta strings
    size_t nbs() const { return nbs_; }

    /// @return the number of determinants
    size_t ndet() const { return ndet_; }

    /// @return the symmetry of the state
    int symmetry() const { return symmetry_; }

    /// @return the number of irreps
    int nirrep() const { return nirrep_; }

    /// @return the number of correlated MOs
    size_t norb() const { return norb_; }

    /// @brief Set the printing level for the class
    void set_log_level(int level) { log_level_ = level; }

    /// @return the alpha string address object
    const auto& alfa_address() const { return alfa_address_; }
    /// @return the beta string address object
    const auto& beta_address() const { return beta_address_; }

    /// @return the alpha string address object
    const auto& alfa_address_1h1p() const { return alfa_address_1h1p_; }
    /// @return the beta string address object
    const auto& beta_address_1h1p() const { return beta_address_1h1p_; }

    /// @return the alpha string address object for N - 1 electrons
    auto alfa_address_1h() const { return alfa_address_1h_; }
    /// @return the beta string address object for N - 1 electrons
    auto beta_address_1h() const { return beta_address_1h_; }
    /// @return the alpha string address object for N - 2 electrons
    auto alfa_address_2h() const { return alfa_address_2h_; }
    /// @return the beta string address object for N - 2 electrons
    auto beta_address_2h() const { return beta_address_2h_; }
    /// @return the alpha string address object for N - 3 electrons
    auto alfa_address_3h() const { return alfa_address_3h_; }
    /// @return the beta string address object for N - 3 electrons
    auto beta_address_3h() const { return beta_address_3h_; }

    /// @return the address of a determinant in the CI vector
    size_t determinant_address(const Determinant& d) const;
    /// @return the determinant corresponding to an address in the CI vector of a given symmetry
    Determinant determinant(size_t address) const;

    /// @return the alpha string list
    const auto& alfa_strings() const { return alfa_strings_; }
    /// @return the beta string list
    const auto& beta_strings() const { return beta_strings_; }
    /// @return the alpha string in irrep h and index I
    String alfa_str(size_t h, size_t I) const { return alfa_strings_[h][I]; }
    /// @return the beta string in irrep h and index I
    String beta_str(size_t h, size_t I) const { return beta_strings_[h][I]; }

    /// @return the string class object
    const auto& string_class() const { return string_class_; }
    /// @return the alpha string classes
    const auto& alfa_string_classes() const { return string_class_->alfa_string_classes(); }
    /// @return the beta string classes
    const auto& beta_string_classes() const { return string_class_->beta_string_classes(); }
    /// @return the alpha/beta string classes
    const auto& determinant_classes() const { return string_class_->determinant_classes(); }

    /// @return the alpha GAS occupations
    const auto& gas_alfa_occupations() const { return gas_alfa_occupations_; }
    /// @return the beta GAS occupations
    const auto& gas_beta_occupations() const { return gas_beta_occupations_; }
    /// @return the alpha GAS 1h1p occupations
    const auto& gas_alfa_1h1p_occupations() const { return gas_alfa_1h1p_occupations_; }
    /// @return the beta GAS 1h1p occupations
    const auto& gas_beta_1h1p_occupations() const { return gas_beta_1h1p_occupations_; }

    /// @return the number of determinants in a given block
    size_t detpblk(size_t block) const { return detpblk_[block]; }

    /// @return the offset of a given block in the CI vector
    size_t block_offset(size_t block) const { return detpblk_offset_[block]; }

    /// @return the list of determinants
    std::vector<Determinant> make_determinants() const;

    const VOListElement& get_alfa_vo_list(int class_I, int class_J) const;
    const VOListElement& get_beta_vo_list(int class_I, int class_J) const;

    const VOListElement2& get_alfa_vo_list2(int class_I, int class_J) const;
    const VOListElement2& get_beta_vo_list2(int class_I, int class_J) const;

    const std::vector<H1StringSubstitution>& get_alfa_1h_list(int h_I, size_t add_I, int h_J) const;
    const std::vector<H1StringSubstitution>& get_beta_1h_list(int h_I, size_t add_I, int h_J) const;

    const std::vector<std::vector<H1StringSubstitution>>& get_alfa_1h_list2(int h_I, int h_J) const;
    const std::vector<std::vector<H1StringSubstitution>>& get_beta_1h_list2(int h_I, int h_J) const;

    const std::vector<H2StringSubstitution>& get_alfa_2h_list(int h_I, size_t add_I, int h_J) const;
    const std::vector<H2StringSubstitution>& get_beta_2h_list(int h_I, size_t add_I, int h_J) const;

    const std::vector<H3StringSubstitution>& get_alfa_3h_list(int h_I, size_t add_I, int h_J) const;
    const std::vector<H3StringSubstitution>& get_beta_3h_list(int h_I, size_t add_I, int h_J) const;

    template <typename Func> void for_each_element(Func&& func) const {
        std::size_t idx = 0;

        for (auto const& [block, class_Ia, class_Ib] : determinant_classes()) {
            auto const nIa = alfa_address()->strpcls(class_Ia);
            auto const nIb = beta_address()->strpcls(class_Ib);

            if (nIa == 0 or nIb == 0)
                continue;

            for (std::size_t Ia = 0; Ia < nIa; ++Ia) {
                for (std::size_t Ib = 0; Ib < nIb; ++Ib) {
                    func(block, class_Ia, class_Ib, Ia, Ib, idx++);
                }
            }
        }
    }

  private:
    // ==> Class Data <==
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The symmetry of the state
    const int symmetry_;
    /// The symmetry of the orbitals
    const std::vector<std::vector<int>> orbital_symmetry_;
    /// The number of irreps
    size_t nirrep_;
    /// The total number of correlated molecular orbitals
    size_t norb_;
    /// The symmetry of the correlated molecular orbitals
    std::vector<int> cmo_sym_;
    /// The offset array for cmopi_
    std::vector<size_t> cmopi_offset_;

    /// The number of alpha strings
    size_t nas_;
    /// The number of beta strings
    size_t nbs_;
    /// The number of determinants
    size_t ndet_ = 0;
    /// The total number of orbital pairs per irrep
    std::vector<int> pairpi_;
    /// The offset array for pairpi
    std::vector<int> pair_offset_;
    /// The logging level for the class
    int log_level_ = 3;

    //   // GAS specific data

    std::vector<std::array<int, 6>> gas_alfa_occupations_;
    std::vector<std::array<int, 6>> gas_beta_occupations_;
    std::vector<std::array<int, 6>> gas_alfa_1h1p_occupations_;
    std::vector<std::array<int, 6>> gas_beta_1h1p_occupations_;
    std::vector<std::array<int, 6>> gas_alfa_1h_occupations_;
    std::vector<std::array<int, 6>> gas_beta_1h_occupations_;
    std::vector<std::array<int, 6>> gas_alfa_2h_occupations_;
    std::vector<std::array<int, 6>> gas_beta_2h_occupations_;
    std::vector<std::array<int, 6>> gas_alfa_3h_occupations_;
    std::vector<std::array<int, 6>> gas_beta_3h_occupations_;

    std::vector<std::pair<size_t, size_t>> gas_occupations_;

    /// The number of GAS spaces used
    size_t ngas_spaces_;
    /// The size of the GAS spaces (ignoring symmetry)
    std::vector<int> gas_size_;
    /// The position of the GAS spaces in the active space. For example, gas_mos_[0] gives the
    /// position of the MOs in the first GAS space in the active space
    std::vector<std::vector<size_t>> gas_mos_;
    /// The minimum number of electrons in each GAS space
    std::vector<int> gas_min_;
    /// The maximum number of electrons in each GAS space
    std::vector<int> gas_max_;
    /// @brief The number of determinants in each block
    std::vector<size_t> detpblk_;
    /// @brief The offset of each block
    std::vector<size_t> detpblk_offset_;

    // String lists
    std::shared_ptr<StringClass> string_class_;

    // Strings
    /// The alpha strings stored by class and address
    StringList alfa_strings_;
    /// The beta strings stored by class and address
    StringList beta_strings_;

    // String lists
    /// The VO string lists
    VOListMap alfa_vo_list;
    VOListMap beta_vo_list;
    VOListMap2 alfa_vo_list2;
    VOListMap2 beta_vo_list2;

    // Empty lists
    const VOListElement empty_vo_list;
    const VOListElement2 empty_vo_list2;

    /// The 1-hole lists
    H1List alfa_1h_list;
    H1List beta_1h_list;
    /// The 1-hole lists
    H1List2 alfa_1h_list2;
    H1List2 beta_1h_list2;
    /// The 2-hole lists
    H2List alfa_2h_list;
    H2List beta_2h_list;
    /// The 3-hole lists
    H3List alfa_3h_list;
    H3List beta_3h_list;

    const std::vector<H1StringSubstitution> empty_1h_list;
    const std::vector<std::vector<H1StringSubstitution>> empty_1h_list2;
    const std::vector<H2StringSubstitution> empty_2h_list;
    const std::vector<H3StringSubstitution> empty_3h_list;

    /// Addressers
    /// The alpha string address
    std::shared_ptr<StringAddress> alfa_address_;
    /// The beta string address
    std::shared_ptr<StringAddress> beta_address_;
    /// The alpha string address for 1h1p substitutions
    std::shared_ptr<StringAddress> alfa_address_1h1p_;
    /// The beta string address for 1h1p substitutions
    std::shared_ptr<StringAddress> beta_address_1h1p_;

    /// The alpha string address for N - 1 electrons
    std::shared_ptr<StringAddress> alfa_address_1h_;
    /// The beta string address for N - 1 electrons
    std::shared_ptr<StringAddress> beta_address_1h_;
    /// The alpha string address for N - 2 electrons
    std::shared_ptr<StringAddress> alfa_address_2h_;
    /// The beta string address for N - 2 electrons
    std::shared_ptr<StringAddress> beta_address_2h_;
    /// The alpha string address for N - 3 electrons
    std::shared_ptr<StringAddress> alfa_address_3h_;
    /// The beta string address for N - 3 electrons
    std::shared_ptr<StringAddress> beta_address_3h_;

    // == Private Functions ==
    template <typename HListT, typename T>
    const std::vector<T>& lookup_hole_list(const HListT& map_ref, int i, size_t add, int j) const {
        HListKey key{i, add, j};
        if (!map_ref.contains(key)) {
            if constexpr (std::is_same_v<T, H1StringSubstitution>) {
                // If the key is not found in the map, return an empty vector
                return empty_1h_list;
            } else if constexpr (std::is_same_v<T, H2StringSubstitution>) {
                return empty_2h_list;
            } else if constexpr (std::is_same_v<T, H3StringSubstitution>) {
                return empty_3h_list;
            } else {
                throw std::runtime_error("Unsupported HList type");
            }
        }
        return map_ref.at(key);
    }
};

// std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>
// find_string_map(const CIStrings& list_left, const CIStrings& list_right, bool alfa);

// VOListMap find_ov_string_map(const CIStrings& list_left, const CIStrings& list_right, bool alfa);

} // namespace forte2

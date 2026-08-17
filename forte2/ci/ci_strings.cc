#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

#include "helpers/printing.hpp"

#include "ci_occupation.h"
#include "ci_string_address.h"
#include "ci_strings_makers.h"

#include "ci_strings.h"

namespace forte2 {

// Global debug flag
bool debug_gas_strings = false;

// Wrapper function
template <typename Func> void debug(Func func) {
    if (debug_gas_strings) {
        func();
    }
}

CIStrings::CIStrings(size_t na, size_t nb, int symmetry,
                     const std::vector<std::vector<int>>& orbital_symmetry,
                     const std::vector<int>& gas_min, const std::vector<int>& gas_max)
    : na_(na), nb_(nb), symmetry_(symmetry), orbital_symmetry_(orbital_symmetry), gas_min_(gas_min),
      gas_max_(gas_max) {
    startup();
}

void CIStrings::startup() {
    // sanity checks
    if ((gas_min_.size() > orbital_symmetry_.size()) or
        gas_max_.size() > orbital_symmetry_.size()) {
        throw std::invalid_argument(
            "CIStrings: The number of GAS spaces specified by gas_min (" +
            std::to_string(gas_min_.size()) + ") and gas_max (" + std::to_string(gas_max_.size()) +
            ") must be less than or equal to the number of orbital symmetries (" +
            std::to_string(orbital_symmetry_.size()) + ")");
    }

    // set the number of GAS spaces and the number of correlated MOs
    ngas_spaces_ = orbital_symmetry_.size();
    nirrep_ = 1;
    {
        norb_ = 0;
        for (const auto& space : orbital_symmetry_) {
            const auto space_size = space.size();

            gas_size_.push_back(space_size);

            std::vector<size_t> gasn_mos(space_size);
            std::iota(gasn_mos.begin(), gasn_mos.end(), norb_);
            gas_mos_.push_back(gasn_mos);

            for (const auto& s : space) {
                nirrep_ = std::max(nirrep_, static_cast<size_t>(std::abs(s + 1)));
            }
            norb_ += space_size;
        }
    }

    // Generate the allowed GAS occupation patterns for alpha and beta string
    debug([&]() {
        std::cout << "CIStrings: Generating GAS occupation patterns for alpha and beta strings..."
                  << std::endl;
    });
    std::tie(ngas_spaces_, gas_alpha_occupations_, gas_beta_occupations_, gas_occupations_) =
        get_ci_occupation_patterns(na_, nb_, gas_min_, gas_max_, gas_size_);

    // Initialize the string class object. The string class is a combination of the irrep and GAS
    // occupation.
    string_class_ =
        std::make_shared<StringClass>(symmetry_, orbital_symmetry_, gas_alpha_occupations_,
                                      gas_beta_occupations_, gas_occupations_);

    // Build the string lists that satisfy the GAS constraints and the corresponding addressers
    debug([&]() {
        std::cout << "CIStrings: Generating string lists for alpha and beta strings..."
                  << std::endl;
    });
    alpha_strings_ = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                  gas_alpha_occupations_, string_class_);
    beta_strings_ = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                 gas_beta_occupations_, string_class_);

    alpha_address_ = std::make_shared<StringAddress>(gas_size_, na_, alpha_strings_);
    beta_address_ = std::make_shared<StringAddress>(gas_size_, nb_, beta_strings_);

    // Build the 1h1p string lists and their addressers. These strings can have occupation patterns
    // that do not fall under the GAS restriction. Used in the Knowles-Handy algorithm.
    gas_alpha_1h1p_occupations_ = generate_1h1p_occupations(gas_alpha_occupations_);
    gas_beta_1h1p_occupations_ = generate_1h1p_occupations(gas_beta_occupations_);

    auto alpha_strings_1h1p_ = make_strings_with_occupation(
        ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_alpha_1h1p_occupations_, string_class_);
    auto beta_strings_1h1p_ = make_strings_with_occupation(
        ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_beta_1h1p_occupations_, string_class_);

    alpha_address_1h1p_ = std::make_shared<StringAddress>(gas_size_, na_, alpha_strings_1h1p_);
    beta_address_1h1p_ = std::make_shared<StringAddress>(gas_size_, nb_, beta_strings_1h1p_);

    debug([&]() {
        std::cout
            << "CIStrings: Generating 1h, 2h, and 3h string lists for alpha and beta strings..."
            << std::endl;
    });

    // Build the 1h, 2h, and 3h occupation patterns and their corresponding string lists.
    const auto gas_alpha_1h_occupations_ = generate_1h_occupations(gas_alpha_occupations_);
    const auto gas_beta_1h_occupations_ = generate_1h_occupations(gas_beta_occupations_);
    const auto gas_alpha_2h_occupations_ = generate_1h_occupations(gas_alpha_1h_occupations_);
    const auto gas_beta_2h_occupations_ = generate_1h_occupations(gas_beta_1h_occupations_);
    const auto gas_alpha_3h_occupations_ = generate_1h_occupations(gas_alpha_2h_occupations_);
    const auto gas_beta_3h_occupations_ = generate_1h_occupations(gas_beta_2h_occupations_);

    auto alpha_1h_strings = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                         gas_alpha_1h_occupations_, string_class_);
    auto beta_1h_strings = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                        gas_beta_1h_occupations_, string_class_);
    alpha_address_1h_ = std::make_shared<StringAddress>(gas_size_, na_ - 1, alpha_1h_strings);
    beta_address_1h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 1, beta_1h_strings);

    auto alpha_2h_strings = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                         gas_alpha_2h_occupations_, string_class_);
    auto beta_2h_strings = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                        gas_beta_2h_occupations_, string_class_);
    alpha_address_2h_ = std::make_shared<StringAddress>(gas_size_, na_ - 2, alpha_2h_strings);
    beta_address_2h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 2, beta_2h_strings);

    auto alpha_3h_strings = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                         gas_alpha_3h_occupations_, string_class_);
    auto beta_3h_strings = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                        gas_beta_3h_occupations_, string_class_);
    alpha_address_3h_ = std::make_shared<StringAddress>(gas_size_, na_ - 3, alpha_3h_strings);
    beta_address_3h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 3, beta_3h_strings);

    // Initialize the number of strings and determinants
    nalpha_strings = 0;
    nbeta_strings = 0;
    for (int class_Ia = 0; class_Ia < alpha_address_->nclasses(); ++class_Ia) {
        nalpha_strings += alpha_address_->strpcls(class_Ia);
    }
    for (int class_Ib = 0; class_Ib < beta_address_->nclasses(); ++class_Ib) {
        nbeta_strings += beta_address_->strpcls(class_Ib);
    }

    // Initialize the number of determinants
    ndet_ = 0;
    for (const auto& [n, class_Ia, class_Ib] : determinant_classes()) {
        const auto nIa = alpha_address_->strpcls(class_Ia);
        const auto nIb = beta_address_->strpcls(class_Ib);
        const auto nI = nIa * nIb;
        ndet_per_block_.push_back(nI);
        ndet_per_block_offset_.push_back(ndet_);
        ndet_ += nI;
    }

    debug([&]() {
        std::cout << "CIStrings: Generating string substitution lists for alpha and beta strings..."
                  << std::endl;
    });

    alpha_vo_list = make_vo_list(alpha_strings_, alpha_address_, alpha_address_);
    beta_vo_list = make_vo_list(beta_strings_, beta_address_, beta_address_);

    alpha_vo_list2 = make_vo_list2(alpha_strings_1h1p_, alpha_address_1h1p_, alpha_address_);
    beta_vo_list2 = make_vo_list2(beta_strings_1h1p_, beta_address_1h1p_, beta_address_);

    alpha_1h_list = make_1h_list(alpha_strings_, alpha_address_, alpha_address_1h_);
    beta_1h_list = make_1h_list(beta_strings_, beta_address_, beta_address_1h_);

    alpha_1h_list2 = make_1h_list2(alpha_strings_, alpha_address_, alpha_address_1h_);
    beta_1h_list2 = make_1h_list2(beta_strings_, beta_address_, beta_address_1h_);

    alpha_2h_list = make_2h_list(alpha_strings_, alpha_address_, alpha_address_2h_);
    beta_2h_list = make_2h_list(beta_strings_, beta_address_, beta_address_2h_);

    alpha_3h_list = make_3h_list(alpha_strings_, alpha_address_, alpha_address_3h_);
    beta_3h_list = make_3h_list(beta_strings_, beta_address_, beta_address_3h_);
}

std::vector<Determinant> CIStrings::make_determinants() const {
    std::vector<Determinant> dets(ndet_);
    this->for_each_element([&](const size_t block, const int class_Ia, const int class_Ib,
                               const size_t Ia, const size_t Ib, const size_t idx) {
        Determinant I(alpha_str(class_Ia, Ia), beta_str(class_Ib, Ib));
        dets[idx] = I;
    });
    return dets;
}

size_t CIStrings::determinant_address(const Determinant& d) const {
    const auto Ia = d.a_string();
    const auto Ib = d.b_string();
    const auto& [addIa, class_Ia] = alpha_address_->address_and_class(Ia);
    const auto& [addIb, class_Ib] = beta_address_->address_and_class(Ib);
    const auto n = string_class_->block_index(class_Ia, class_Ib);
    return block_offset(n) + addIa * beta_address_->strpcls(class_Ib) + addIb;
}

Determinant CIStrings::determinant(size_t address) const {
    // find the irreps of alpha and beta strings
    size_t n = 0;
    size_t addI = 0;
    // keep adding the number of determinants in each irrep until we reach the right one
    for (size_t maxh = determinant_classes().size(); n < maxh; n++) {
        if (addI + ndet_per_block_[n] > address) {
            break;
        }
        addI += ndet_per_block_[n];
    }
    const size_t shift = address - addI;
    const auto& [_, class_Ia, class_Ib] = determinant_classes().at(n);
    const size_t beta_size = beta_address_->strpcls(class_Ib);
    const size_t addIa = shift / beta_size;
    const size_t addIb = shift % beta_size;
    String Ia{alpha_str(class_Ia, addIa)};
    String Ib{beta_str(class_Ib, addIb)};
    return Determinant(Ia, Ib);
}

/**
 * Returns a vector of tuples containing the sign, I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
const VOListElement& CIStrings::get_alpha_vo_list(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = alpha_vo_list.find(std::make_pair(class_I, class_J)); it != alpha_vo_list.end()) {
        return it->second;
    }
    return empty_vo_list;
}

/**
 * Returns a vector of tuples containing the sign,I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
const VOListElement& CIStrings::get_beta_vo_list(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = beta_vo_list.find(std::make_pair(class_I, class_J)); it != beta_vo_list.end()) {
        return it->second;
    }
    return empty_vo_list;
}

/**
 * Returns a vector of tuples containing the sign, I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
const VOListElement2& CIStrings::get_alpha_vo_list2(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = alpha_vo_list2.find(std::make_pair(class_I, class_J));
        it != alpha_vo_list2.end()) {
        return it->second;
    }
    return empty_vo_list2;
}

/**
 * Returns a vector of tuples containing the sign,I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
const VOListElement2& CIStrings::get_beta_vo_list2(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = beta_vo_list2.find(std::make_pair(class_I, class_J)); it != beta_vo_list2.end()) {
        return it->second;
    }
    return empty_vo_list2;
}

const std::vector<H1StringSubstitution>& CIStrings::get_alpha_1h_list(int class_I, size_t add_I,
                                                                      int class_J) const {
    return lookup_hole_list<H1List, H1StringSubstitution>(alpha_1h_list, class_I, add_I, class_J);
}

const std::vector<H1StringSubstitution>& CIStrings::get_beta_1h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H1List, H1StringSubstitution>(beta_1h_list, class_I, add_I, class_J);
}

const std::vector<std::vector<H1StringSubstitution>>&
CIStrings::get_alpha_1h_list2(int class_I, int class_J) const {
    // find the key in the map
    std::pair<size_t, size_t> key{class_I, class_J};
    if (auto it = alpha_1h_list2.find(key); it != alpha_1h_list2.end()) {
        return it->second;
    }
    return empty_1h_list2;
}

const std::vector<std::vector<H1StringSubstitution>>&
CIStrings::get_beta_1h_list2(int class_I, int class_J) const {
    // find the key in the map
    std::pair<size_t, size_t> key{class_I, class_J};
    if (auto it = beta_1h_list2.find(key); it != beta_1h_list2.end()) {
        return it->second;
    }
    return empty_1h_list2;
}

const std::vector<H2StringSubstitution>& CIStrings::get_alpha_2h_list(int class_I, size_t add_I,
                                                                      int class_J) const {
    return lookup_hole_list<H2List, H2StringSubstitution>(alpha_2h_list, class_I, add_I, class_J);
}

const std::vector<H2StringSubstitution>& CIStrings::get_beta_2h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H2List, H2StringSubstitution>(beta_2h_list, class_I, add_I, class_J);
}

const std::vector<H3StringSubstitution>& CIStrings::get_alpha_3h_list(int class_I, size_t add_I,
                                                                      int class_J) const {
    return lookup_hole_list<H3List, H3StringSubstitution>(alpha_3h_list, class_I, add_I, class_J);
}

const std::vector<H3StringSubstitution>& CIStrings::get_beta_3h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H3List, H3StringSubstitution>(beta_3h_list, class_I, add_I, class_J);
}

} // namespace forte2
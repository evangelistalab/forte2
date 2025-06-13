#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

#include "helpers/logger.h"

#include "ci_occupation.h"
#include "ci_string_address.h"
#include "ci_string_lists_makers.h"
#include "ci_string_lists.h"

namespace forte2 {

// Global debug flag
bool debug_gas_strings = true;

// Wrapper function
template <typename Func> void debug(Func func) {
    if (debug_gas_strings) {
        func();
    }
}

CIStrings::CIStrings(size_t na, size_t nb, int symmetry,
                     std::vector<std::vector<int>> orbital_symmetry, const std::vector<int> gas_min,
                     const std::vector<int> gas_max)
    : na_(na), nb_(nb), symmetry_(symmetry), orbital_symmetry_(orbital_symmetry), gas_min_(gas_min),
      gas_max_(gas_max) {

    std::vector<std::pair<int, int>> orbital_index_and_symmetry;
    {
        int k = 0;
        for (const auto& space : orbital_symmetry_) {
            for (const auto& s : space) {
                orbital_index_and_symmetry.emplace_back(k, s);
                k++;
            }
        }
        norb_ = k;
    }

    {
        int k = 0;
        for (int n{0}; const auto& space : orbital_symmetry_) {
            std::vector<size_t> gas_mos_n;
            for (const auto& s : space) {
                gas_mos_n.push_back(k);
                k++;
            }
            gas_mos_.push_back(gas_mos_n);
        }
    }

    nirrep_ = 1;
    ngas_spaces_ = orbital_symmetry_.size();
    for (const auto& space : orbital_symmetry_) {
        const auto size = space.size();
        gas_size_.push_back(size);
        LOG_INFO << "\n    GAS space size: " << size;
        for (const auto& s : space) {
            nirrep_ = std::max(nirrep_, static_cast<size_t>(std::abs(s + 1)));
        }
    }

    std::tie(ngas_spaces_, gas_alfa_occupations_, gas_beta_occupations_, gas_occupations_) =
        get_ci_occupation_patterns(na_, nb_, gas_min_, gas_max_, gas_size_);

    // print_h2("Possible Electron Occupations");
    auto table = occupation_table(ngas_spaces_, gas_alfa_occupations_, gas_beta_occupations_,
                                  gas_occupations_);
    LOG_INFO << table;

    // local_timers
    double str_list_timer = 0.0;
    double vo_list_timer = 0.0;
    double nn_list_timer = 0.0;
    double oo_list_timer = 0.0;
    double h1_list_timer = 0.0;
    double h2_list_timer = 0.0;
    double h3_list_timer = 0.0;
    double vovo_list_timer = 0.0;
    double vvoo_list_timer = 0.0;

    // this object is used to compute the class of a string (a generalization of the irrep)
    string_class_ =
        std::make_shared<StringClass>(symmetry_, orbital_symmetry_, gas_alfa_occupations_,
                                      gas_beta_occupations_, gas_occupations_);

    // Build the string lists and the addresser
    {
        // local_timer t;
        alfa_strings_ = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                     gas_alfa_occupations_, string_class_);
        beta_strings_ = make_strings_with_occupation(ngas_spaces_, nirrep_, gas_size_, gas_mos_,
                                                     gas_beta_occupations_, string_class_);

        alfa_address_ = std::make_shared<StringAddress>(gas_size_, na_, alfa_strings_);
        beta_address_ = std::make_shared<StringAddress>(gas_size_, nb_, beta_strings_);

        // str_list_timer += t.get();
    }

    // // from here down the code has to be rewritten to use the new StringAddress class

    gas_alfa_1h_occupations_ = generate_1h_occupations(gas_alfa_occupations_);
    gas_beta_1h_occupations_ = generate_1h_occupations(gas_beta_occupations_);
    gas_alfa_2h_occupations_ = generate_1h_occupations(gas_alfa_1h_occupations_);
    gas_beta_2h_occupations_ = generate_1h_occupations(gas_beta_1h_occupations_);
    gas_alfa_3h_occupations_ = generate_1h_occupations(gas_alfa_2h_occupations_);
    gas_beta_3h_occupations_ = generate_1h_occupations(gas_beta_2h_occupations_);

    if (na_ >= 1) {
        auto alfa_1h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_alfa_1h_occupations_, string_class_);
        alfa_address_1h_ = std::make_shared<StringAddress>(gas_size_, na_ - 1, alfa_1h_strings);
    }
    if (nb_ >= 1) {
        auto beta_1h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_beta_1h_occupations_, string_class_);
        beta_address_1h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 1, beta_1h_strings);
    }

    if (na_ >= 2) {
        auto alfa_2h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_alfa_2h_occupations_, string_class_);
        alfa_address_2h_ = std::make_shared<StringAddress>(gas_size_, na_ - 2, alfa_2h_strings);
    }
    if (nb_ >= 2) {
        auto beta_2h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_beta_2h_occupations_, string_class_);
        beta_address_2h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 2, beta_2h_strings);
    }
    if (na_ >= 3) {
        auto alfa_3h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_alfa_3h_occupations_, string_class_);
        alfa_address_3h_ = std::make_shared<StringAddress>(gas_size_, na_ - 3, alfa_3h_strings);
    }
    if (nb_ >= 3) {
        auto beta_3h_strings = make_strings_with_occupation(
            ngas_spaces_, nirrep_, gas_size_, gas_mos_, gas_beta_3h_occupations_, string_class_);
        beta_address_3h_ = std::make_shared<StringAddress>(gas_size_, nb_ - 3, beta_3h_strings);
    }

    nas_ = 0;
    nbs_ = 0;

    for (int class_Ia = 0; class_Ia < alfa_address_->nclasses(); ++class_Ia) {
        nas_ += alfa_address_->strpcls(class_Ia);
    }
    for (int class_Ib = 0; class_Ib < beta_address_->nclasses(); ++class_Ib) {
        nbs_ += beta_address_->strpcls(class_Ib);
    }

    ndet_ = 0;
    for (const auto& [n, class_Ia, class_Ib] : determinant_classes()) {
        const auto nIa = alfa_address_->strpcls(class_Ia);
        const auto nIb = beta_address_->strpcls(class_Ib);
        const auto nI = nIa * nIb;
        detpblk_.push_back(nI);
        detpblk_offset_.push_back(ndet_);
        ndet_ += nI;
    }

    alfa_vo_list = make_vo_list(alfa_strings_, alfa_address_, alfa_address_);
    beta_vo_list = make_vo_list(beta_strings_, beta_address_, beta_address_);

    alfa_vo_list2 = make_vo_list2(alfa_strings_, alfa_address_, alfa_address_);
    beta_vo_list2 = make_vo_list2(beta_strings_, beta_address_, beta_address_);

    alfa_1h_list = make_1h_list(alfa_strings_, alfa_address_, alfa_address_1h_);
    beta_1h_list = make_1h_list(beta_strings_, beta_address_, beta_address_1h_);

    alfa_1h_list2 = make_1h_list2(alfa_strings_, alfa_address_, alfa_address_1h_);
    beta_1h_list2 = make_1h_list2(beta_strings_, beta_address_, beta_address_1h_);

    alfa_2h_list = make_2h_list(alfa_strings_, alfa_address_, alfa_address_2h_);
    beta_2h_list = make_2h_list(beta_strings_, beta_address_, beta_address_2h_);

    alfa_3h_list = make_3h_list(alfa_strings_, alfa_address_, alfa_address_3h_);
    beta_3h_list = make_3h_list(beta_strings_, beta_address_, beta_address_3h_);
}

std::vector<Determinant> CIStrings::make_determinants() const {
    std::vector<Determinant> dets(ndet_);
    this->for_each_element([&](const size_t block, const int class_Ia, const int class_Ib,
                               const size_t Ia, const size_t Ib, const size_t idx) {
        Determinant I(alfa_str(class_Ia, Ia), beta_str(class_Ib, Ib));
        dets[idx] = I;
    });
    return dets;
}

size_t CIStrings::determinant_address(const Determinant& d) const {
    const auto Ia = d.get_alfa_bits();
    const auto Ib = d.get_beta_bits();

    const auto& [addIa, class_Ia] = alfa_address_->address_and_class(Ia);
    const auto& [addIb, class_Ib] = beta_address_->address_and_class(Ib);
    size_t addI = addIa * beta_address_->strpcls(class_Ib) + addIb;
    int n = string_class_->block_index(class_Ia, class_Ib);
    addI += detpblk_offset_[n];
    return addI;
}

Determinant CIStrings::determinant(size_t address) const {
    // find the irreps of alpha and beta strings
    size_t n = 0;
    size_t addI = 0;
    // keep adding the number of determinants in each irrep until we reach the right one
    for (size_t maxh = determinant_classes().size(); n < maxh; n++) {
        if (addI + detpblk_[n] > address) {
            break;
        }
        addI += detpblk_[n];
    }
    const size_t shift = address - addI;
    const auto& [_, class_Ia, class_Ib] = determinant_classes().at(n);
    const size_t beta_size = beta_address_->strpcls(class_Ib);
    const size_t addIa = shift / beta_size;
    const size_t addIb = shift % beta_size;
    String Ia = alfa_str(class_Ia, addIa);
    String Ib = beta_str(class_Ib, addIb);
    return Determinant(Ia, Ib);
}

// const OOListElement& CIStrings::get_alfa_oo_list(int class_I) const {
//     // check if the key exists, if not return an empty list
//     if (auto it = alfa_oo_list.find(class_I); it != alfa_oo_list.end()) {
//         return it->second;
//     }
//     return empty_oo_list;
// }

// const OOListElement& CIStrings::get_beta_oo_list(int class_I) const {
//     // check if the key exists, if not return an empty list
//     if (auto it = beta_oo_list.find(class_I); it != beta_oo_list.end()) {
//         return it->second;
//     }
//     return empty_oo_list;
// }

/**
 * Returns a vector of tuples containing the sign, I, and J connected by a^{+}_p
 * a_q
 * that is: J = ± a^{+}_p a_q I. p and q are absolute indices and I belongs to
 * the irrep h.
 */
const VOListElement& CIStrings::get_alfa_vo_list(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = alfa_vo_list.find(std::make_pair(class_I, class_J)); it != alfa_vo_list.end()) {
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
const VOListElement2& CIStrings::get_alfa_vo_list2(int class_I, int class_J) const {
    // check if the key exists, if not return an empty list
    if (auto it = alfa_vo_list2.find(std::make_pair(class_I, class_J)); it != alfa_vo_list2.end()) {
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

// const VVOOListElement& CIStrings::get_alfa_vvoo_list(int class_I, int class_J) const {
//     // check if the key exists, if not return an empty list
//     if (auto it = alfa_vvoo_list.find(std::make_pair(class_I, class_J));
//         it != alfa_vvoo_list.end()) {
//         return it->second;
//     }
//     return empty_vvoo_list;
// }

// const VVOOListElement& CIStrings::get_beta_vvoo_list(int class_I, int class_J) const {
//     // check if the key exists, if not return an empty list
//     if (auto it = beta_vvoo_list.find(std::make_pair(class_I, class_J));
//         it != beta_vvoo_list.end()) {
//         return it->second;
//     }
//     return empty_vvoo_list;
// }

const std::vector<H1StringSubstitution>& CIStrings::get_alfa_1h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H1List, H1StringSubstitution>(alfa_1h_list, class_I, add_I, class_J);
}

const std::vector<H1StringSubstitution>& CIStrings::get_beta_1h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H1List, H1StringSubstitution>(beta_1h_list, class_I, add_I, class_J);
}

const std::vector<std::vector<H1StringSubstitution>>&
CIStrings::get_alfa_1h_list2(int class_I, int class_J) const {
    // find the key in the map
    std::pair<size_t, size_t> key{class_I, class_J};
    if (auto it = alfa_1h_list2.find(key); it != alfa_1h_list2.end()) {
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

const std::vector<H2StringSubstitution>& CIStrings::get_alfa_2h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H2List, H2StringSubstitution>(alfa_2h_list, class_I, add_I, class_J);
}

const std::vector<H2StringSubstitution>& CIStrings::get_beta_2h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H2List, H2StringSubstitution>(beta_2h_list, class_I, add_I, class_J);
}

const std::vector<H3StringSubstitution>& CIStrings::get_alfa_3h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H3List, H3StringSubstitution>(alfa_3h_list, class_I, add_I, class_J);
}

const std::vector<H3StringSubstitution>& CIStrings::get_beta_3h_list(int class_I, size_t add_I,
                                                                     int class_J) const {
    return lookup_hole_list<H3List, H3StringSubstitution>(beta_3h_list, class_I, add_I, class_J);
}

} // namespace forte2
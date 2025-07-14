#pragma once

#include <memory>

#include "ci/ci_string_defs.h"
#include "ci/ci_string_address.h"
#include "ci/ci_string_class.h"

namespace forte2 {

/// Make strings of for norb bits with ne of these set to 1 and (norb - ne) set to 0
/// @return strings sorted according to their irrep
StringList make_strings_with_occupation(size_t num_spaces, int nirrep,
                                        const std::vector<int>& space_size,
                                        std::vector<std::vector<size_t>> space_mos,
                                        const std::vector<std::array<int, 6>>& occupations,
                                        std::shared_ptr<StringClass>& string_class);

/// Make the VO list
VOListMap make_vo_list(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
                       const std::shared_ptr<StringAddress>& J_addresser);

void make_vo(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
             const std::shared_ptr<StringAddress>& J_addresser, VOListMap& list, int p, int q);

/// Make the VO list
VOListMap2 make_vo_list2(const StringList& strings,
                         const std::shared_ptr<StringAddress>& I_addresser,
                         const std::shared_ptr<StringAddress>& J_addresser);

void make_vo2(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
              const std::shared_ptr<StringAddress>& J_addresser, VOListMap2& list, int p, int q);

/// Make 1-hole lists  a_p |I> = sign |J>
/// @param strings the list of strings {|I>}
/// @param address the addresser for the |I> strings
/// @param address_1h the addresser for the |J> strings
/// @return a map from (g_J, add_J, g_I) where g_J is the irrep of J, add_J is the address of a
///         string |J> and g_I is the irrep of the strings {|I>} to a list of substitutions of the
///         form (sign, p, add_I) where sign is the sign of the substitution, p is the index of the
///         orbital that is removed from |I> and add_I is the address of the string |I>.
H1List make_1h_list(const StringList& strings, std::shared_ptr<StringAddress> address,
                    std::shared_ptr<StringAddress> address_1h);

/// Make 2-hole lists a_p a_q |I> = sign |J> with p > q
/// @param strings the list of strings {|I>}
/// @param address the addresser for the |I> strings
/// @param address_2h the addresser for the |J> strings
/// @return a map from (g_J, add_J, g_I) where g_J is the irrep of J, add_J is the address of a
///         string |J> and g_I is the irrep of the strings {|I>} to a list of substitutions of the
///         form (sign, p, q, add_I) where sign is the sign of the substitution, p and q are the
///         indices of the orbitals that are removed from |I> and add_I is the address of the string
///         |I>.
H2List make_2h_list(const StringList& strings, std::shared_ptr<StringAddress> address,
                    std::shared_ptr<StringAddress> address_2h);

/// Make 3-hole lists a_p a_q a_r |I> = sign |J> with p > q > r
/// @param strings the list of strings {|I>}
/// @param address the addresser for the |I> strings
/// @param address_3h the addresser for the |J> strings
/// @return a map from (g_J, add_J, g_I) where g_J is the irrep of J, add_J is the address of a
///         string |J> and g_I is the irrep of the strings {|I>} to a list of substitutions of the
///         form (sign, p, q, r, add_I) where sign is the sign of the substitution, p, q, and r are
///         the indices of the orbitals that are removed from |I> and add_I is the address of the
///         string |I>.
H3List make_3h_list(const StringList& strings, std::shared_ptr<StringAddress> address,
                    std::shared_ptr<StringAddress> address_3h);

/// Make 1-hole lists (I -> a_p I = sgn J)
H1List2 make_1h_list2(const StringList& strings, std::shared_ptr<StringAddress> address,
                      std::shared_ptr<StringAddress> address_1h);

} // namespace forte2

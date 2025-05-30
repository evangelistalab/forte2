#pragma once

#include <memory>

#include "string_list_defs.h"
#include "ci_string_address.h"

namespace forte2 {

/// Make strings of for norb bits with ne of these set to 1 and (norb - ne) set to 0
/// @return strings sorted according to their irrep
StringList make_strings_with_occupation(size_t num_spaces, int nirrep,
                                        const std::vector<int>& space_size,
                                        std::vector<std::vector<size_t>> space_mos,
                                        const std::vector<std::array<int, 6>>& occupations,
                                        std::shared_ptr<StringClass>& string_class);
/// Make the pair list
PairList make_pair_list(size_t nirrep,
                        const std::vector<std::pair<int, int>>& orbital_index_and_symmetry);

/// Make the VO list
VOListMap make_vo_list(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
                       const std::shared_ptr<StringAddress>& J_addresser);

void make_vo(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
             const std::shared_ptr<StringAddress>& J_addresser, VOListMap& list, int p, int q);

OOListMap make_oo_list(const StringList& strings, std::shared_ptr<StringAddress> addresser);

void make_oo(const StringList& strings, OOListMap& list, int p, int q);

VVOOListMap make_vvoo_list(const StringList& strings, std::shared_ptr<StringAddress> address,
                           const std::vector<std::pair<int, int>>& orbital_index_and_symmetry);

void make_vvoo(const StringList& strings, std::shared_ptr<StringAddress> address, VVOOListMap& list,
               int p, int q, int r, int s);

/// Make 1-hole lists (I -> a_p I = sgn J)
H1List make_1h_list(const StringList& strings, std::shared_ptr<StringAddress> address,
                    std::shared_ptr<StringAddress> address_1h);
/// Make 2-hole lists (I -> a_p a_q I = sgn J)
H2List make_2h_list(const StringList& strings, std::shared_ptr<StringAddress> address,
                    std::shared_ptr<StringAddress> address_2h);
/// Make 3-hole lists (I -> a_p a_q a_r I = sgn J)
H3List make_3h_list(const StringList& strings, std::shared_ptr<StringAddress> address,
                    std::shared_ptr<StringAddress> address_3h);

} // namespace forte2

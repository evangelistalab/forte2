#include <algorithm>
#include <numeric>

#include "ci/ci_string_address.h"
#include "ci/ci_string_lists.h"
#include "ci/ci_string_lists_makers.h"

#include "helpers/cartesian_product.hpp"
#include "helpers/indexing.hpp"

namespace forte2 {

StringList make_strings_with_occupation(size_t num_spaces, int nirrep,
                                        const std::vector<int>& space_size,
                                        std::vector<std::vector<size_t>> space_mos,
                                        const std::vector<std::array<int, 6>>& occupations,
                                        std::shared_ptr<StringClass>& string_class) {
    auto list = StringList();
    for (const auto& occupation : occupations) {
        // Something to keep in mind: Here we use ACTIVE as a composite index, which means that we
        // will group the orbitals first by symmetry and then by space. For example, if we have the
        // following GAS: GAS1 = [A1 A1 A1 | A2 | B1 | B2 B2] GAS2 = [A1 | | B1 | B2 ] then the
        // composite space ACTIVE = GAS1 + GAS2 will be: ACTIVE = [A1 A1 A1 A1 | A2 | B1 B1 | B2 B2
        // B2]
        //           G1 G1 G1 G2   G1   G1 G1   G1 G1 G2

        // container to store the strings that generate a give gas space
        std::vector<std::vector<String>> gas_space_string(num_spaces, std::vector<String>{});
        std::vector<std::vector<String>> full_strings(nirrep, std::vector<String>{});

        // enumerate all the possible strings in each GAS space
        for (size_t n = 0; n < num_spaces; n++) {
            String I;
            auto gas_norb = space_size[n];
            auto gas_ne = occupation[n];
            if ((gas_ne >= 0) and (gas_ne <= gas_norb)) {
                const auto I_begin = I.begin();
                const auto I_end = I.begin() + gas_norb;

                I.clear();
                for (int i = std::max(0, gas_norb - gas_ne); i < gas_norb; ++i)
                    I[i] = true; // Generate the string 000011111

                do {
                    String J;
                    J.clear();
                    for (int i = 0; i < gas_norb; ++i) {
                        if (I[i])
                            J[space_mos[n][i]] = true;
                    }
                    gas_space_string[n].push_back(J);
                } while (std::next_permutation(I_begin, I_end));
            }
        }

        auto product_strings = math::cartesian_product(gas_space_string);

        for (const auto& strings : product_strings) {
            String I;
            I.clear();
            for (const auto& J : strings) {
                I |= J;
            }
            size_t sym_I = string_class->symmetry(I);
            full_strings[sym_I].push_back(I);
        }

        for (int h = 0; h < nirrep; h++) {
            list.push_back(full_strings[h]);
        }
    }
    return list;
}

VOListMap make_vo_list(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
                       const std::shared_ptr<StringAddress>& J_addresser) {
    VOListMap list;
    const int nmo = I_addresser->nbits();
    for (int p = 0; p < nmo; p++) {
        for (int q = 0; q < nmo; q++) {
            make_vo(strings, I_addresser, J_addresser, list, p, q);
        }
    }
    return list;
}

void make_vo(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
             const std::shared_ptr<StringAddress>& J_addresser, VOListMap& list, int p, int q) {
    for (const auto& string_class : strings) {
        for (const auto& I : string_class) {
            const auto& [add_I, class_I] = I_addresser->address_and_class(I);
            auto J = I;
            double sign = 1.0;
            if (J[q]) {
                sign *= J.slater_sign(q);
                J[q] = false;
                if (!J[p]) {
                    sign *= J.slater_sign(p);
                    J[p] = true;
                    if (auto it = J_addresser->find(J); it != J_addresser->end()) {
                        const auto& [add_J, class_J] = it->second;
                        auto& list_IJ = list[std::make_pair(class_I, class_J)];
                        list_IJ[std::make_tuple(p, q)].push_back(
                            StringSubstitution(sign, add_I, add_J));
                    }
                }
            }
        }
    }
}

VOListMap2 make_vo_list2(const StringList& strings,
                         const std::shared_ptr<StringAddress>& I_addresser,
                         const std::shared_ptr<StringAddress>& J_addresser) {
    VOListMap2 list;
    auto nclasses_I = I_addresser->nclasses();
    auto nclasses_J = J_addresser->nclasses();
    for (size_t class_I = 0; class_I < nclasses_I; ++class_I) {
        size_t nstr = I_addresser->strpcls(class_I); // ensure that the addresser has all classes
        for (size_t class_J = 0; class_J < nclasses_J; ++class_J) {
            list[std::make_pair(class_I, class_J)] =
                std::vector<std::vector<StringSubstitution2>>(nstr);
        }
    }
    const int nmo = I_addresser->nbits();
    for (int p = 0; p < nmo; p++) {
        for (int q = 0; q < nmo; q++) {
            make_vo2(strings, I_addresser, J_addresser, list, p, q);
        }
    }
    return list;
}

void make_vo2(const StringList& strings, const std::shared_ptr<StringAddress>& I_addresser,
              const std::shared_ptr<StringAddress>& J_addresser, VOListMap2& list, int p, int q) {

    for (const auto& string_class : strings) {
        for (const auto& I : string_class) {
            const auto& [add_I, class_I] = I_addresser->address_and_class(I);
            auto J = I;
            double sign = 1.0;
            if (J[q]) {
                sign *= J.slater_sign(q);
                J[q] = false;
                if (!J[p]) {
                    sign *= J.slater_sign(p);
                    J[p] = true;
                    if (auto it = J_addresser->find(J); it != J_addresser->end()) {
                        const auto& [add_J, class_J] = it->second;
                        auto& list_IJ = list[std::make_pair(class_I, class_J)];
                        // cast sign to integer and scale if diagonal
                        double scaled_sign = sign * (p == q ? 2. : 1.);
                        // use a single index for (p, q)
                        size_t pq = static_cast<size_t>(pair_index_geq(p, q));
                        list_IJ[add_I].push_back(StringSubstitution2(scaled_sign, pq, add_J));
                    }
                }
            }
        }
    }
}

H1List2 make_1h_list2(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                      std::shared_ptr<StringAddress> addresser_1h) {
    H1List2 list;
    int n = addresser->nbits();
    int k = addresser->nones();
    size_t nmo = addresser->nbits();

    // for (const auto& string_class_I : strings) {
    //     for (const auto& I : string_class_I) {
    //         const auto& [add_I, class_I] = addresser->address_and_class(I);
    //         for (const auto& string_class_J : strings) {
    //             for (const auto& J : string_class_J) {
    //                 const auto& [add_J, class_J] = addresser->address_and_class(J);
    //             }
    //         }
    //     }
    // }

    if (n == 0)
        return list; // if n is 0, return an empty list

    auto nclasses_1h = addresser_1h->nclasses();
    auto nclasses = addresser->nclasses();
    for (size_t h = 0; h < nclasses_1h; ++h) {
        size_t nstr_1h = addresser_1h->strpcls(h); // ensure that the addresser_1h has all classes
        for (size_t h2 = 0; h2 < nclasses; ++h2) {
            list[std::make_pair(h, h2)] = std::vector<std::vector<H1StringSubstitution>>(nstr_1h);
        }
    }

    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (const auto& string_class : strings) {
            for (const auto& I : string_class) {
                const auto& [add_I, class_I] = addresser->address_and_class(I);
                for (size_t p = 0; p < nmo; ++p) {
                    if (I[p]) {
                        auto J = I;
                        const auto sign = J.slater_sign(p);
                        J[p] = false;
                        if (auto it = addresser_1h->find(J); it != addresser_1h->end()) {
                            const auto& [add_J, class_J] = it->second;
                            std::pair<size_t, size_t> I_tuple(class_J, class_I);
                            list[I_tuple][add_J].push_back(H1StringSubstitution(sign, p, add_I));
                        }
                    }
                }
            }
        }
    }
    return list;
}

H1List make_1h_list(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                    std::shared_ptr<StringAddress> addresser_1h) {
    H1List list;
    int n = addresser->nbits();
    int k = addresser->nones();
    size_t nmo = addresser->nbits();
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (const auto& string_class : strings) {
            for (const auto& I : string_class) {
                // std::cout << "String " < < < < std::endl;
                const auto& [add_I, class_I] = addresser->address_and_class(I);
                for (size_t p = 0; p < nmo; ++p) {
                    if (I[p]) {
                        auto J = I;
                        const auto sign = J.slater_sign(p);
                        J[p] = false;
                        if (auto it = addresser_1h->find(J); it != addresser_1h->end()) {
                            const auto& [add_J, class_J] = it->second;
                            std::tuple<int, size_t, int> I_tuple(class_J, add_J, class_I);
                            list[I_tuple].push_back(H1StringSubstitution(sign, p, add_I));
                        }
                    }
                }
            }
        }
    }
    return list;
}

H2List make_2h_list(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                    std::shared_ptr<StringAddress> addresser_2h) {
    H2List list;
    int n = addresser->nbits();
    int k = addresser->nones();
    size_t nmo = addresser->nbits();
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (const auto& string_class : strings) {
            for (const auto& I : string_class) {
                const auto& [add_I, class_I] = addresser->address_and_class(I);
                for (size_t q = 0; q < nmo; ++q) {
                    for (size_t p = q + 1; p < nmo; ++p) {
                        if (I[p] and I[q]) {
                            auto J = I;
                            J[q] = false;
                            const auto q_sign = J.slater_sign(q);
                            J[p] = false;
                            const auto p_sign = J.slater_sign(p);
                            if (auto it = addresser_2h->find(J); it != addresser_2h->end()) {
                                const auto sign = p_sign * q_sign;
                                const auto& [add_J, class_J] = it->second;
                                std::tuple<int, size_t, int> I_tuple(class_J, add_J, class_I);
                                list[I_tuple].push_back(H2StringSubstitution(sign, p, q, add_I));
                            }
                        }
                    }
                }
            }
        }
    }
    return list;
}

H3List make_3h_list(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                    std::shared_ptr<StringAddress> addresser_3h) {
    H3List list;
    int n = addresser->nbits();
    int k = addresser->nones();
    size_t nmo = addresser->nbits();
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        for (const auto& string_class : strings) {
            for (const auto& I : string_class) {
                const auto& [add_I, class_I] = addresser->address_and_class(I);
                for (size_t r = 0; r < nmo; ++r) {
                    for (size_t q = r + 1; q < nmo; ++q) {
                        for (size_t p = q + 1; p < nmo; ++p) {
                            if (I[p] and I[q] and I[r]) {
                                auto J = I;
                                J[r] = false;
                                const auto r_sign = J.slater_sign(r);
                                J[q] = false;
                                const auto q_sign = J.slater_sign(q);
                                J[p] = false;
                                const auto p_sign = J.slater_sign(p);
                                if (auto it = addresser_3h->find(J); it != addresser_3h->end()) {
                                    const auto sign = p_sign * q_sign * r_sign;
                                    const auto& [add_J, class_J] = it->second;
                                    std::tuple<int, size_t, int> I_tuple(class_J, add_J, class_I);
                                    list[I_tuple].push_back(
                                        H3StringSubstitution(sign, p, q, r, add_I));
                                    // list[I_tuple].push_back(
                                    //     H3StringSubstitution(-sign, p, r, q, add_I));
                                    // list[I_tuple].push_back(
                                    //     H3StringSubstitution(-sign, q, p, r, add_I));
                                    // list[I_tuple].push_back(
                                    //     H3StringSubstitution(+sign, q, r, p, add_I));
                                    // list[I_tuple].push_back(
                                    //     H3StringSubstitution(-sign, r, q, p, add_I));
                                    // list[I_tuple].push_back(
                                    //     H3StringSubstitution(+sign, r, p, q, add_I));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return list;
}

std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>
find_string_map(const CIStrings& list_left, const CIStrings& list_right, bool alfa) {
    std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> m;
    const auto& strings_right = alfa ? list_right.beta_strings() : list_right.alfa_strings();
    const auto& address_left = alfa ? list_left.beta_address() : list_left.alfa_address();
    // loop over all the right string classes (I)
    for (int class_I{0}; const auto& string_class_right : strings_right) {
        // loop over all the right strings (I)
        for (size_t addI{0}; const auto& I : string_class_right) {
            // find the left string class (class_J) and string address (addJ) of the
            // string J = I
            if (auto it = address_left->find(I); it != address_left->end()) {
                const auto& [addJ, class_J] = it->second;
                m[std::make_pair(class_I, class_J)].push_back(std::make_pair(addI, addJ));
            }
            addI++;
        }
        class_I++;
    }
    return m;
}

VOListMap find_ov_string_map(const CIStrings& list_left, const CIStrings& list_right, bool alfa) {
    const auto& strings_right = alfa ? list_right.alfa_strings() : list_right.beta_strings();
    const auto& I_address = alfa ? list_right.alfa_address() : list_right.beta_address();
    const auto& J_address = alfa ? list_left.alfa_address() : list_left.beta_address();
    auto vo_list = make_vo_list(strings_right, I_address, J_address);
    return vo_list;
}

} // namespace forte2

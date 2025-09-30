
#include "helpers/logger.h"
#include "helpers/timer.hpp"
#include "helpers/unordered_dense.h"
#include "helpers/sorting.hpp"
#include "helpers/ndarray.h"
#include "helpers/np_vector_functions.h"

#include "determinant_helpers.h"
#include "sci_helper.h"

namespace forte2 {

SortedStringList::SortedStringList(size_t norb, std::vector<Determinant>&& dets)
    : norb_(norb), sorted_dets_(std::move(dets)) {

    det_permutation_ = sort_permutation(sorted_dets_, Determinant::reverse_less_than);
    apply_permutation_in_place(sorted_dets_, det_permutation_);

    ndets_ = sorted_dets_.size();

    size_t i = 0;
    String first_string = sorted_dets_[0].a_string();
    String second_string = sorted_dets_[0].b_string();
    String old_first_string = first_string;

    first_string_range_.push_back(std::make_pair(i, i + 1));
    sorted_first_string_.push_back(first_string);
    sorted_second_string_.push_back(second_string);
    first_string_index_[first_string] = 0;
    second_string_index_[second_string] = 0;
    sorted_dets_second_string_.push_back(second_string_index_[second_string]);

    for (size_t j{1}; j < ndets_; j++) {
        first_string = sorted_dets_[j].a_string();
        second_string = sorted_dets_[j].b_string();
        if (second_string_index_.find(second_string) == second_string_index_.end()) {
            second_string_index_[second_string] = second_string_index_.size();
            sorted_second_string_.push_back(second_string);
        }
        sorted_dets_second_string_.push_back(second_string_index_[second_string]);
        // if the first string changed, store the range
        if (first_string != old_first_string) {
            first_string_range_[i].second = j; // end of the range
            // start a new range
            i++;
            first_string_range_.push_back(std::make_pair(j, j + 1));
            sorted_first_string_.push_back(first_string);
            first_string_index_[first_string] = i;
            old_first_string = first_string;
        }
    }
    // set the end of the last range
    first_string_range_[i].second = ndets_;

    second_string_to_det_index_.reserve(first_string_range_.size());

    for (size_t k{0}; const auto [start, end] : first_string_range_) {
        ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>> map;
        for (size_t idx{start}; idx < end; ++idx) {
            map[sorted_dets_second_string_[idx]] =
                det_permutation_[k]; // use the original index here
            k++;
        }
        second_string_to_det_index_.push_back(std::move(map));
    }

    one_hole_string_list_.reserve(sorted_first_string_.size());
    std::vector<size_t> occ(norb_, 0); // at most norb occupied orbitals
    for (size_t i = 0, imax{sorted_first_string_.size()}; i < imax; ++i) {
        const auto& first_str = sorted_first_string_[i];
        // Find the occupied orbitals in the first string
        size_t n = 0;
        first_str.find_set_bits(occ, n);

        std::vector<std::tuple<size_t, size_t, double>> one_hole_string_list_entry;
        one_hole_string_list_entry.reserve(n);

        // For each occupied orbital, create the one-hole string and store it
        for (size_t p = 0; p < n; ++p) {
            const size_t orb = occ[p];
            String one_hole = first_str;
            one_hole.set_bit(orb, false);
            if (one_hole_strings_index_.find(one_hole) == one_hole_strings_index_.end()) {
                one_hole_strings_index_[one_hole] = one_hole_strings_index_.size();
                one_hole_strings_.push_back(one_hole);
            }
            const size_t hole_idx = one_hole_strings_index_[one_hole];
            const double sign = first_str.slater_sign(orb);

            one_hole_string_list_entry.emplace_back(orb, hole_idx, sign);
        }
        one_hole_string_list_.emplace_back(std::move(one_hole_string_list_entry));
    }

    // Create the inverse mapping from one-hole strings to full strings
    one_hole_string_list_inv_.resize(one_hole_strings_index_.size());
    for (size_t i = 0, imax{sorted_first_string_.size()}; i < imax; ++i) {
        for (const auto& [orb, hole_idx, sign] : one_hole_string_list_[i]) {
            one_hole_string_list_inv_[hole_idx].emplace_back(orb, i, sign);
        }
    }

    two_hole_string_list_.reserve(sorted_first_string_.size());
    for (size_t i = 0, imax{sorted_first_string_.size()}; i < imax; ++i) {
        const auto& first_str = sorted_first_string_[i];
        // Find the occupied orbitals in the first string
        size_t n = 0;
        first_str.find_set_bits(occ, n);

        std::vector<std::tuple<size_t, size_t, size_t, double>> two_hole_string_list_entry;
        two_hole_string_list_entry.reserve(n * (n - 1) / 2);

        // For each pair of occupied orbitals, create the two-hole string and store it (p < q)
        for (size_t p = 0; p < n; ++p) {
            const size_t orb_p = occ[p];
            for (size_t q = p + 1; q < n; ++q) {
                const size_t orb_q = occ[q];
                String two_hole = first_str;
                double sign = 1.0;
                two_hole.set_bit(orb_p, false);
                sign *= two_hole.slater_sign(orb_p);
                two_hole.set_bit(orb_q, false);
                sign *= two_hole.slater_sign(orb_q);
                if (two_hole_strings_index_.find(two_hole) == two_hole_strings_index_.end()) {
                    two_hole_strings_index_[two_hole] = two_hole_strings_index_.size();
                    two_hole_strings_.push_back(two_hole);
                }
                const size_t hole_idx = two_hole_strings_index_[two_hole];
                two_hole_string_list_entry.emplace_back(orb_p, orb_q, hole_idx, sign);
            }
        }
        two_hole_string_list_.emplace_back(std::move(two_hole_string_list_entry));
    }

    // Create the inverse mapping from two-hole strings to full strings
    two_hole_string_list_inv_.resize(two_hole_strings_index_.size());
    for (size_t i = 0, imax{sorted_first_string_.size()}; i < imax; ++i) {
        for (const auto& [orb_p, orb_q, hole_idx, sign] : two_hole_string_list_[i]) {
            two_hole_string_list_inv_[hole_idx].emplace_back(orb_p, orb_q, i, sign);
        }
    }
}

void SelectedCIHelper::prepare_sigma_build() {
    // compute the energy of all the determinants
    det_energies_.resize(dets_.size());
    for (size_t i{0}, i_max{dets_.size()}; i < i_max; ++i) {
        det_energies_[i] = slater_rules_.energy(dets_[i]);
    }

    // Make the sorted lists of determinants for alpha-beta and beta-alpha
    // Alpha-Beta
    std::vector<Determinant> ab_dets_ = dets_;
    ab_list_ = SortedStringList(norb_, std::move(ab_dets_));

    // Beta-Alpha
    std::vector<Determinant> ba_dets_;
    ba_dets_.reserve(dets_.size());
    for (const auto& d : dets_) {
        ba_dets_.emplace_back(d.spin_flip());
    }
    ba_list_ = SortedStringList(norb_, std::move(ba_dets_));

    for (auto& list : {ab_list_, ba_list_}) {
        auto& dets = list.sorted_dets();
        for (size_t i = 0; i < dets.size(); ++i) {
            LOG(log_level_) << str(dets[i], norb_) << " " << i;
        }

        for (size_t i = 0; i < list.first_string_size(); ++i) {
            auto [start, end] = list.range(i);
            LOG(log_level_) << "Alpha string: " << str(list.sorted_first_string(i), norb_)
                            << " Range: [" << start << ", " << end << ")";
            for (size_t j = start; j < end; ++j) {
                auto idx = list.sorted_dets_second_string(j);
                Determinant d = list.sorted_dets()[j];
                LOG(log_level_) << "   " << str(list.sorted_first_string(i), norb_) << " x "
                                << str(list.sorted_second_string(idx), norb_);
                LOG(log_level_) << "   " << str(d, norb_)
                                << " Index: " << list.second_string_to_det_index()[i].at(idx);
            }
        }
    }
    fullHamiltonian();
}

np_matrix SelectedCIHelper::fullHamiltonian() const {
    auto H = make_zeros<nb::numpy, double, 2>({dets_.size(), dets_.size()});
    std::vector<double> vec_basis(dets_.size(), 0.0);
    std::vector<double> vec_sigma(dets_.size(), 0.0);
    auto basis = std::span<double>(vec_basis.data(), vec_basis.size());
    auto sigma = std::span<double>(vec_sigma.data(), vec_sigma.size());
    for (size_t i{0}; i < dets_.size(); ++i) {
        basis[i] = 1.0;
        H0(basis, sigma);
        H1a(basis, sigma);
        // H1b(basis, sigma);
        for (size_t j{0}; j < dets_.size(); ++j) {
            H(i, j) = sigma[j];
        }
        basis[i] = 0.0;
        for (size_t j{0}; j < dets_.size(); ++j) {
            sigma[j] = 0.0;
        }
    }
    return H;
}

void SelectedCIHelper::Hamiltonian(np_vector basis, np_vector sigma) const {
    local_timer t;
    vector::zero<double>(sigma);
    auto b_span = vector::as_span<double>(basis);
    auto s_span = vector::as_span<double>(sigma);

    H0(b_span, s_span);
    // H1(b_span, s_span, Spin::Alpha);
    // H1(b_span, s_span, Spin::Beta);
    // H2_same_spin(b_span, s_span, Spin::Alpha);
    // H2_same_spin(b_span, s_span, Spin::Beta);
    // H2_opposite_spin(b_span, s_span);
}

void SelectedCIHelper::H0(std::span<double> basis, std::span<double> sigma) const {
    // H0 is diagonal in the determinant basis
    for (size_t i{0}, i_max{dets_.size()}; i < i_max; ++i) {
        sigma[i] = det_energies_[i] * basis[i];
    }
}

void SelectedCIHelper::find_matching_dets(std::span<double> basis, std::span<double> sigma,
                                          const SortedStringList& list, size_t i, size_t j,
                                          double int_sign) const {
    // Find the range of determinants with the current alpha string
    const auto& [istart, iend] = list.range(i);
    const auto& [jstart, jend] = list.range(j);

    if (iend - istart >= jend - jstart) {
        const auto& i_second_string_to_det_index = list.second_string_to_det_index()[i];
        // Loop over the determinants in the smaller range
        for (size_t jj{jstart}; jj < jend; ++jj) {
            // Get the index of the jj determinant in the full list
            const auto idx_j = list.sorted_dets_second_string(jj);
            // Check if the determinant with the new beta string exists
            const auto it = i_second_string_to_det_index.find(idx_j);
            if (it != i_second_string_to_det_index.end()) {
                const size_t ii = it->second;
                // LOG(log_level_)
                //     << "a+(" << q << ") a(" << p << ") " << str(dets_[ii], norb_)
                //     << " -> " << sign << " * " << h_pq << " " << str(dets_[jj],
                //     norb_);
                sigma[ii] += int_sign * basis[jj];
            }
        }
    } else {
        const auto& j_second_string_to_det_index = list.second_string_to_det_index()[j];
        for (size_t ii{istart}; ii < iend; ++ii) {
            const auto idx_i = list.sorted_dets_second_string(ii);
            // Check if the determinant with the new beta string exists
            const auto it = j_second_string_to_det_index.find(idx_i);
            if (it != j_second_string_to_det_index.end()) {
                const size_t jj = it->second;
                sigma[ii] += int_sign * basis[jj];
            }
        }
    }
}

void SelectedCIHelper::H1a(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();
    const auto& one_hole_strings = ab_list_.one_hole_strings();
    LOG(log_level_) << "Number of unique alpha strings: " << first_string_size;
    LOG(log_level_) << "Number of unique one-hole alpha strings: " << one_hole_strings.size();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& i_second_string_to_det_index = ab_list_.second_string_to_det_index()[i];
        const auto& sublist = ab_list_.one_hole_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            LOG(log_level_) << "a(" << p << ") " << str(ab_list_.sorted_first_string(i), norb_)
                            << " -> " << sign_p << " " << str(one_hole_strings[hole_idx], norb_);
            // Find the corresponding beta string
            const auto& inv_sublist = ab_list_.one_hole_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                if (p == q)
                    continue; // skip same orbital
                // LOG(log_level_) << "a(" << q << ")+" << str(one_hole_strings[hole_idx], norb_)
                //                 << " -> " << sign_q << " "
                //                 << str(ab_list_.sorted_first_string(j), norb_);
                const double sign = sign_p * sign_q;
                const double h_pq = h_[p * norb_ + q];
                if (std::abs(h_pq) < 1e-12)
                    continue;

                find_matching_dets(basis, sigma, ab_list_, i, j, h_pq * sign);

                // // Find the range of determinants with the current alpha string
                // auto [istart, iend] = ab_list_.range(i);
                // auto [jstart, jend] = ab_list_.range(j);

                // const auto& j_second_string_to_det_index =
                //     ab_list_.second_string_to_det_index()[j];

                // if (iend - istart >= jend - jstart) {
                //     // Loop over the determinants in the smaller range
                //     for (size_t jj{jstart}; jj < jend; ++jj) {
                //         // Get the index of the jj determinant in the full list
                //         const auto idx_j = ab_list_.sorted_dets_second_string(jj);
                //         // Check if the determinant with the new beta string exists
                //         const auto it = i_second_string_to_det_index.find(idx_j);
                //         if (it != i_second_string_to_det_index.end()) {
                //             const size_t ii = it->second;
                //             // LOG(log_level_)
                //             //     << "a+(" << q << ") a(" << p << ") " << str(dets_[ii], norb_)
                //             //     << " -> " << sign << " * " << h_pq << " " << str(dets_[jj],
                //             //     norb_);
                //             sigma[ii] += h_pq * sign * basis[jj];
                //         }
                //     }
                // } else {
                //     for (size_t ii{istart}; ii < iend; ++ii) {
                //         const auto idx_i = ab_list_.sorted_dets_second_string(ii);
                //         // Check if the determinant with the new beta string exists
                //         const auto it = ab_list_.second_string_to_det_index()[j].find(idx_i);
                //         if (it != ab_list_.second_string_to_det_index()[j].end()) {
                //             const size_t jj = it->second;
                //             sigma[ii] += h_pq * sign * basis[jj];
                //         }
                //     }
                // }
            }
        }
    }
}

void SelectedCIHelper::H1b(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ba_list_.first_string_size();
    const auto& one_hole_strings = ba_list_.one_hole_strings();
    LOG(log_level_) << "Number of unique beta strings: " << first_string_size;
    LOG(log_level_) << "Number of unique one-hole beta strings: " << one_hole_strings.size();
    // Loop over all unique beta strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& i_second_string_to_det_index = ba_list_.second_string_to_det_index()[i];
        const auto& sublist = ba_list_.one_hole_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            // LOG(log_level_) << "a(" << p << ") " << str(ba_list_.sorted_first_string(i), norb_)
            //                 << " -> " << sign_p << " " << str(one_hole_strings[hole_idx], norb_);
            // Find the corresponding beta string
            const auto& inv_sublist = ba_list_.one_hole_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                if (p == q)
                    continue; // skip same orbital
                const double sign = sign_p * sign_q;
                const double h_pq = h_[p * norb_ + q];
                if (std::abs(h_pq) < 1e-12)
                    continue;

                find_matching_dets(basis, sigma, ba_list_, i, j, h_pq * sign);
            }
        }
    }
}

void SelectedCIHelper::H2a(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();
    const auto& two_hole_strings = ab_list_.two_hole_strings();
    LOG(log_level_) << "Number of unique alpha strings: " << first_string_size;
    LOG(log_level_) << "Number of unique two-hole alpha strings: " << two_hole_strings.size();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& i_second_string_to_det_index = ab_list_.second_string_to_det_index()[i];
        const auto& sublist = ab_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) {
            LOG(log_level_) << "a(" << q << ") "
                            << "a(" << p << ") " << str(ab_list_.sorted_first_string(i), norb_)
                            << " -> " << sign_pq << " " << str(two_hole_strings[hole_idx], norb_);
            // Find the corresponding beta string
            const auto& inv_sublist = ab_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) {
                if ((p == r) and (q == s))
                    continue; // skip the term that would give back the original determinant
                // LOG(log_level_) << "a(" << q << ")+" << str(one_hole_strings[hole_idx], norb_)
                //                 << " -> " << sign_q << " "
                //                 << str(ab_list_.sorted_first_string(j), norb_);
                const double sign = sign_pq * sign_rs;
                const double v_pqrs = Va(p, q, r, s);
                if (std::abs(v_pqrs) < 1e-12)
                    continue;

                find_matching_dets(basis, sigma, ab_list_, i, j, v_pqrs * sign);
            }
        }
    }
}

void SelectedCIHelper::H2b(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ba_list_.first_string_size();
    const auto& two_hole_strings = ba_list_.two_hole_strings();
    // Loop over all unique beta strings
    for (size_t i{0}; i < first_string_size; ++i) {
        // find the two-hole strings for this beta string
        const auto& sublist = ba_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) {
            // find the connected beta strings by adding back two electrons
            const auto& inv_sublist = ba_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) {
                // |j> = sign_pq * sign_rs a^_s a^_r a_q a_p |i> (?)
                if ((p == r) and (q == s))
                    continue; // skip the term that would give back the original determinant
                const double sign = sign_pq * sign_rs;
                const double v_pqrs = Va(p, q, r, s);
                if (std::abs(v_pqrs) < 1e-12)
                    continue;
                find_matching_dets(basis, sigma, ba_list_, i, j, v_pqrs * sign);
            }
        }
    }
}

void SelectedCIHelper::H2ab(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();
}

} // namespace forte2

/*
SortedStringList::SortedStringList() {}

SortedStringList::SortedStringList(size_t nmo, const DeterminantHashVec& space,
                                   DetSpinType sorted_string_spin)
    : nmo_(nmo) {
    // Copy and sort the determinants
    auto dets = space.determinants();
    num_dets_ = dets.size();
    sorted_dets_.reserve(num_dets_);
    for (const auto& d : dets) { // TODO: this appears redundant now (Francesco)
        sorted_dets_.push_back(d);
    }
    if (sorted_string_spin == DetSpinType::Alpha) {
        map_to_hashdets_ = sort_permutation(sorted_dets_, Determinant::reverse_less_than);
        apply_permutation_in_place(sorted_dets_, map_to_hashdets_);
        //        std::sort(sorted_dets_.begin(), sorted_dets_.end(),
        //        UI64Determinant::reverse_less_then);
    } else {
        map_to_hashdets_ = sort_permutation(sorted_dets_, Determinant::less_than);
        apply_permutation_in_place(sorted_dets_, map_to_hashdets_);
        //        std::sort(sorted_dets_.begin(), sorted_dets_.end());
    }

    //    outfile->Printf("\n\n Sorted determinants (%zu,%s)\n", num_dets_,
    //                    sorted_string_spin == DetSpinType::Alpha ? "Alpha" : "Beta");
    // Find the unique strings and their range

    sorted_spin_type_ =
        sorted_string_spin == DetSpinType::Alpha ? DetSpinType::Alpha : DetSpinType::Beta;
    String first_string = sorted_dets_[0].get_bits(sorted_spin_type_);
    String old_first_string = first_string;

    first_string_range_[old_first_string] = std::make_pair(0, 0);
    sorted_half_dets_.push_back(old_first_string);

    //    outfile->Printf("\n %6d %s", 0, sorted_dets_[0].str2().c_str());
    size_t min_per_string = std::numeric_limits<std::size_t>::max();
    size_t max_per_string = 0;
    for (size_t i = 1; i < num_dets_; i++) {
        //        outfile->Printf("\n %6d %s", i, sorted_dets_[i].str2().c_str());
        first_string = sorted_dets_[i].get_bits(sorted_spin_type_);

        //        first_string.set_bits(sorted_dets_[i].bits() & half_bit_mask.bits());
        if (not(first_string == old_first_string)) {
            first_string_range_[old_first_string].second = i;
            first_string_range_[first_string] = std::make_pair(i, 0);
            //            outfile->Printf(" <- new determinant (%zu -> %zu)",
            //                            first_string_range_[old_first_string].first,
            //                            first_string_range_[old_first_string].second);
            old_first_string = first_string;
            sorted_half_dets_.push_back(first_string);
        }
    }
    first_string_range_[old_first_string].second = num_dets_;

    for (const auto& k_v : first_string_range_) {
        size_t range = k_v.second.second - k_v.second.first;
        min_per_string = std::min(min_per_string, range);
        max_per_string = std::max(max_per_string, range);
    }

    //   outfile->Printf("\n\n  SortedStringList Summary:");
    //   outfile->Printf("\n    Number of determinants: %zu", num_dets_);
    //   outfile->Printf("\n    Number of strings:      %zu (%.2f %%)",
sorted_half_dets_.size(),
    //                   100.0 * double(sorted_half_dets_.size()) / double(num_dets_));
    //   outfile->Printf("\n    Max block size:         %zu", max_per_string);
    //   outfile->Printf("\n    Min block size:         %zu", min_per_string);
    //   outfile->Printf("\n    Avg block size:         %0.f\n",
    //                   double(num_dets_) / double(sorted_half_dets_.size()));
}

SortedStringList::~SortedStringList() {}

const std::vector<Determinant>& SortedStringList::sorted_dets() const { return sorted_dets_; }

const std::vector<String>& SortedStringList::sorted_half_dets() const { return
sorted_half_dets_; }

const std::pair<size_t, size_t>& SortedStringList::range(const String& d) const {
    return first_string_range_.at(d);
}
*/
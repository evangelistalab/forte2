#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "determinant/bitarray.hpp"
#include "determinant/bitwise_operations.hpp"

namespace forte2 {

/**
 * @brief A class to represent a Slater determinant with N spin-orbital slots.
 *
 * The determinant stores N/2 alpha occupations and N/2 beta occupations. The alpha
 * occupations are stored first, followed by the beta occupations, so orbital index p refers to
 * the same spatial orbital in both spin sectors.
 */
template <size_t N> class DeterminantImpl : public BitArray<N> {
  protected:
    using BitArray<N>::getword;
    using BitArray<N>::maskbit;
    using BitArray<N>::whichbit;
    using BitArray<N>::whichword;
    using BitArray<N>::words_;

  public:
    // Since the template parent (BitArray) of this template class is not instantiated during the
    // compilation pass, here we declare all the member variables and functions inherited and used
    using BitArray<N>::nbits;
    using BitArray<N>::bits_per_word;
    using BitArray<N>::nwords_;
    using BitArray<N>::count;
    using BitArray<N>::get_bit;
    using BitArray<N>::set_bit;
    using BitArray<N>::slater_sign;
    using BitArray<N>::operator|;
    using BitArray<N>::operator^;
    using BitArray<N>::operator&;
    using BitArray<N>::operator<;
    using BitArray<N>::symmetric_difference_count;
    using BitArray<N>::is_disjoint_from;
    using BitArray<N>::find_first_one;
    using BitArray<N>::find_last_one;
    using BitArray<N>::clear;
    using BitArray<N>::for_each_set_bit;
    using BitArray<N>::find_set_bits;
    using Hash = typename BitArray<N>::Hash;

    /// Maximum number of spatial orbitals represented in each spin sector.
    static constexpr size_t norb_capacity = N / 2;

    static_assert(N % 128 == 0,
                  "Determinant storage must contain whole alpha/beta 64-orbital blocks");

    /// Number of packed storage words used by each spin sector.
    static constexpr size_t storage_words_per_spin = BitArray<N>::nwords_ / 2;

    /// Offset of the beta occupations in the packed storage inherited from BitArray.
    static constexpr size_t beta_storage_offset =
        storage_words_per_spin * BitArray<N>::bits_per_word;

    /// Return the maximum number of spatial orbitals represented by this determinant.
    static constexpr size_t norb() { return norb_capacity; }

    /// Default constructor
    DeterminantImpl() : BitArray<N>() {}

    /// Constructor from packed occupation storage.
    DeterminantImpl(const BitArray<N>& ba) : BitArray<N>(ba) {}

    static DeterminantImpl zero() {
        DeterminantImpl d;
        d.clear();
        return d;
    }

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DeterminantImpl(const std::vector<bool>& occupation_a,
                             const std::vector<bool>& occupation_b) {
        clear();
        int a_size = occupation_a.size();
        for (int p = 0; p < a_size; ++p)
            set_na(p, occupation_a[p]);
        int b_size = occupation_b.size();
        for (int p = 0; p < b_size; ++p)
            set_nb(p, occupation_b[p]);
    }

    /// Construct the determinant from two initializer lists of occupied alpha and beta orbitals.
    explicit DeterminantImpl(std::initializer_list<size_t> alpha_list,
                             std::initializer_list<size_t> beta_list) {
        clear();
        for (auto i : alpha_list) {
            set_na(i, true);
        }
        for (auto i : beta_list) {
            set_nb(i, true);
        }
    }

    explicit DeterminantImpl(std::vector<size_t> alpha_list, std::vector<size_t> beta_list) {
        clear();
        for (auto i : alpha_list) {
            set_na(i, true);
        }
        for (auto i : beta_list) {
            set_nb(i, true);
        }
    }

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DeterminantImpl(const BitArray<norb_capacity>& Ia, const BitArray<norb_capacity>& Ib) {
        this->set_strings(Ia, Ib);
    }

    /// Construct from the compact occupation-string representation used by str().
    DeterminantImpl(const std::string& str) { set_str(*this, str); }

    // Occupation accessors

    /// Return true if alpha orbital pos is occupied.
    bool na(size_t pos) const {
        if constexpr (nbits == 128) {
            return words_[0] & maskbit(pos);
        } else {
            return get_bit(pos);
        }
    }

    /// Return true if beta orbital pos is occupied.
    bool nb(size_t pos) const {
        if constexpr (nbits == 128) {
            return words_[1] & maskbit(pos);
        } else {
            return get_bit(pos + beta_storage_offset);
        }
    }

    // Occupation mutators

    /// Set the occupation of alpha orbital pos.
    void set_na(size_t pos, bool val) { set_bit(pos, val); }

    /// Set the occupation of beta orbital pos.
    void set_nb(size_t pos, bool val) { set_bit(pos + beta_storage_offset, val); }

    /// Set the alpha and beta occupation strings.
    void set_strings(const BitArray<norb_capacity>& sa, const BitArray<norb_capacity>& sb) {
        for (size_t n = 0; n < storage_words_per_spin; n++) {
            words_[n] = sa.get_word(n);
            words_[n + storage_words_per_spin] = sb.get_word(n);
        }
    }

    void set_a_string(const BitArray<norb_capacity>& sa) {
        for (size_t n = 0; n < storage_words_per_spin; n++) {
            words_[n] = sa.get_word(n);
        }
    }

    void set_b_string(const BitArray<norb_capacity>& sb) {
        for (size_t n = 0; n < storage_words_per_spin; n++) {
            words_[n + storage_words_per_spin] = sb.get_word(n);
        }
    }

    void set(std::initializer_list<size_t> alpha_list, std::initializer_list<size_t> beta_list) {
        clear();
        for (auto i : alpha_list) {
            set_na(i, true);
        }
        for (auto i : beta_list) {
            set_nb(i, true);
        }
    }

    // Comparison operators
    static bool less_than(const DeterminantImpl<N>& rhs, const DeterminantImpl<N>& lhs) {
        return rhs < lhs;
    }

    static bool reverse_less_than(const DeterminantImpl<N>& rhs, const DeterminantImpl<N>& lhs) {
        if constexpr (nbits == 128) {
            return (rhs.words_[0] < lhs.words_[0]) or
                   ((rhs.words_[0] == lhs.words_[0]) and (rhs.words_[1] < lhs.words_[1]));
        } else {
            for (size_t n = storage_words_per_spin; n > 0;) {
                --n;
                if (rhs.words_[n] > lhs.words_[n])
                    return false;
                if (rhs.words_[n] < lhs.words_[n])
                    return true;
            }
            for (size_t n = nwords_; n > storage_words_per_spin + 1;) {
                --n;
                if (rhs.words_[n] > lhs.words_[n])
                    return false;
                if (rhs.words_[n] < lhs.words_[n])
                    return true;
            }
            return rhs.words_[storage_words_per_spin] < lhs.words_[storage_words_per_spin];
        }
    }

    /// Apply a callable to each occupied alpha orbital, in ascending orbital order.
    ///
    /// The callback may return either void or a bool-like value. Void callbacks always continue.
    /// Bool callbacks continue when they return true and stop early when they return false.
    /// @param func a callable that accepts the alpha orbital index as a size_t
    /// @return true if all occupied alpha orbitals were visited, false if the callback stopped
    template <typename Func> bool for_each_a_occ(Func&& func) const {
        return for_each_set_bit(func, 0, storage_words_per_spin);
    }

    /// Apply a callable to each occupied beta orbital, in ascending orbital order.
    ///
    /// The callback may return either void or a bool-like value. Void callbacks always continue.
    /// Bool callbacks continue when they return true and stop early when they return false.
    /// The callback receives beta orbital indices in the spatial-orbital range [0, norb()).
    /// @param func a callable that accepts the beta orbital index as a size_t
    /// @return true if all occupied beta orbitals were visited, false if the callback stopped
    template <typename Func> bool for_each_b_occ(Func&& func) const {
        return for_each_set_bit(
            [&](size_t pos) {
                const size_t orb = pos - beta_storage_offset;
                if constexpr (std::is_void_v<std::invoke_result_t<Func&, size_t>>) {
                    std::invoke(func, orb);
                } else {
                    return std::invoke(func, orb);
                }
            },
            storage_words_per_spin, nwords_);
    }

    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ(int norb) const {
        std::vector<int> occ;
        if (static_cast<size_t>(norb) > norb_capacity) {
            throw std::range_error(
                "Determinant::get_alfa_occ(int norb) was passed a value of norb (" +
                std::to_string(norb) +
                "), which is larger than the maximum number of alpha orbitals (" +
                std::to_string(norb_capacity) + ").");
        }
        const size_t limit = static_cast<size_t>(norb);
        for_each_a_occ([&](size_t p) {
            if (p >= limit) {
                return false;
            }
            occ.push_back(static_cast<int>(p));
            return true;
        });
        return occ;
    }

    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ(int norb) const {
        std::vector<int> occ;
        if (static_cast<size_t>(norb) > norb_capacity) {
            throw std::range_error(
                "Determinant::get_beta_occ(int norb) was passed a value of norb (" +
                std::to_string(norb) +
                "), which is larger than the maximum number of beta orbitals (" +
                std::to_string(norb_capacity) + ").");
        }
        const size_t limit = static_cast<size_t>(norb);
        for_each_b_occ([&](size_t p) {
            if (p >= limit) {
                return false;
            }
            occ.push_back(static_cast<int>(p));
            return true;
        });
        return occ;
    }

    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir(int norb) const {
        std::vector<int> vir;
        if (static_cast<size_t>(norb) > norb_capacity) {
            throw std::range_error(
                "Determinant::get_alfa_occ(int norb) was passed a value of norb (" +
                std::to_string(norb) +
                "), which is larger than the maximum number of alpha orbitals (" +
                std::to_string(norb_capacity) + ").");
        }
        for (int p = 0; p < norb; ++p) {
            if (not na(p)) {
                vir.push_back(p);
            }
        }
        return vir;
    }

    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir(int norb) const {
        std::vector<int> vir;
        if (static_cast<size_t>(norb) > norb_capacity) {
            throw std::range_error(
                "Determinant::get_beta_occ(int norb) was passed a value of norb (" +
                std::to_string(norb) +
                "), which is larger than the maximum number of beta orbitals (" +
                std::to_string(norb_capacity) + ").");
        }
        for (int p = 0; p < norb; ++p) {
            if (not nb(p)) {
                vir.push_back(p);
            }
        }
        return vir;
    }

    /// @brief Find the occupied alpha orbitals and store them in occ
    /// @param occ a vector to store the occupied orbitals
    /// @param n the number of occupied orbitals found
    void collect_alpha_occupied(std::vector<size_t>& occ, size_t& n) const {
        n = 0;
        for_each_a_occ([&](size_t p) { occ[n++] = p; });
    }

    /// @brief Find the occupied beta orbitals and store them in occ
    /// @param occ a vector to store the occupied orbitals
    /// @param n the number of occupied orbitals found
    void collect_beta_occupied(std::vector<size_t>& occ, size_t& n) const {
        n = 0;
        for_each_b_occ([&](size_t p) { occ[n++] = p; });
    }

    /// Apply the alpha creation operator a^+_n to this determinant
    /// If orbital n is unoccupied, create the electron and return the sign
    /// If orbital n is occupied, do not modify the determinant and return 0
    double create_a(int n) {
        if (na(n))
            return 0.0;
        set_na(n, true);
        return slater_sign_a(n);
    }

    /// Apply the beta creation operator a^+_n to this determinant
    /// If orbital n is unoccupied, create the electron and return the sign
    /// If orbital n is occupied, do not modify the determinant and return 0
    double create_b(int n) {
        if (nb(n))
            return 0.0;
        set_nb(n, true);
        return slater_sign_b(n);
    }

    /// Apply the alpha annihilation operator a^+_n to this determinant
    /// If orbital n is occupied, annihilate the electron and return the sign
    /// If orbital n is unoccupied, do not modify the determinant and return 0
    double destroy_a(int n) {
        if (not na(n))
            return 0.0;
        set_na(n, false);
        return slater_sign_a(n);
    }

    /// Apply the beta annihilation operator a^+_n to this determinant
    /// If orbital n is occupied, annihilate the electron and return the sign
    /// If orbital n is unoccupied, do not modify the determinant and return 0
    double destroy_b(int n) {
        if (not nb(n))
            return 0.0;
        set_nb(n, false);
        return slater_sign_b(n);
    }

    /// Return the sign for a single second quantized operator
    /// This function ignores whether alpha orbital n is occupied.
    double slater_sign_a(size_t n) const {
        if constexpr (nbits == 128) {
            // specialization for one 64-orbital word per spin sector
            return ui64_sign(words_[0], n);
        } else {
            return slater_sign(n);
        }
    }

    /// Return the sign for a single second quantized operator
    /// This function ignores whether beta orbital n is occupied.
    double slater_sign_b(size_t n) const {
        if constexpr (nbits == 128) {
            // specialization for one 64-orbital word per spin sector
            return ui64_sign(words_[1], n) * ui64_bit_parity(words_[0]);
        } else {
            return slater_sign(n + beta_storage_offset);
        }
    }

    /// Return the sign for a pair of alpha second quantized operators
    /// The sign depends only on the parity of occupied alpha orbitals between n and m.
    /// n and m are not assumed to have any specific order
    double slater_sign_aa(size_t n, size_t m) const {
        if constexpr (nbits == 128) {
            // specialization for one 64-orbital word per spin sector
            return ui64_sign(words_[0], n, m);
        } else {
            return slater_sign(n, m);
        }
    }

    /// Return the sign for a pair of beta second quantized operators
    /// The sign depends only on the parity of occupied beta orbitals between n and m.
    /// n and m are not assumed to have any specific order
    double slater_sign_bb(size_t n, size_t m) const {
        if constexpr (nbits == 128) {
            // specialization for one 64-orbital word per spin sector
            return ui64_sign(words_[1], n, m);
        } else {
            return slater_sign(n + beta_storage_offset, m + beta_storage_offset);
        }
    }

    /// Return the sign of a^+_a a^+_b a_j a_i applied to this determinant for a same-spin alpha
    /// double excitation. The four orbital indices are assumed to be distinct. The case analysis
    /// handles all relative orderings of i, j, a, and b and produces the fermionic sign.
    double slater_sign_aaaa(size_t i, size_t j, size_t a, size_t b) const {
        if ((((i < a) && (j < a) && (i < b) && (j < b)) == true) ||
            (((i < a) || (j < a) || (i < b) || (j < b)) == false)) {
            if ((i < j) ^ (a < b)) {
                return (-1.0 * slater_sign_aa(i, j) * slater_sign_aa(a, b));
            } else {
                return (slater_sign_aa(i, j) * slater_sign_aa(a, b));
            }
        } else {
            if ((i < j) ^ (a < b)) {
                return (-1.0 * slater_sign_aa(i, b) * slater_sign_aa(j, a));
            } else {
                return (slater_sign_aa(i, a) * slater_sign_aa(j, b));
            }
        }
    }

    /// Return the sign of a^+_a a^+_b a_j a_i applied to this determinant for a same-spin beta
    /// double excitation. The four orbital indices are assumed to be distinct. The case analysis
    /// handles all relative orderings of i, j, a, and b and produces the fermionic sign.
    double slater_sign_bbbb(size_t i, size_t j, size_t a, size_t b) const {
        if ((((i < a) && (j < a) && (i < b) && (j < b)) == true) ||
            (((i < a) || (j < a) || (i < b) || (j < b)) == false)) {
            if ((i < j) ^ (a < b)) {
                return (-1.0 * slater_sign_bb(i, j) * slater_sign_bb(a, b));
            } else {
                return (slater_sign_bb(i, j) * slater_sign_bb(a, b));
            }
        } else {
            if ((i < j) ^ (a < b)) {
                return (-1.0 * slater_sign_bb(i, b) * slater_sign_bb(j, a));
            } else {
                return (slater_sign_bb(i, a) * slater_sign_bb(j, b));
            }
        }
    }

    /// Count the number of occupied alpha orbitals.
    int count_a() const {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return std::popcount(words_[0]);
        } else if (N == 256) {
            return std::popcount(words_[0]) + std::popcount(words_[1]);
        } else {
            return count(0, storage_words_per_spin);
        }
    }

    /// Count the number of occupied beta orbitals.
    int count_b() const {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return std::popcount(words_[1]);
        } else if (N == 256) {
            return std::popcount(words_[2]) + std::popcount(words_[3]);
        } else {
            return count(storage_words_per_spin, nwords_);
        }
    };

    /// Find the highest occupied alpha orbital.
    uint64_t find_last_alpha_occ() const { return find_last_one(0, storage_words_per_spin); }

    /// Find the highest occupied beta orbital.
    uint64_t find_last_beta_occ() const {
        if (auto res = find_last_one(storage_words_per_spin, nwords_); res != ui64_bit_not_found)
            return res - norb();
        return ui64_bit_not_found;
    }

    /// Return the number of alpha/beta pairs
    int npair() const {
        int count = 0;
        for (size_t k = 0; k < storage_words_per_spin; ++k) {
            count += std::popcount(words_[k] & words_[k + storage_words_per_spin]);
        }
        return count;
    }

    /// Perform an alpha-alpha single excitation (i->a)
    /// assuming that i is occupied and a is empty
    double single_excitation_a(int i, int a) {
        set_na(i, false);
        set_na(a, true);
        return slater_sign_aa(i, a);
    }

    /// Perform an beta-beta single excitation (I -> A)
    /// assuming that i is occupied and a is empty
    double single_excitation_b(int i, int a) {
        set_nb(i, false);
        set_nb(a, true);
        return slater_sign_bb(i, a);
    }

    /// Perform an alpha-alpha double excitation (ij->ab)
    /// assuming that ij are occupied and ab are empty
    double double_excitation_aa(int i, int j, int a, int b) {
        set_na(i, false);
        set_na(j, false);
        set_na(b, true);
        set_na(a, true);
        return slater_sign_aaaa(i, j, a, b);
    }

    /// Perform an alpha-beta double excitation (iJ -> aB)
    /// /// assuming that ij are occupied and ab are empty
    double double_excitation_ab(int i, int j, int a, int b) {
        set_na(i, false);
        set_nb(j, false);
        set_nb(b, true);
        set_na(a, true);
        return slater_sign_aa(i, a) * slater_sign_bb(j, b);
    }

    /// Perform an beta-beta double excitation (IJ -> AB)
    double double_excitation_bb(int i, int j, int a, int b) {
        set_nb(i, false);
        set_nb(j, false);
        set_nb(b, true);
        set_nb(a, true);
        return slater_sign_bbbb(i, j, a, b);
    }

    BitArray<norb_capacity> a_string() const {
        BitArray<norb_capacity> s;
        for (size_t i = 0; i < storage_words_per_spin; i++) {
            s.set_word(i, words_[i]);
        }
        return s;
    }

    BitArray<norb_capacity> b_string() const {
        BitArray<norb_capacity> s;
        for (size_t i = 0; i < storage_words_per_spin; i++) {
            s.set_word(i, words_[storage_words_per_spin + i]);
        }
        return s;
    }

    /// Swap the alpha and beta occupations of a determinant
    DeterminantImpl<N> spin_flip() const noexcept {
        DeterminantImpl<N> d;
        for (size_t n = 0; n < storage_words_per_spin; n++) {
            d.words_[n] = words_[n + storage_words_per_spin];
            d.words_[n + storage_words_per_spin] = words_[n];
        }
        return d;
    }

    /// Describe the excitation connection of a determinant d,
    /// relative to this one. The excitation connection is defined
    /// as the creation and annihilation operators that need to be applied
    /// to this determinant to obtain d.
    /// The excitation connection is a vector of 4 vectors:
    /// [[alpha annihilation], [alpha creation],
    ///  [beta annihilation], [beta creation]]
    std::vector<std::vector<size_t>> excitation_connection(const DeterminantImpl<N>& d) const {
        std::vector<std::vector<size_t>> excitation(4);
        for (size_t i = 0; i < norb_capacity; i++) {
            if (na(i) and not d.na(i)) {
                excitation[0].push_back(i);
            }
            if (not na(i) and d.na(i)) {
                excitation[1].push_back(i);
            }
            if (nb(i) and not d.nb(i)) {
                excitation[2].push_back(i);
            }
            if (not nb(i) and d.nb(i)) {
                excitation[3].push_back(i);
            }
        }
        return excitation;
    }
};

// Functions

template <size_t N>
std::string str(const DeterminantImpl<N>& d, int n = DeterminantImpl<N>::norb_capacity) {
    std::string s;
    s += "|";
    for (int p = 0; p < n; ++p) {
        if (d.na(p) and d.nb(p)) {
            s += "2";
        } else if (d.na(p) and not d.nb(p)) {
            s += "a";
        } else if (not d.na(p) and d.nb(p)) {
            s += "b";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

template <size_t N> std::ostream& operator<<(std::ostream& os, const DeterminantImpl<N>& d) {
    os << str(d);
    return os;
}

template <size_t N> void set_str(DeterminantImpl<N>& d, const std::string& str) {
    // Clear all occupations and set the packed occupation pattern passed as a string.
    d.clear();
    if (str.size() <= DeterminantImpl<N>::nbits) {
        size_t k = 0;
        for (auto c : str) {
            if (c == '0') {
                d.set_bit(k, 0);
            } else {
                d.set_bit(k, 1);
            }
            k++;
        }
    } else {
        throw std::range_error("template <size_t N> void set_str(DeterminantImpl<N>&, const "
                               "std::string&)\nmismatch "
                               "between the determinant storage size and the input string\n");
    }
}

template <size_t N>
std::vector<std::vector<int>> get_asym_occ(const DeterminantImpl<N>& d,
                                           const std::vector<int>& act_mo) {

    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (size_t h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (d.na(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

template <size_t N>
std::vector<std::vector<int>> get_bsym_occ(const DeterminantImpl<N>& d,
                                           const std::vector<int>& act_mo) {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (size_t h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (d.nb(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

template <size_t N>
std::vector<std::vector<int>> get_asym_vir(const DeterminantImpl<N>& d,
                                           const std::vector<int>& act_mo) {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (size_t h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (not d.na(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

template <size_t N>
std::vector<std::vector<int>> get_bsym_vir(const DeterminantImpl<N>& d,
                                           const std::vector<int>& act_mo) {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (size_t h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (not d.nb(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

/**
 * @brief Apply a general excitation operator to this determinant
 *        Details:
 *        (bc)_n ... (bc)_1 (ba)_n ... (ba)_1 (ac)_n ... (ac)_1 (aa)_n ... (aa)_1 |det>
 *        where aa = alpha annihilation operator
 *        where ac = alpha creation operator
 *        where ba = alpha annihilation operator
 *        where bc = alpha creation operator
 * @param aann list of alpha orbitals to annihilate
 * @param acre list of alpha orbitals to create
 * @param bann list of beta orbitals to annihilate
 * @param bcre list of beta orbitals to create
 * @return the sign of the final determinant (+1, -1, or 0)
 */
template <size_t N>
double gen_excitation(DeterminantImpl<N>& d, const std::vector<int>& aann,
                      const std::vector<int>& acre, const std::vector<int>& bann,
                      const std::vector<int>& bcre) {
    double sign = 1.0;
    for (auto i : aann) {
        sign *= d.slater_sign_a(i) * d.na(i);
        d.set_na(i, false);
    }
    for (auto i : acre) {
        sign *= d.slater_sign_a(i) * (1 - d.na(i));
        d.set_na(i, true);
    }
    for (auto i : bann) {
        sign *= d.slater_sign_b(i) * d.nb(i);
        d.set_nb(i, false);
    }
    for (auto i : bcre) {
        sign *= d.slater_sign_b(i) * (1 - d.nb(i));
        d.set_nb(i, true);
    }
    return sign;
}

/// @brief Apply a general operator to this determinant without checking applicability.
///
/// This function assumes the caller has already verified that the operator can be applied, for
/// example with can_apply_operator(cre, ann). Calling it on an inapplicable determinant may
/// produce a determinant and sign for a different algebraic operation.
///
/// @param d the determinant
/// @param new_d the new determinant
/// @param cre the creation operator
/// @param ann the annihilation operator
/// @param sign the sign mask (precomputed by the user) of the operator
/// @return the sign of the final determinant (+1, -1)
///
/// Example:
///
///   Determinant det, new_det, cre, ann, sign_mask;
///   // test if the operator can be applied
///   if (det.can_apply_operator(cre,ann)) {
///       // compute the sign mask
///       compute_sign_mask(cre, ann, sign_mask);
///       auto value = apply_operator_to_det_unchecked(det, new_det, cre, ann, sign_mask);
///       // do something with value and new_det
///   }
///
template <size_t N>
inline double
apply_operator_to_det_unchecked(const DeterminantImpl<N>& d, DeterminantImpl<N>& new_d,
                                const DeterminantImpl<N>& cre, const DeterminantImpl<N>& ann,
                                const DeterminantImpl<N>& sign) {
    size_t n = 0;
    if constexpr (N == 128) {
        // specialization for one 64-orbital word per spin sector
        const auto w0 = d.get_word(0) & (~ann.get_word(0));
        const auto w1 = d.get_word(1) & (~ann.get_word(1));
        n += std::popcount(w0 & sign.get_word(0));
        n += std::popcount(w1 & sign.get_word(1));
        new_d.set_word(0, w0 | cre.get_word(0));
        new_d.set_word(1, w1 | cre.get_word(1));
    } else if constexpr (N == 256) {
        const auto w0 = d.get_word(0) & (~ann.get_word(0));
        const auto w1 = d.get_word(1) & (~ann.get_word(1));
        const auto w2 = d.get_word(2) & (~ann.get_word(2));
        const auto w3 = d.get_word(3) & (~ann.get_word(3));
        n += std::popcount(w0 & sign.get_word(0));
        n += std::popcount(w1 & sign.get_word(1));
        n += std::popcount(w2 & sign.get_word(2));
        n += std::popcount(w3 & sign.get_word(3));
        new_d.set_word(0, w0 | cre.get_word(0));
        new_d.set_word(1, w1 | cre.get_word(1));
        new_d.set_word(2, w2 | cre.get_word(2));
        new_d.set_word(3, w3 | cre.get_word(3));
    } else {
        // loop over packed storage words
        for (size_t i = 0; i < DeterminantImpl<N>::nwords_; ++i) {
            // apply the annihilation operator
            const auto w = d.get_word(i) & (~ann.get_word(i));
            // compute the sign
            n += std::popcount(w & sign.get_word(i));
            // apply the creation operator
            new_d.set_word(i, w | cre.get_word(i));
        }
    }
    return parity_to_sign(n);
}

template <size_t N> double spin2(const DeterminantImpl<N>& lhs, const DeterminantImpl<N>& rhs) {
    int nmo = DeterminantImpl<N>::norb_capacity;
    const DeterminantImpl<N>& I = lhs;
    const DeterminantImpl<N>& J = rhs;

    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    // Make sure that Ms is the same otherwise the matrix element is automatically zero
    if ((lhs.count_a() != rhs.count_a()) or (lhs.count_b() != rhs.count_b())) {
        return 0.0;
    }

    DeterminantImpl<N> lr_diff = lhs ^ rhs;

    int nadiff = lr_diff.count_a() / 2;
    int nbdiff = lr_diff.count_b() / 2;
    int na = lhs.count_a();
    int nb = lhs.count_b();
    int npair = lhs.npair();

    double Ms = 0.5 * static_cast<double>(na - nb);

    // PhiI = PhiJ -> S^2 = Sz (Sz + 1) + Nbeta - Npairs
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element += Ms * (Ms + 1.0) + double(nb) - double(npair);
    }

    // PhiI = a+(qa) a+(pb) a-(qb) a-(pa) PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Find a pair of spin coupled electrons
        int i = -1;
        int j = -1;
        // The logic here follows the spin-flip coupling between opposite-spin occupations.
        for (int p = 0; p < nmo; ++p) {
            if (J.na(p) and I.nb(p) and (not J.nb(p)) and (not I.na(p)))
                i = p;
            if (J.nb(p) and I.na(p) and (not J.na(p)) and (not I.nb(p)))
                j = p;
        }
        if (i != j and i >= 0 and j >= 0) {
            double sign = rhs.slater_sign_a(i) * rhs.slater_sign_b(j) * lhs.slater_sign_a(j) *
                          lhs.slater_sign_b(i);
            matrix_element -= sign;
        }
    }
    return (matrix_element);
}

} // namespace forte2

namespace std {
// specialization of std::hash for forte2::OccupationVector
template <size_t N> struct hash<forte2::DeterminantImpl<N>> {
    std::size_t operator()(const forte2::DeterminantImpl<N>& d) const noexcept {
        using HashT = typename forte2::DeterminantImpl<N>::Hash;
        return HashT{}(d);
    }
};

} // namespace std

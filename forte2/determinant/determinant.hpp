#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "determinant/bitarray.hpp"
#include "determinant/bitwise_operations.hpp"
#include "determinant/string.hpp"

namespace forte2 {

/**
 * @brief A class to represent a Slater determinant with N spin-orbital slots where N is a multiple
 * of 128.
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
    using BitArray<N>::slater_sign;
    using Hash = typename BitArray<N>::Hash;

    static_assert(N % 128 == 0,
                  "Determinant storage must contain whole alpha/beta 64-orbital blocks");

    /// Maximum number of spatial orbitals represented in each spin sector.
    static constexpr size_t norb_capacity = N / 2;

    /// Number of packed storage words used by each spin sector.
    static constexpr size_t storage_words_per_spin = BitArray<N>::nwords_ / 2;

    /// Offset of the beta occupations in the packed storage inherited from BitArray.
    static constexpr size_t beta_storage_offset =
        storage_words_per_spin * BitArray<N>::bits_per_word;

    /// @return the maximum number of spatial orbitals represented by this determinant.
    static constexpr size_t norb() { return norb_capacity; }

    // ==> Constructors <==

    /// @brief Default constructor
    DeterminantImpl() : BitArray<N>() {}

    /// @brief Constructor from packed occupation storage.
    DeterminantImpl(const BitArray<N>& ba) : BitArray<N>(ba) {}

    /// @brief Return a determinant with all occupations set to zero (unoccupied).
    /// @return a DeterminantImpl with all occupations set to zero
    static DeterminantImpl zero() {
        DeterminantImpl d;
        d.clear();
        return d;
    }

    /// @brief Construct the determinant from two vectors of occupied alpha and beta orbitals.
    /// @param alpha_list a vector of occupied alpha orbital indices
    /// @param beta_list a vector of occupied beta orbital indices
    explicit DeterminantImpl(const std::vector<size_t>& alpha_list,
                             const std::vector<size_t>& beta_list) {
        clear();
        for (auto i : alpha_list)
            set_na(i, true);
        for (auto i : beta_list)
            set_nb(i, true);
    }

    /// @brief Construct the determinant from two bit strings representing the alpha and beta
    /// occupations.
    /// @param sa a BitArray representing the alpha occupation string
    /// @param sb a BitArray representing the beta occupation string
    explicit DeterminantImpl(const BitArray<norb_capacity>& sa, const BitArray<norb_capacity>& sb) {
        this->set_strings(sa, sb);
    }

    // ==> Occupation accessors <==

    /// @brief Return the occupation of alpha orbital n.
    /// @param n the orbital index
    /// @return true if the orbital is occupied, false otherwise
    bool na(size_t n) const {
        if constexpr (nbits == 128) {
            return words_[0] & maskbit(n);
        } else {
            return get_bit(n);
        }
    }

    /// @brief Return the occupation of beta orbital n.
    /// @param n the orbital index
    /// @return true if the orbital is occupied, false otherwise
    bool nb(size_t n) const {
        if constexpr (nbits == 128) {
            return words_[1] & maskbit(n);
        } else {
            return get_bit(n + beta_storage_offset);
        }
    }

    // ==> Occupation mutators <==

    /// @brief Set the occupation of alpha orbital n.
    /// @param n the orbital index
    /// @param val the occupation value to set (true for occupied, false for unoccupied)
    void set_na(size_t n, bool val) { set_bit(n, val); }

    /// @brief Set the occupation of beta orbital n.
    /// @param n the orbital index
    /// @param val the occupation value to set (true for occupied, false for unoccupied)
    void set_nb(size_t n, bool val) { set_bit(n + beta_storage_offset, val); }

    /// @brief Set the alpha and beta occupation strings.
    /// @param sa a BitArray representing the alpha occupation string
    /// @param sb a BitArray representing the beta occupation string
    void set_strings(const StringImpl<norb_capacity>& sa, const StringImpl<norb_capacity>& sb) {
        set_alpha_string(sa);
        set_beta_string(sb);
    }

    /// @brief Set the alpha occupation string.
    /// @param sa a StringImpl representing the alpha occupation string
    void set_alpha_string(const StringImpl<norb_capacity>& sa) {
        for (size_t n = 0; n < storage_words_per_spin; n++) {
            words_[n] = sa.get_word(n);
        }
    }

    /// @brief Set the beta occupation string.
    /// @param sb a StringImpl representing the beta occupation string
    void set_beta_string(const StringImpl<norb_capacity>& sb) {
        for (size_t n = 0; n < storage_words_per_spin; n++) {
            words_[n + storage_words_per_spin] = sb.get_word(n);
        }
    }

    // ==> Comparison <==

    /// @brief Return rhs < lhs. This is the default ordering used for sorting determinants, which
    /// is the same as the lexicographical ordering of the bit strings.
    static bool less_than(const DeterminantImpl<N>& lhs, const DeterminantImpl<N>& rhs) {
        return lhs < rhs;
    }

    /// @brief Return rhs < lhs in reverse order, which is the same as the lexicographical
    /// ordering of the bit strings when read from the last storage word to the first.
    static bool reverse_less_than(const DeterminantImpl<N>& lhs, const DeterminantImpl<N>& rhs) {
        if constexpr (nbits == 128) {
            return (lhs.words_[0] < rhs.words_[0]) or
                   ((lhs.words_[0] == rhs.words_[0]) and (lhs.words_[1] < rhs.words_[1]));
        } else {
            for (size_t n = storage_words_per_spin; n > 0;) {
                --n;
                if (lhs.words_[n] > rhs.words_[n])
                    return false;
                if (lhs.words_[n] < rhs.words_[n])
                    return true;
            }
            for (size_t n = nwords_; n > storage_words_per_spin + 1;) {
                --n;
                if (lhs.words_[n] > rhs.words_[n])
                    return false;
                if (lhs.words_[n] < rhs.words_[n])
                    return true;
            }
            return lhs.words_[storage_words_per_spin] < rhs.words_[storage_words_per_spin];
        }
    }

    // ==> Iteration over occupied orbitals <==

    /// @brief Apply a callable to each occupied alpha orbital, in ascending orbital order.
    /// The callable may return either void or a bool-like value. Void callables always continue.
    /// Bool callables continue when they return true and stop early when they return false.
    /// The callable receives alpha orbital indices in the spatial-orbital range [0, norb()).
    /// @param func a callable that accepts the alpha orbital index as a size_t
    /// @return true if all occupied alpha orbitals were visited, false if the callback stopped
    template <typename Func> bool for_each_a_occ(Func&& func) const {
        return for_each_set_bit(func, 0, storage_words_per_spin);
    }

    /// @brief Apply a callable to each occupied beta orbital, in ascending orbital order.
    /// The callable may return either void or a bool-like value. Void callables always continue.
    /// Bool callables continue when they return true and stop early when they return false.
    /// The callable receives beta orbital indices in the spatial-orbital range [0, norb()).
    /// @param func a callable that accepts the beta orbital index as a size_t
    /// @return true if all occupied beta orbitals were visited, false if the callable stopped
    template <typename Func> bool for_each_b_occ(Func&& func) const {
        return for_each_set_bit(
            [&](size_t n) {
                const size_t orb = n - beta_storage_offset;
                if constexpr (std::is_void_v<std::invoke_result_t<Func, size_t>>) {
                    std::invoke(std::forward<Func>(func), orb);
                } else {
                    return std::invoke(std::forward<Func>(func), orb);
                }
            },
            storage_words_per_spin, nwords_);
    }

    /// @brief Apply a callable to each occupied orbital (alpha and beta), in ascending orbital
    /// order. The callable may return either void or a bool-like value. Void callables always
    /// continue. Bool callables continue when they return true and stop early when they return
    /// false. The callable receives orbital indices in the spatial-orbital range [0, 2 * norb()).
    /// @param func a callable that accepts the orbital index as a size_t.
    /// @return true if all occupied orbitals were visited, false if the callback stopped
    template <typename Func> bool for_each_occ(Func&& func) const { return for_each_set_bit(func); }

    /// @brief Find all occupied alpha orbitals
    /// @return a vector of occupied alpha orbital indices
    std::vector<size_t> get_alpha_occ() const {
        auto n = count_alpha();
        std::vector<size_t> occ(n);
        collect_alpha_occupied(occ, n);
        return occ;
    }

    /// @brief Find all occupied beta orbitals
    /// @return a vector of occupied beta orbital indices
    std::vector<size_t> get_beta_occ() const {
        auto n = count_beta();
        std::vector<size_t> occ(n);
        collect_beta_occupied(occ, n);
        return occ;
    }

    /// @brief Find the occupied alpha orbitals and store them in occ
    /// This is a more efficient version of get_alpha_occ that avoids dynamic resizing of the output
    /// vector.
    /// @param occ a vector to store the occupied orbitals
    /// @param n the number of occupied orbitals found
    void collect_alpha_occupied(std::vector<size_t>& occ, size_t& n) const {
        n = 0;
        for_each_a_occ([&](size_t p) { occ[n++] = p; });
    }

    /// @brief Find the occupied beta orbitals and store them in occ
    /// This is a more efficient version of get_beta_occ that avoids dynamic resizing of the output
    /// vector.
    /// @param occ a vector to store the occupied orbitals
    /// @param n the number of occupied orbitals found
    void collect_beta_occupied(std::vector<size_t>& occ, size_t& n) const {
        n = 0;
        for_each_b_occ([&](size_t p) { occ[n++] = p; });
    }

    /// @brief Apply the alpha creation operator a^+_n to this determinant
    /// If orbital n is unoccupied, create the electron and return the sign
    /// If orbital n is occupied, do not modify the determinant and return 0
    /// @param n the orbital index
    /// @return the sign of the creation operator if the orbital was unoccupied, 0 otherwise
    double create_alpha(size_t n) {
        if (na(n))
            return 0.0;
        set_na(n, true);
        return slater_sign_a(n);
    }

    /// @brief Apply the beta creation operator a^+_n to this determinant
    /// If orbital n is unoccupied, create the electron and return the sign
    /// If orbital n is occupied, do not modify the determinant and return 0
    /// @param n the orbital index
    /// @return the sign of the creation operator if the orbital was unoccupied, 0 otherwise
    double create_beta(size_t n) {
        if (nb(n))
            return 0.0;
        set_nb(n, true);
        return slater_sign_b(n);
    }

    /// @brief Apply the alpha annihilation operator a_n to this determinant
    /// If orbital n is occupied, annihilate the electron and return the sign
    /// If orbital n is unoccupied, do not modify the determinant and return 0
    /// @param n the orbital index
    /// @return the sign of the annihilation operator if the orbital was occupied, 0 otherwise
    double destroy_alpha(size_t n) {
        if (not na(n))
            return 0.0;
        set_na(n, false);
        return slater_sign_a(n);
    }

    /// @brief Apply the beta annihilation operator a_n to this determinant
    /// If orbital n is occupied, annihilate the electron and return the sign
    /// If orbital n is unoccupied, do not modify the determinant and return 0
    /// @param n the orbital index
    /// @return the sign of the annihilation operator if the orbital was occupied, 0 otherwise
    double destroy_beta(size_t n) {
        if (not nb(n))
            return 0.0;
        set_nb(n, false);
        return slater_sign_b(n);
    }

    /// @brief Return the fermionic sign for an alpha orbital
    /// This function ignores whether alpha orbital n is occupied.
    /// @param n the orbital index
    /// @return the fermionic sign
    double slater_sign_a(size_t n) const {
        if constexpr (nbits == 128) {
            // specialization for one 64-orbital word per spin sector
            return ui64_sign(words_[0], n);
        } else {
            return slater_sign(n);
        }
    }

    /// @brief Return the fermionic sign for a beta orbital
    /// This function ignores whether beta orbital n is occupied.
    /// @param n the orbital index
    /// @return the fermionic sign
    double slater_sign_b(size_t n) const {
        if constexpr (nbits == 128) {
            // specialization for one 64-orbital word per spin sector
            return ui64_sign(words_[1], n) * ui64_bit_parity(words_[0]);
        } else {
            return slater_sign(n + beta_storage_offset);
        }
    }

    /// @brief Return the fermionic sign for a pair of alpha second quantized operators
    /// The sign depends only on the parity of occupied alpha orbitals between n and m.
    /// n and m are not assumed to have any specific order
    /// @param n the first orbital index
    /// @param m the second orbital index
    /// @return the fermionic sign
    double slater_sign_aa(size_t n, size_t m) const {
        if constexpr (nbits == 128) {
            // specialization for one 64-orbital word per spin sector
            return ui64_sign(words_[0], n, m);
        } else {
            return slater_sign(n, m);
        }
    }

    /// @brief Return the fermionic sign for a pair of beta second quantized operators
    /// The sign depends only on the parity of occupied beta orbitals between n and m.
    /// n and m are not assumed to have any specific order
    /// @param n the first orbital index
    /// @param m the second orbital index
    /// @return the fermionic sign
    double slater_sign_bb(size_t n, size_t m) const {
        if constexpr (nbits == 128) {
            // specialization for one 64-orbital word per spin sector
            return ui64_sign(words_[1], n, m);
        } else {
            return slater_sign(n + beta_storage_offset, m + beta_storage_offset);
        }
    }

    /// @brief Return the fermionic sign of a^+_a a^+_b a_j a_i applied to this determinant for a
    /// same-spin alpha double excitation. The four orbital indices are assumed to be distinct. The
    /// case analysis handles all relative orderings of i, j, a, and b and produces the fermionic
    /// sign.
    /// @param i the first occupied orbital index
    /// @param j the second occupied orbital index
    /// @param a the first unoccupied orbital index
    /// @param b the second unoccupied orbital index
    /// @return the fermionic sign of the double excitation operator
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

    /// @brief Return the fermionic sign of a^+_a a^+_b a_j a_i applied to this determinant for a
    /// same-spin beta double excitation. The four orbital indices are assumed to be distinct. The
    /// case analysis handles all relative orderings of i, j, a, and b and produces the fermionic
    /// sign.
    /// @param i the first occupied orbital index
    /// @param j the second occupied orbital index
    /// @param a the first unoccupied orbital index
    /// @param b the second unoccupied orbital index
    /// @return the fermionic sign of the double excitation operator
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

    /// @brief Count the number of occupied alpha orbitals.
    /// @return the number of occupied alpha orbitals
    size_t count_alpha() const noexcept {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return std::popcount(words_[0]);
        } else if (N == 256) {
            return std::popcount(words_[0]) + std::popcount(words_[1]);
        } else {
            return count(0, storage_words_per_spin);
        }
    }

    /// @brief Count the number of occupied beta orbitals.
    /// @return the number of occupied beta orbitals
    size_t count_beta() const noexcept {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return std::popcount(words_[1]);
        } else if (N == 256) {
            return std::popcount(words_[2]) + std::popcount(words_[3]);
        } else {
            return count(storage_words_per_spin, nwords_);
        }
    };

    /// @brief Return the number of orbitals in which both alpha and beta are occupied (number of
    /// alpha/beta pairs).
    /// @return the number of alpha/beta pairs
    int npair() const noexcept {
        int count = 0;
        for (size_t k = 0; k < storage_words_per_spin; ++k) {
            count += std::popcount(words_[k] & words_[k + storage_words_per_spin]);
        }
        return count;
    }

    /// @brief Return the alpha occupation string.
    /// @return a StringImpl representing the alpha occupation string
    StringImpl<norb_capacity> a_string() const {
        StringImpl<norb_capacity> s;
        for (size_t i = 0; i < storage_words_per_spin; i++) {
            s.set_word(i, words_[i]);
        }
        return s;
    }

    /// @brief Return the beta occupation string.
    /// @return a StringImpl representing the beta occupation string
    StringImpl<norb_capacity> b_string() const {
        StringImpl<norb_capacity> s;
        for (size_t i = 0; i < storage_words_per_spin; i++) {
            s.set_word(i, words_[storage_words_per_spin + i]);
        }
        return s;
    }

    /// @brief Swap the alpha and beta occupations of a determinant
    /// @return a new DeterminantImpl with swapped occupations
    DeterminantImpl<N> spin_flip() const noexcept {
        DeterminantImpl<N> d;
        for (size_t n = 0; n < storage_words_per_spin; n++) {
            d.words_[n] = words_[n + storage_words_per_spin];
            d.words_[n + storage_words_per_spin] = words_[n];
        }
        return d;
    }

    /// @brief Check if an orbital index is within bounds for this determinant and throw an
    /// exception if it is not.
    /// @param n the orbital index to check
    /// @param msg an additional message to include in the exception if the index is out of bounds
    void check_index_bounds(size_t n, const std::string& msg) const {
        if (n >= norb_capacity) {
            throw std::out_of_range("Orbital index " + std::to_string(n) +
                                    " is out of range for Determinant with capacity " +
                                    std::to_string(norb_capacity) + ". " + msg);
        }
    }
};

// Functions

/// @brief Convert a determinant to a string representation
/// @tparam N The number of spin-orbital slots in the determinant, which must be a multiple of 128
/// @param d The determinant to convert to a string
/// @param n The number of spatial orbitals to include in the string representation. Must be less
/// than or equal to the capacity of the determinant. Default is the full capacity of the
/// determinant.
/// @throws std::out_of_range if n is greater than the capacity of the determinant
/// @return a string representation of the determinant, where occupied alpha orbitals are
/// represented by 'a', occupied beta orbitals are represented by 'b', and unoccupied orbitals are
/// represented by '0'. The string is enclosed in angle brackets, e.g. "|a0b0>".
template <size_t N>
std::string str(const DeterminantImpl<N>& d, size_t n = DeterminantImpl<N>::norb_capacity) {
    std::string s;
    s += "|";
    for (size_t p = 0; p < n; ++p) {
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

/// @brief Output stream operator for DeterminantImpl. This allows you to print a DeterminantImpl
/// object using std::cout or any other output stream. The output format is the same as the string
/// representation produced by the str() function.
template <size_t N> std::ostream& operator<<(std::ostream& os, const DeterminantImpl<N>& d) {
    os << str(d);
    return os;
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

#pragma once

#include "determinant/bitarray.hpp"

namespace forte2 {

/**
 * @brief A class to represent an occupation number string with up to N orbitals where N is a
 * multiple of 64.
 */
template <size_t N> class StringImpl : public BitArray<N> {
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
    using Hash = typename BitArray<N>::Hash;

    /// @brief Constructor
    StringImpl() : BitArray<N>() {}

    /// @brief Constructor from packed occupation storage.
    StringImpl(const BitArray<N>& ba) : BitArray<N>(ba) {}

    /// @brief Return a StringImpl object with all bits set to zero. Users should use this method
    /// instead of the default constructor.
    /// @return a StringImpl object with all bits set to zero
    static StringImpl zero() {
        StringImpl s;
        s.clear();
        return s;
    }

    /// @brief Return the fermionic sign of creating a particle in orbital n.
    /// The sign is +1 if the number of occupied orbitals before n is even, and -1 if it is odd.
    /// For n out of bounds, the behavior is undefined.
    /// @param n the orbital index with respect to which the sign is calculated
    /// @return the fermionic sign
    double slater_sign(size_t n) const {
        if constexpr (N == 64) {
            return ui64_sign(words_[0], n);
        } else {
            size_t count = 0;
            // count all the preceeding bits only if we are looking past the first word
            if (n >= bits_per_word) {
                size_t last_full_word = whichword(n);
                for (size_t k = 0; k < last_full_word; ++k) {
                    count += std::popcount(words_[k]);
                }
            }
            const double word_sign = ui64_sign(getword(n), whichbit(n));
            return (count % 2 == 0) ? word_sign : -word_sign;
        }
    }

    /// @brief Return the fermionic sign of creating a particle in orbital n in reverse order.
    /// The sign is +1 if the number of occupied orbitals after n is even, and -1 if it is odd.
    /// For n out of bounds, the behavior is undefined
    /// @param n the orbital index with respect to which the sign is calculated
    /// @return the fermionic sign
    double slater_sign_reverse(size_t n) const {
        if constexpr (N == 64) {
            return ui64_sign_reverse(words_[0], n);
        } else {
            size_t count = 0;
            // count all the following bits only if we are not looking at the last word
            size_t start_word =
                whichword(n) + 1; // Start from the word following the one containing bit n
            if (start_word < nwords_) {
                for (size_t k = start_word; k < nwords_; ++k) {
                    count += std::popcount(words_[k]);
                }
            }
            const double word_sign = ui64_sign_reverse(getword(n), whichbit(n));
            return (count % 2 == 0) ? word_sign : -word_sign;
        }
    }

    /// Return the sign for a pair of second quantized operators
    /// The sign depends only on the number of bits = 1 between n and m
    /// There are no restrictions on n and m
    double slater_sign(size_t n, size_t m) const {
        if constexpr (N == 64) {
            return ui64_sign(words_[0], n, m);
        } else if constexpr (N == 128) {
            // XXXXXXXX YYYYYYYY
            // XmXXXXnX YYYYYYYY (case 1)
            //   cccc
            // XmXXXXnX YYYYYYYY (case 1)
            //   cccccc
            // cccccc
            // XmXXXXXX YYYYYnYY (case 2)
            //   cccccc ccccc
            // let's first order the numbers so that m <= n
            if (n < m)
                std::swap(m, n);
            size_t word_m = whichword(m);
            size_t word_n = whichword(n);
            // if both bits are in the same word use an optimized version
            if (word_n == word_m) {
                return ui64_sign(words_[word_n], whichbit(n), whichbit(m));
            }
            // count the bits after m in word[m]
            // count the bits before n in word[n]
            return ui64_sign_reverse(words_[word_m], whichbit(m)) *
                   ui64_sign(words_[word_n], whichbit(n));
        } else {
            // let's first order the numbers so that m <= n
            if (n < m)
                std::swap(m, n);
            size_t word_m = whichword(m);
            size_t word_n = whichword(n);
            // if both bits are in the same word use an optimized version
            if (word_n == word_m) {
                return ui64_sign(words_[word_n], whichbit(n), whichbit(m));
            }
            size_t count = 0;
            // count the number of bits in bitween the words of m and n
            for (size_t k = word_m + 1; k < word_n; ++k) {
                count += std::popcount(words_[k]);
            }
            // count the bits after m in word[m]
            // count the bits before n in word[n]
            double sign = ui64_sign_reverse(words_[word_m], whichbit(m)) *
                          ui64_sign(words_[word_n], whichbit(n));
            return (count % 2 == 0) ? sign : -sign;
        }
    }

    /// Return the sign of a_n applied to this determinant
    /// this version is inefficient and should be used only for testing/debugging
    double slater_sign_safe(int n) const {
        size_t count = 0;
        for (int k = 0; k < n; ++k) {
            if (get_bit(k))
                count++;
        }
        return (count % 2 == 0) ? 1.0 : -1.0;
    }

    /// @brief Create a particle in orbital n and return the sign of the resulting determinant. If
    /// orbital n is already occupied, this function returns 0 and does not modify the determinant.
    /// @param n the orbital index where the particle is created
    /// @return the sign of the resulting determinant, or 0 if the creation is not possible or if n
    /// is out of bounds
    double create(int n) {
        if (get_bit(n) or (n < 0) or (n >= static_cast<int>(N)))
            return 0.0;
        return create_unchecked(n);
    }

    /// @brief Destroy a particle in orbital n and return the sign of the resulting determinant. If
    /// orbital n is already empty, this function returns 0 and does not modify the determinant.
    /// @param n the orbital index where the particle is destroyed
    /// @return the sign of the resulting determinant, or 0 if the destruction is not possible or if
    /// n is out of bounds
    double destroy(int n) {
        if (not get_bit(n) or (n < 0) or (n >= static_cast<int>(N)))
            return 0.0;
        return destroy_unchecked(n);
    }

    /// @brief Create a particle in orbital n assuming the orbital is empty,
    /// and return the sign of the resulting determinant.
    /// @param n the orbital index where the particle is created
    /// @return the sign of the resulting determinant
    double create_unchecked(int n) {
        set_bit(n, true);
        return slater_sign(n);
    }

    /// @brief Destroy a particle in orbital n assuming the orbital is occupied,
    /// and return the sign of the resulting determinant.
    /// @param n the orbital index where the particle is destroyed
    /// @return the sign of the resulting determinant
    double destroy_unchecked(int n) {
        set_bit(n, false);
        return slater_sign(n);
    }

    /// @brief Find the irreducible representation of a product of spin orbitals
    /// @param irrep a vector of irrep values
    /// @return the irrep
    int symmetry(const std::vector<int>& irrep) const {
        int sym = 0;
        for_each_set_bit([&](size_t pos) { sym ^= irrep[pos]; });
        return sym;
    }
};

} // namespace forte2

namespace std {
// specialization of std::hash for forte2::OccupationVector
template <size_t N> struct hash<forte2::StringImpl<N>> {
    std::size_t operator()(const forte2::StringImpl<N>& d) const noexcept {
        using HashT = typename forte2::StringImpl<N>::Hash;
        return HashT{}(d);
    }
};

} // namespace std

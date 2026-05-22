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
    using BitArray<N>::symmetric_difference_count;
    using BitArray<N>::is_disjoint_from;
    using BitArray<N>::find_first_one;
    using BitArray<N>::find_last_one;
    using BitArray<N>::clear;
    using BitArray<N>::for_each_set_bit;
    using BitArray<N>::find_set_bits;
    using BitArray<N>::slater_sign;
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

    /// @brief Create a particle in orbital n and return the sign of the resulting determinant. If
    /// orbital n is already occupied, this function returns 0 and does not modify the determinant.
    /// @param n the orbital index where the particle is created
    /// @return the sign of the resulting determinant, or 0 if the creation is not possible or if n
    /// is out of bounds
    double create(int n) {
        if ((n < 0) or (n >= static_cast<int>(N)) or get_bit(n))
            return 0.0;
        return create_unchecked(n);
    }

    /// @brief Destroy a particle in orbital n and return the sign of the resulting determinant. If
    /// orbital n is already empty, this function returns 0 and does not modify the determinant.
    /// @param n the orbital index where the particle is destroyed
    /// @return the sign of the resulting determinant, or 0 if the destruction is not possible or if
    /// n is out of bounds
    double destroy(int n) {
        if ((n < 0) or (n >= static_cast<int>(N)) or (not get_bit(n)))
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
};

} // namespace forte2

namespace std {
/// @brief Specialization of std::hash for forte2::StringImpl
template <size_t N> struct hash<forte2::StringImpl<N>> {
    /// @brief Compute a hash value for a StringImpl object using the hash function defined in the
    /// StringImpl class.
    /// @param d the StringImpl object to hash
    /// @return a hash value for the StringImpl object
    std::size_t operator()(const forte2::StringImpl<N>& d) const noexcept {
        using HashT = typename forte2::StringImpl<N>::Hash;
        return HashT{}(d);
    }
};

} // namespace std

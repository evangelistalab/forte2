#pragma once

#include "determinant/bitarray.hpp"
#include "determinant/determinant.hpp"
#include "determinant/string.hpp"

namespace forte2 {

/// @brief A two-component determinant in which spinor p is bit p.
///
/// It has the same storage as DeterminantImpl<N>
template <size_t N> class SpinorDeterminantImpl final : public StringImpl<N> {
  public:
    using StringImpl<N>::StringImpl;

    // explicitly disallow constructing a SpinorDet from a regular Det
    SpinorDeterminantImpl(const DeterminantImpl<N>&) = delete;

    static SpinorDeterminantImpl zero() {
        SpinorDeterminantImpl d;
        d.clear();
        return d;
    }
};

/// @return d as a Determinant
template <size_t N> DeterminantImpl<N> to_determinant(const SpinorDeterminantImpl<N>& d) {
    return DeterminantImpl<N>(static_cast<const BitArray<N>&>(d));
}

/// @return a spinor determinant
template <size_t N> SpinorDeterminantImpl<N> from_determinant(const DeterminantImpl<N>& d) {
    return SpinorDeterminantImpl<N>(static_cast<const BitArray<N>&>(d));
}

} // namespace forte2

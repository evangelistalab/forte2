#pragma once

#include <bit>
#include <functional>
#include <string>

#include "helpers/hash_combine.h"

namespace forte2 {

struct OccupationVector {
    // the type used to represent a word (a 64 bit unsigned integer)
    using bits_t = uint64_t;

    // the number of bits in one word (8 * 8 = 64)
    static constexpr size_t N = 8 * sizeof(bits_t);

    OccupationVector() noexcept {}

    static OccupationVector zero() noexcept {
        OccupationVector ov;
        ov.clear();
        return ov;
    }

    // the bits
    bits_t bits_;

    void clear() noexcept { bits_ = 0; }

    // compare two occupation vectors
    bool operator==(const OccupationVector& other) const noexcept { return bits_ == other.bits_; }

    bits_t raw() const noexcept { return bits_; }

    // the number of bits in the vector set to 1
    int count() const noexcept { return std::popcount(bits_); }

    static constexpr size_t maskbit(size_t pos) noexcept { return (static_cast<bits_t>(1)) << pos; }

    // get the value of bit in position pos
    bool operator[](size_t pos) const { return bits_ & maskbit(pos); }

    /// set bit in position pos to the value val
    void set(size_t pos, bool val) {
        bits_ ^= (-val ^ bits_) & maskbit(pos); // if-free implementation
    }
};

// std::string str(const OccupationVector& ov, int n = OccupationVector::N) {
//     std::string s;
//     s += "|";
//     for (int p = 0; p < n; ++p) {
//         if (ov[p]) {
//             s += "1";
//         } else {
//             s += "0";
//         }
//     }
//     s += ">";
//     return s;
// }

} // namespace forte2

namespace std {
template <> struct hash<forte2::OccupationVector> {
    std::size_t operator()(const forte2::OccupationVector& ov) const noexcept {
        return std::hash<forte2::OccupationVector::bits_t>()(ov.raw());
    }
};
} // namespace std
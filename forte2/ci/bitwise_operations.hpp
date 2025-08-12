#pragma once

#include <bit>
#include <cstdint>
#include <limits>

/// @brief Accumulate hash values of 64 bit unsigned integers
/// based on boost/functional/hash/hash.hpp
inline void hash_combine_uint64(std::size_t& seed, std::uint64_t value) {
    const std::size_t hv = std::hash<std::uint64_t>{}(value);
    // 0x9e3779b97f4a7c15ULL is the 64-bit variant of the constant used in boost::hash_combine
    // (0x9e3779b9), derived from the fractional part of the golden ratio. This helps to spread hash
    // values uniformly.
    seed ^= hv + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

/// @brief Hash a range of uint64_t values stored in an iterable container
template <typename Container>
inline constexpr std::size_t hash_range_uint64(const Container& container) noexcept {
    static_assert(
        std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(*std::begin(container))>>,
                       std::uint64_t>,
        "hash_range_uint64 can only be used with containers that provide iterators over uint64_t");
    std::size_t seed = 0;
    for (auto el : container) {
        hash_combine_uint64(seed, el);
    }
    return seed;
}

/// @brief Count the number of bit set to 1 in a uint64_t
/// @param x the uint64_t integer to test
/// @return the number of bits that are set to 1
inline uint64_t ui64_bit_count(uint64_t x) { return std::popcount(x); }

/// @brief Compute the parity of a uint64_t integer (1 if odd number of bits set, -1 otherwise)
/// @param x the uint64_t integer to test
/// @return parity = (-1)^(number of bits set to 1)
inline double ui64_bit_parity(uint64_t x) { return 1.0 - 2.0 * (std::popcount(x) & 1); }

/// @brief Bit-scan to find the first set bit (least significant bit)
/// @param x the uint64_t integer to test
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_lowest_one_bit(uint64_t x) {
    return (x == 0) ? ~uint64_t(0) : std::countr_zero(x);
}

/// @brief Bit-scan to find the last set bit (most significant bit)
/// @param x the uint64_t integer to test
/// @return the index of the least significant 1-bit of x, or if x is ~0, returns ~0
inline uint64_t ui64_find_highest_one_bit(uint64_t x) {
    return (x == ~uint64_t(0)) ? x : 63 - std::countl_zero(x);
}

/// @brief Bit-scan to find next set bit after position pos
/// @param x the uint64_t integer to test
/// @param pos the position where we should start scanning (must be less than 64)
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_lowest_one_bit_at_pos(uint64_t x, int pos) {
    const uint64_t mask = uint64_t(~0) << pos; // set all bits to one and shift left
    x &= mask;
    return ui64_find_lowest_one_bit(x);
}

/// @brief Clear the lowest bit set in a uint64_t word
/// @param x the uint64_t word
/// @return a modified version of x with the lowest bit set to 1 turned into a 0
inline uint64_t ui64_clear_lowest_one_bit(uint64_t x) { return x & (x - 1); }

/// @brief Find the index of the lowest bit set in a uint64_t word and clear it. A modified version
///        of x with the lowest bit set to 1 turned into a 0 is stored in x
/// @param x the uint64_t integer to test. This value is modified by the function
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_and_clear_lowest_one_bit(uint64_t& x) {
    uint64_t result = ui64_find_lowest_one_bit(x);
    x = ui64_clear_lowest_one_bit(x);
    return result;
}

/// @brief Count the number of 1's from position 0 up to n - 1 and return the parity of this number.
/// @param x the uint64_t integer to test
/// @param n the end position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position 0 and n-1)
inline double ui64_sign(uint64_t x, int n) {
    constexpr unsigned int digits = std::numeric_limits<uint64_t>::digits;
    // This implementation is 20 x times faster than one based on for loops
    // First build a mask with n bit set. Then & with string and count.
    // Example for 16 bit string
    //                 n            (n = 5)
    // want       11111000 00000000
    // mask       11111111 11111111
    // mask >> 11 11111000 00000000  16 - n = 16 - 5 = 11
    //
    // Note: This strategy does not work when n = 0 because ~0 >> 64 = ~0 (!!!)
    // so we treat this case separately
    if (n == 0)
        return 1.0;
    return 1.0 - 2.0 * (ui64_bit_count(x & (~0ULL >> (digits - n))) & 1);
    // uint64_t mask = ~0;
    // // mask = mask << (64 - n);             // make a string with n bits set
    // mask = mask >> (digits - n);   // move it right by n
    // mask = x & mask;               // intersect with string
    // mask = ui64_bit_count(mask);   // count bits in between
    // return 1.0 - 2.0 * (mask & 1); // compute sign
}

/// @brief Count the number of 1's from position m + 1 up to n - 1 and return the parity of this
/// number.
/// @param x the uint64_t integer to test
/// @param m the starting position (not counted)
/// @param n the end position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position m+1 and
/// n-1)
inline double ui64_sign(uint64_t x, int m, int n) {
    // TODO PERF: speedup by avoiding the mask altogether
    // This implementation is a bit faster than one based on for loops
    // First build a mask with bit set between positions m and n. Then & with string and count.
    // Example for 16 bit string
    //              m  n            (m = 2, n = 5)
    // want       00011000 00000000
    // mask       11111111 11111111
    // mask << 14 00000000 00000011 14 = 16 + 1 - |5-2|
    // mask >> 11 00011000 00000000 11 = 16 - |5-2| -  min(2,5)
    // the mask should have |m - n| - 1 bits turned on
    uint64_t gap = std::abs(m - n);
    if (gap < 2) { // special cases
        return 1.0;
    }
    uint64_t mask = ~0;
    mask = mask << (65 - gap);                  // make a string with |m - n| - 1 bits set
    mask = mask >> (64 - gap - std::min(m, n)); // move it right after min(m, n)
    mask = x & mask;                            // intersect with string
    mask = ui64_bit_count(mask);                // count bits in between
    return (mask % 2 == 0) ? 1.0 : -1.0;        // compute sign
}

/// @brief Count the number of 1's from position n + 1 up to 63 and return the parity of this
/// number.
/// @param x the uint64_t integer to test
/// @param n the start position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position n+1 and
/// 63)
inline double ui64_sign_reverse(uint64_t x, int n) {
    // TODO PERF: speedup by avoiding the mask altogether
    // This implementation is 20 x times faster than one based on for loops
    // First build a mask with n bit set. Then & with string and count.
    // Example for 16 bit string
    //                 n            (n = 5)
    // want       00000011 11111111
    // mask       11111111 11111111
    // mask << 6  00000011 11111111 6 = 5 + 1
    //
    // Note: This strategy does not work when n = 0 because ~0 << 64 = ~0 (!!!)
    // so we treat this case separately
    if (n == 63)
        return 1.0;
    uint64_t mask = ~0;
    mask = mask << (n + 1);        // make a string with 64 - n - 1 bits set
    mask = x & mask;               // intersect with string
    mask = ui64_bit_count(mask);   // count bits in between
    return 1.0 - 2.0 * (mask & 1); // compute sign
}

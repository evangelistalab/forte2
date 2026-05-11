#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

inline constexpr uint64_t ui64_bit_not_found = ~uint64_t(0);

/// @brief Hash function for combining two size_t values.
/// @param a The first value.
/// @param b The second value.
/// @return A combined hash value.
inline std::size_t hash_combine(std::size_t a, std::size_t b) noexcept {
    static_assert(std::numeric_limits<std::size_t>::digits == 64,
                  "The function forte2::hash_combine requires size_t to be a 64-bit integer.");
    constexpr std::size_t C = 0x9e3779b97f4a7c15ull;
    std::size_t t = a * C + b;
    t ^= t >> 33;
    return t;
}

/// @brief Compute the parity of a uint64_t integer (1 if odd number of bits set, -1 otherwise)
/// @param x the uint64_t integer to test
/// @return parity = (-1)^(number of bits set to 1)
inline double ui64_bit_parity(uint64_t x) noexcept { return 1.0 - 2.0 * (std::popcount(x) & 1); }

/// @brief Compute the exclusive suffix XOR scan of a uint64_t word
/// @param x the uint64_t word
/// @return bit n is the XOR of bits n + 1 through 63 of x
inline uint64_t ui64_exclusive_suffix_xor(uint64_t x) noexcept {
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;
    x ^= x >> 16;
    x ^= x >> 32;
    return x >> 1;
}

/// @brief Bit-scan to find the first set bit (least significant bit)
/// @param x the uint64_t integer to test
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_lowest_one_bit(uint64_t x) noexcept {
    return (x == 0) ? ui64_bit_not_found : std::countr_zero(x);
}

/// @brief Bit-scan to find the last set bit (most significant bit)
/// @param x the uint64_t integer to test
/// @return the index of the most significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_highest_one_bit(uint64_t x) noexcept {
    return (x == 0) ? ui64_bit_not_found : 63 - std::countl_zero(x);
}

/// @brief Bit-scan to find next set bit after position pos
/// @param x the uint64_t integer to test
/// @param pos the position where we should start scanning (must be less than 64)
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_lowest_one_bit_at_pos(uint64_t x, int pos) noexcept {
    const uint64_t mask = uint64_t(~0) << pos; // set all bits to one and shift left
    x &= mask;
    return ui64_find_lowest_one_bit(x);
}

/// @brief Clear the lowest bit set in a uint64_t word
/// @param x the uint64_t word
/// @return a modified version of x with the lowest bit set to 1 turned into a 0
inline uint64_t ui64_clear_lowest_one_bit(uint64_t x) noexcept { return x & (x - 1); }

/// @brief Find the index of the lowest bit set in a uint64_t word and clear it. A modified version
///        of x with the lowest bit set to 1 turned into a 0 is stored in x
/// @param x the uint64_t integer to test. This value is modified by the function
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_and_clear_lowest_one_bit(uint64_t& x) noexcept {
    uint64_t result = ui64_find_lowest_one_bit(x);
    x = ui64_clear_lowest_one_bit(x);
    return result;
}

/// @brief Count the number of 1's from position 0 up to n - 1 and return the parity of this number.
/// @param x the uint64_t integer to test
/// @param n the end position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position 0 and n-1)
inline double ui64_sign(uint64_t x, std::size_t n) noexcept {
    if (n == 0)
        return 1.0;
    return 1.0 - 2.0 * (std::popcount(x << (64 - n)) & 1);
}

/// @brief Count the number of 1's from position m + 1 up to n - 1 and return the parity of this
/// number.
/// @param x the uint64_t integer to test
/// @param m the starting position (not counted)
/// @param n the end position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position m+1 and
/// n-1)
inline double ui64_sign(uint64_t x, std::size_t m, std::size_t n) noexcept {
    if (n < m) {
        std::swap(m, n);
    }

    if (n - m <= 1) {
        return 1.0;
    }

    const std::size_t width = n - m - 1;
    return 1.0 - 2.0 * (std::popcount((x >> (m + 1)) << (64 - width)) & 1);
}

/// @brief Count the number of 1's from position n + 1 up to 63 and return the parity of this
/// number.
/// @param x the uint64_t integer to test
/// @param n the start position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position n+1 and
/// 63)
inline double ui64_sign_reverse(uint64_t x, std::size_t n) noexcept {
    if (n == 63)
        return 1.0;
    return 1.0 - 2.0 * (std::popcount(x >> (n + 1)) & 1);
}

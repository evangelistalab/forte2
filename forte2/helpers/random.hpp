#pragma once

#include <cstdint>

namespace forte2 {

/// @brief The splitmix64 finalizer/mixer: spreads the bits of a 64-bit integer
/// @param x The input value to mix
/// @return The mixed value
inline uint64_t splitmix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

} // namespace forte2

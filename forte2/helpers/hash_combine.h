#pragma once

namespace forte2 {

/// @brief A simple 64-bit has combine function
inline std::size_t hash_combine(uint64_t a, uint64_t b) noexcept {
    return a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
}

} // namespace forte2
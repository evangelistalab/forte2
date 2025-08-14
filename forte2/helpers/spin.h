#pragma once

namespace forte2 {

/// @brief Enumeration for spin states
enum class Spin { Alpha, Beta };

/// @brief A convenience function to check if a spin is alpha
/// @param s The spin state to check
/// @return True if the spin is alpha, false otherwise
[[nodiscard]] inline bool is_alpha(Spin s) noexcept { return s == Spin::Alpha; }

/// @brief A convenience function to check if a spin is beta
/// @param s The spin state to check
/// @return True if the spin is beta, false otherwise
[[nodiscard]] inline bool is_beta(Spin s) noexcept { return s == Spin::Beta; }

} // namespace forte2
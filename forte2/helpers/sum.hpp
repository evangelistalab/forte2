#pragma once

#include <numeric>

namespace forte2 {

namespace math {

template <typename Container> auto sum(const Container& c) -> typename Container::value_type {
    using T = typename Container::value_type;
    return std::accumulate(std::begin(c), std::end(c), T(0));
}

} // namespace math
} // namespace forte2
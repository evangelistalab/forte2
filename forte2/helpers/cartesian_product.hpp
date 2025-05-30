#pragma once

#include <numeric>

namespace forte2 {

namespace math {
/// Return the Cartesian product of the input vector<vector<T>>
/// https://stackoverflow.com/a/17050528/4101036
template <typename T>
std::vector<std::vector<T>> cartesian_product(const std::vector<std::vector<T>>& input) {
    std::vector<std::vector<T>> product{{}};

    for (const auto& vec : input) {
        std::vector<std::vector<T>> tmp;
        for (const auto& x : product) {
            for (const auto& y : vec) {
                tmp.push_back(x);
                tmp.back().push_back(y);
            }
        }
        product = std::move(tmp);
    }

    return product;
}

} // namespace math
} // namespace forte2

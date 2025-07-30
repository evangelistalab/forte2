#pragma once

#include <sstream>
#include <string>

namespace forte2 {
/// @brief Convert the contents of a container to a string representation
template <typename Container> std::string container_to_string(const Container& c) {
    if (c.empty()) {
        return "[]";
    }
    std::stringstream ss;
    ss << "[";
    for (const auto& item : c) {
        ss << item << ",";
    }
    std::string result = ss.str();
    if (!result.empty() and result.back() == ',') {
        result.pop_back(); // Remove the trailing comma
    }
    result += "]";
    return result;
}
} // namespace forte2